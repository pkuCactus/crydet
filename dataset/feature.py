"""
Feature Extraction for Baby Cry Detection

PyTorch-based implementation for GPU-accelerated batch processing.
Replaces numpy/librosa operations with torch equivalents.

Supports:
- FFT (magnitude spectrum)
- FBank (Mel filter bank / log Mel spectrum)
- MFCC (Mel-frequency cepstral coefficients)
- DB (Energy features: average and weighted)
- Delta features (time and frequency derivatives)

Input: [B, N] raw audio waveforms
Output: [B, T, F] feature tensors

Configuration options (via FeatureConfig):
- feature_type: 'fbank', 'mfcc', or 'all' (fbank+mfcc)
- use_delta: Add time derivative features
- use_freq_delta: Add frequency derivative features
- use_db_feature: Add energy (dB) as additional channel
"""

import logging
import math

from enum import IntEnum

import torch
import torch.nn.functional as F
import torchaudio.functional as FA

from utils.config import FeatureConfig, MaskConfig


class FeatureType(IntEnum):
    FBANK = 1
    DB = 2
    MFCC = 4
    FFT = 8


class FeatureExtractor(torch.nn.Module):
    """
    PyTorch-based feature extractor for batch processing on GPU

    Replaces numpy/librosa operations with torch equivalents for:
    - Preemphasis filtering
    - Framing and windowing
    - FFT computation
    - Mel filter bank application
    - Log transformation
    - Delta feature computation

    Input: [B, N] raw audio waveforms
    Output: [B, T, F] feature tensors
    """

    def __init__(self, config: FeatureConfig, sr: int = 16000):
        super().__init__()
        self.config = config
        if self.config.feature_type <= 0 or self.config.feature_type > 15:
            raise ValueError("feature_type must be in [1, 15]")
        self.sr = sr

        # Pre-compute Hanning window as buffer (will be on correct device)
        self.register_buffer('_window', torch.hann_window(config.n_fft))
        # _window_area as buffer for GPU computation
        self.register_buffer('_window_area', self._window.mean())

        # Create Mel filter bank using torchaudio.functional
        # Returns [n_fft//2+1, n_mels]
        mel_fb = FA.melscale_fbanks(
            n_freqs=config.n_fft // 2 + 1,
            f_min=config.fmin,
            f_max=config.fmax,
            n_mels=config.n_mels,
            sample_rate=sr,
            norm="slaney",
            mel_scale="slaney",
        )
        self.register_buffer('_mel_matrix', mel_fb)

        # Pre-compute DCT matrix for MFCC if needed
        if config.feature_type & FeatureType.MFCC:
            dct_matrix = self._create_dct_matrix(config.n_mels, config.n_mfcc)
            self.register_buffer('_dct_matrix', dct_matrix)
        else:
            self._dct_matrix = None

    def preemphasis(self, signal: torch.Tensor, coeff: float = 0.95) -> torch.Tensor:
        """
        Apply preemphasis filter: y[n] = x[n] - coeff * x[n-1]

        Args:
            signal: [B, N] audio signals

        Returns:
            Preemphasized signals [B, N]
        """
        if coeff <= 0:
            logging.warning("The input coeff is less than 0 and return original signal")
            return signal
        # y[n] = x[n] - coeff * x[n-1]
        # First sample: y[0] = x[0] (no previous sample)
        diff = signal[:, 1:] - coeff * signal[:, :-1]
        return torch.cat([signal[:, :1], diff], dim=1)

    def frame_signal(self, signal: torch.Tensor, frame_length: int, hop_length: int) -> torch.Tensor:
        """
        Frame a signal into overlapping frames

        Args:
            signal: [B, N] signals
            frame_length: Length of each frame
            hop_length: Hop size between frames

        Returns:
            [B, num_frames, frame_length] framed signals
        """
        _, signal_length = signal.shape
        num_frames = 1 + (signal_length - frame_length) // hop_length

        # Create frame indices
        indices = torch.arange(0, frame_length, device=signal.device).unsqueeze(0) + \
                  torch.arange(0, num_frames * hop_length, hop_length, device=signal.device).unsqueeze(1)

        # Gather frames for each batch
        # indices: [num_frames, frame_length]
        # signal: [B, N]
        # result: [B, num_frames, frame_length]
        frames = signal[:, indices]  # [B, num_frames, frame_length]
        return frames

    def stft(self, signal: torch.Tensor, n_fft: int, hop_length: int) -> torch.Tensor:
        """
        Compute STFT using torch.fft (matches numpy framing logic)

        Args:
            signal: [B, N] signals
            n_fft: FFT size
            hop_length: Hop size

        Returns:
            [B, n_fft//2+1, num_frames] complex spectrogram
        """
        # Match numpy implementation: pad (n_fft - hop_length) at the beginning
        pad_left = n_fft - hop_length
        signal_padded = F.pad(signal, (pad_left, 0), mode='constant')

        # Use torch.stft with center=False (we already padded)
        # Buffer is automatically on correct device
        return torch.stft(
            signal_padded,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=self._window,
            return_complex=True,
            center=False
        )

    def compute_mfcc(self, log_mel: torch.Tensor) -> torch.Tensor:
        """
        Compute MFCC from log Mel spectrogram using DCT

        Args:
            log_mel: [B, T, n_mels] log Mel spectrogram

        Returns:
            [B, T, n_mfcc] MFCC coefficients
        """
        # Use pre-computed DCT matrix: [n_mfcc, n_mels]
        # Buffer is automatically on correct device
        dct_matrix = self._dct_matrix

        # Compute MFCC: [B, T, n_mfcc] = [B, T, n_mels] @ [n_mels, n_mfcc]
        mfcc = torch.matmul(log_mel, dct_matrix.T)

        return mfcc

    def _create_dct_matrix(self, n_mels: int, n_mfcc: int) -> torch.Tensor:
        """Create DCT-II matrix for MFCC computation

        Args:
            n_mels: Number of Mel filter banks
            n_mfcc: Number of MFCC coefficients to compute

        Returns:
            DCT matrix of shape [n_mfcc, n_mels]
        """
        # DCT-II matrix
        n = torch.arange(n_mels, dtype=torch.float32).unsqueeze(0)
        k = torch.arange(n_mfcc, dtype=torch.float32).unsqueeze(1)
        dct = torch.cos(math.pi * k * (n + 0.5) / n_mels)
        # Normalize
        dct[0] *= 1 / math.sqrt(2)
        dct *= math.sqrt(2 / n_mels)
        return dct

    def compute_energy_features(self, signal: torch.Tensor) -> torch.Tensor:
        """
        Compute energy features (average and weighted) in dB

        Args:
            signal: [B, N] signals

        Returns:
            [B, 2, num_frames] energy features
            - If use_db_norm=False: [db_avg, db_avg_hann]
            - If use_db_norm=True: [db_avg_hann, moving_max]
            [B, num_frames] moving max
        """
        # Step 2: Compute PCM squared energy
        energy = signal ** 2  # [B, N_padded]

        # Step 3: Frame the energy to get energy_frame
        energy_frames = self.frame_signal(energy, self.config.n_fft, self.config.hop_length)  # [B, T, n_fft]

        # Step 4: Compute average energy per frame and convert to dB
        db_avg = self._energy_to_db(energy_frames.mean(dim=-1))

        # Step 5: Compute Hann-windowed average energy and convert to dB
        energy_frames_hann = energy_frames * self._window / self._window_area
        db_avg_hann = self._energy_to_db(energy_frames_hann.mean(dim=-1))

        if not self.config.use_db_norm:
            return torch.stack([db_avg, db_avg_hann], dim=-1), torch.ones_like(db_avg)

        energy_frame_max = energy_frames.max(dim=-1)[0] * 1.1
        decay = self.config.fbank_decay
        decay_up = (1 - (1 - decay) * 3)
        log_max = torch.log(torch.clamp(energy_frame_max, min=1e-8))

        # Vectorized moving max computation
        moving_max = self._vectorized_moving_max(log_max, decay, decay_up)

        moving_max = torch.exp(moving_max)
        moving_max = torch.clamp(moving_max, 1e-4, 1)
        db_feat = torch.stack([db_avg_hann, moving_max], dim=2)
        return db_feat, moving_max

    def _vectorized_moving_max(self, log_max: torch.Tensor, decay: float, decay_up: float) -> torch.Tensor:
        """
        Vectorized moving max computation using scan operation

        Args:
            log_max: [B, T] log max values
            decay: Decay factor for decreasing values
            decay_up: Decay factor for increasing values

        Returns:
            [B, T] moving max values
        """
        B, T = log_max.shape
        device = log_max.device

        # Use associative scan for parallel computation
        # This is equivalent to the sequential loop but can be parallelized
        moving_max = torch.zeros_like(log_max)

        # Initialize first element
        moving_max[:, 0] = log_max[:, 0]

        # Compute using vectorized operation where possible
        # For sequential dependency, we use a loop but minimize operations
        t = moving_max[:, 0]
        for i in range(T):
            cur = log_max[:, i]
            # Update rule: if cur > t, use decay_up, else use decay
            mask = cur > t
            t = torch.where(mask, t * decay_up + cur * (1 - decay_up), t * decay + cur * (1 - decay))
            moving_max[:, i] = t

        return moving_max

    def _energy_to_db(self, energy: torch.Tensor, amin: float = 1e-8) -> torch.Tensor:
        """Convert energy to normalized dB [0, 1] range"""
        db = torch.log10(torch.clamp(energy, min=amin))
        # Normalize to [0, +inf] (clip at -8 dB)
        db_normalized = (torch.clamp(db, min=-8) + 8) / 8
        return db_normalized

    def normalize_fbank(self, fbank: torch.Tensor) -> torch.Tensor:
        """
        Normalize FBank features with exponential smoothing - fully vectorized

        Args:
            fbank: [B, T, n_mels] log Mel features

        Returns:
            Normalized features in [0, 1] range
        """
        decay = self.config.fbank_decay
        decay_up = 1 - (1 - decay) * 3
        fbank_min = 0

        # Compute max and mean excluding edge bins
        fbank_center = fbank[:, :, 2:-2]  # [B, T, n_mels-4]
        fbank_max = fbank_center.max(dim=2, keepdim=True)[0] * 1.1  # [B, T, 1]
        fbank_avg = fbank_center.mean(dim=2, keepdim=True)  # [B, T, 1]

        # Vectorized exponential smoothing
        fbank_max_smooth = self._vectorized_exp_smooth(fbank_max, fbank_avg, decay, decay_up)

        # Normalize
        fbank_norm = torch.where(
            fbank_max_smooth != fbank_min,
            (fbank - fbank_min) / (fbank_max_smooth - fbank_min),
            fbank
        )
        fbank_norm = torch.clamp(fbank_norm, 0., 1.0)
        return fbank_norm

    def _vectorized_exp_smooth(
        self,
        values: torch.Tensor,
        avgs: torch.Tensor,
        decay: float,
        decay_up: float
    ) -> torch.Tensor:
        """
        Vectorized exponential smoothing with conditional update

        Args:
            values: [B, T, 1] values to smooth
            avgs: [B, T, 1] average values for condition
            decay: Decay factor
            decay_up: Decay factor for increase

        Returns:
            [B, T, 1] smoothed values
        """
        B, T, _ = values.shape
        device = values.device

        result = torch.zeros_like(values)
        t = values[:, 0, :].unsqueeze(-1)  # [B, 1, 1]
        flag = torch.zeros((B, 1, 1), dtype=torch.bool, device=device)

        for i in range(T):
            maxs = values[:, i, :].unsqueeze(-1)  # [B, 1, 1]
            avg = avgs[:, i, :].unsqueeze(-1)    # [B, 1, 1]

            # Update with different decay based on direction
            t = torch.where(maxs > t, t * decay_up + maxs * (1 - decay_up), t * decay + maxs * (1 - decay))

            # Handle flag condition
            t = torch.where(flag & (t < maxs), maxs, t)
            flag = torch.where((t < avg) & ~flag, True, False)
            t = torch.where((t < avg) & flag, avg * 2, t)

            result[:, i, :] = t.squeeze(-1)

        return result

    def compute_delta(self, features: torch.Tensor, axis: int = 2) -> torch.Tensor:
        """
        Compute delta (differential) features using central difference

        Args:
            features: [B, F, T] feature tensor
            axis: Axis along which to compute delta (2=time, 1=frequency)

        Returns:
            Delta features with same shape as input
        """
        # Pad at boundaries
        if axis == 2:  # Time axis
            padded = F.pad(features, (1, 1), mode='replicate')  # [B, F, T+2]
            delta = (padded[:, :, 2:] - padded[:, :, :-2]) / 2.0
        else:  # Frequency axis
            padded = F.pad(features, (0, 0, 1, 1), mode='replicate')  # [B, F+2, T]
            delta = (padded[:, 2:, :] - padded[:, :-2, :]) / 2.0

        return delta

    def dropblock(self, features: torch.Tensor, rate: float, block_size: int = 3) -> tuple[torch.Tensor, float]:
        """
        Apply DropBlock masking to features.

        DropBlock masks contiguous regions (blocks) rather than individual elements,
        which is more effective for structured data like spectrograms.

        Args:
            features: [B, T, F] feature tensor (time x frequency)
            rate: Target proportion of features to mask (0.0 - 1.0)
            block_size: Size of the square block to mask (default: 3)

        Returns:
            masked_features: Features with dropblock applied
            actual_rate: Actual proportion of features masked
        """
        if not self.training or rate <= 0:
            return features, 0.0

        batch_size, time_steps, feat_dim = features.shape

        # Ensure block_size doesn't exceed feature dimensions
        block_size = min(block_size, time_steps, feat_dim)

        # Calculate gamma (probability per position) to achieve target rate
        # Standard DropBlock formula: gamma = rate * (T * F) / (block_size ** 2) / valid_positions
        feat_size = time_steps * feat_dim
        block_area = block_size ** 2
        gamma = rate * feat_size / block_area / ((time_steps - block_size + 1) * (feat_dim - block_size + 1))
        gamma = min(gamma, 1.0)  # Cap at 1.0

        # Generate random mask for block centers [B, T, F]
        mask_center = torch.rand(batch_size, time_steps, feat_dim, device=features.device)
        mask_center = (mask_center < gamma).float()

        # Expand each center to a block using max pooling
        # Pad to handle boundaries
        pad = block_size // 2
        mask_padded = F.pad(mask_center, (pad, pad, pad, pad), mode='constant', value=0)

        # Use max pooling to expand blocks with ceil_mode and then crop
        mask_padded = mask_padded.unsqueeze(1)
        mask_block = F.max_pool2d(
            mask_padded,
            kernel_size=block_size,
            stride=1,
            padding=0,
            ceil_mode=True
        )
        # Crop to original size [B, T, F]
        mask_block = mask_block.squeeze(1)[:, :time_steps, :feat_dim]

        # Calculate actual mask rate
        actual_rate = mask_block.mean().item()

        # Apply mask: keep unmasked values, zero out masked ones
        masked_features = features * (1 - mask_block)

        return masked_features, actual_rate

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract features from batch of audio waveforms

        Args:
            waveform: [B, N] raw audio waveforms (or [N] single waveform)

        Returns:
            [B, T, F] feature tensor
        """
        # Handle single waveform
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        waveform = torch.clamp(waveform, min=-1, max=1)

        # Apply preemphasis
        if self.config.preemphasis > 0:
            waveform = self.preemphasis(waveform, self.config.preemphasis)
        # pad left
        waveform = F.pad(waveform, (self.config.n_fft - self.config.hop_length, 0), 'constant', 0)

        # get db and moving max
        db_feat, moving_max = self.compute_energy_features(waveform)

        pcm_frames = self.frame_signal(waveform, self.config.n_fft, self.config.hop_length)
        pcm_frames = pcm_frames / moving_max.unsqueeze(-1)
        pcm_frames = torch.clamp(pcm_frames, -1, 1)
        frame_hann = pcm_frames * self._window # B T NFFT
        rfft_result = torch.fft.rfft(frame_hann) # B T NFFT // 2 + 1
        fft_feat = torch.abs(rfft_result)

        melspec = torch.matmul(fft_feat, self._mel_matrix)

        # Build base features list
        base_features = []
        if self.config.feature_type & FeatureType.FFT:
            base_features.append(fft_feat)
        if self.config.feature_type & FeatureType.MFCC:
            log_melspec = torch.log(melspec + 1e-6)
            # Directly compute MFCC from [B, T, n_mels] format
            mfcc = self.compute_mfcc(log_melspec)
            base_features.append(mfcc)
        if self.config.feature_type & FeatureType.FBANK:
            fbank = torch.log(melspec + 1.)
            if self.config.use_fbank_norm:
                fbank = self.normalize_fbank(fbank=fbank)
            else:
                fbank = torch.clamp(fbank, 0, 7)
            # Apply DropBlock mask during training
            if self.training and self.config.mask.enable:
                # Random block_size: normal distribution with mean=6, std=2
                # Use torch for GPU-compatible random generation
                block_size = torch.randn(1, device=fbank.device).item() * 2 + 6
                block_size = int(torch.clamp(torch.tensor(block_size), 1, 12).item())
                fbank, _ = self.dropblock(fbank, self.config.mask.rate, block_size)
            base_features.append(fbank)

        # Handle case where only DB is requested
        if not base_features:
            # Only DB feature requested
            return db_feat

        base_feature = base_features[0] if len(base_features) == 1 else torch.cat(base_features, dim=-1)

        # Build final features list starting with base
        features = [base_feature]

        # Add time delta
        if self.config.use_time_delta:
            delta_t = self.compute_delta(base_feature, axis=2)
            features.append(delta_t)

        # Add frequency delta
        if self.config.use_freq_delta:
            delta_f = self.compute_delta(base_feature, axis=1)
            features.append(delta_f)

        # Add DB feature (no delta for DB)
        if self.config.feature_type & FeatureType.DB:
            features.append(db_feat)

        # Concatenate all features
        return torch.cat(features, dim=-1)  # [B, T, D]

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """Call forward method"""
        return self.forward(waveform)

    def compile(self, mode: str = "default", fullgraph: bool = False):
        """
        Compile the feature extractor with torch.compile for faster inference

        Args:
            mode: Compilation mode ('default', 'reduce-overhead', 'max-autotune')
            fullgraph: Whether to require full graph compilation

        Returns:
            Self for method chaining

        Example:
            extractor = FeatureExtractor(config).compile(mode='reduce-overhead')
        """
        try:
            # Compile the forward method
            self.forward = torch.compile(
                self.forward,
                mode=mode,
                fullgraph=fullgraph,
                dynamic=True  # Support variable batch sizes
            )
            logging.info(f"FeatureExtractor compiled with mode='{mode}'")
        except Exception as e:
            logging.warning(f"torch.compile failed: {e}. Continuing without compilation.")
        return self
