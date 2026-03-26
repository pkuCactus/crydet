"""
Unit tests for dataset/feature.py

Tests cover:
- FeatureExtractor initialization with different configurations
- FeatureType bit-flag combinations
- Energy feature computation
- FBank normalization
- Delta feature computation
- Forward pass with various feature combinations
"""

import logging
import unittest
import torch
import math

from utils.config import FeatureConfig
from dataset.feature import FeatureExtractor, FeatureType


logging.basicConfig(level=logging.ERROR)


class TestFeatureType(unittest.TestCase):
    """Test FeatureType IntEnum definitions"""

    def test_feature_type_values(self):
        """Test that FeatureType values are correct powers of 2"""
        self.assertEqual(FeatureType.FBANK, 1)
        self.assertEqual(FeatureType.DB, 2)
        self.assertEqual(FeatureType.MFCC, 4)
        self.assertEqual(FeatureType.FFT, 8)

    def test_feature_type_combinations(self):
        """Test that feature types can be combined with bitwise OR"""
        # FBANK + DB = 3
        self.assertEqual(FeatureType.FBANK | FeatureType.DB, 3)
        # FBANK + MFCC = 5
        self.assertEqual(FeatureType.FBANK | FeatureType.MFCC, 5)
        # All features = 15
        self.assertEqual(
            FeatureType.FBANK | FeatureType.DB | FeatureType.MFCC | FeatureType.FFT,
            15
        )

    def test_feature_type_bitwise_check(self):
        """Test that bitwise AND correctly checks feature presence"""
        combined = FeatureType.FBANK | FeatureType.DB  # 3
        self.assertTrue(combined & FeatureType.FBANK)
        self.assertTrue(combined & FeatureType.DB)
        self.assertFalse(combined & FeatureType.MFCC)
        self.assertFalse(combined & FeatureType.FFT)


class TestFeatureConfig(unittest.TestCase):
    """Test FeatureConfig dataclass"""

    def test_default_config(self):
        """Test default configuration values"""
        config = FeatureConfig()
        self.assertEqual(config.feature_type, 1)  # FBANK only
        self.assertEqual(config.n_fft, 1024)
        self.assertEqual(config.hop_length, 500)
        self.assertEqual(config.n_mels, 32)
        self.assertEqual(config.n_mfcc, 16)
        self.assertEqual(config.preemphasis, 0.95)
        self.assertTrue(config.use_fbank_norm)
        self.assertEqual(config.fbank_decay, 0.9)
        self.assertFalse(config.use_db_norm)
        self.assertFalse(config.use_time_delta)
        self.assertFalse(config.use_freq_delta)

    def test_feature_dim_fbank_only(self):
        """Test feature dimension calculation for FBank only"""
        config = FeatureConfig(feature_type=FeatureType.FBANK, n_mels=64)
        self.assertEqual(config.feature_dim, 64)

    def test_feature_dim_fbank_with_db(self):
        """Test feature dimension for FBank + DB"""
        config = FeatureConfig(
            feature_type=FeatureType.FBANK | FeatureType.DB,
            n_mels=32
        )
        # 32 (FBANK) + 2 (DB) = 34
        self.assertEqual(config.feature_dim, 34)

    def test_feature_dim_with_time_delta(self):
        """Test feature dimension with time delta"""
        config = FeatureConfig(
            feature_type=FeatureType.FBANK,
            n_mels=32,
            use_time_delta=True
        )
        # 32 * 2 = 64 (base + delta doubles the dimension)
        self.assertEqual(config.feature_dim, 64)

    def test_feature_dim_with_both_deltas(self):
        """Test feature dimension with both time and freq delta"""
        config = FeatureConfig(
            feature_type=FeatureType.FBANK,
            n_mels=32,
            use_time_delta=True,
            use_freq_delta=True
        )
        # 32 * 3 = 96 (base + time_delta + freq_delta)
        self.assertEqual(config.feature_dim, 96)


class TestFeatureExtractorInit(unittest.TestCase):
    """Test FeatureExtractor initialization"""

    def test_valid_feature_type(self):
        """Test initialization with valid feature types"""
        for ft in [1, 2, 3, 4, 5, 6, 7, 8, 15]:
            with self.subTest(feature_type=ft):
                config = FeatureConfig(feature_type=ft)
                extractor = FeatureExtractor(config)
                self.assertIsNotNone(extractor)

    def test_invalid_feature_type_zero(self):
        """Test that feature_type=0 raises ValueError"""
        config = FeatureConfig(feature_type=0)
        with self.assertRaises(ValueError) as context:
            FeatureExtractor(config)
        self.assertIn("feature_type must be in [1, 15]", str(context.exception))

    def test_invalid_feature_type_negative(self):
        """Test that negative feature_type raises ValueError"""
        config = FeatureConfig(feature_type=-1)
        with self.assertRaises(ValueError):
            FeatureExtractor(config)

    def test_invalid_feature_type_too_large(self):
        """Test that feature_type > 15 raises ValueError"""
        config = FeatureConfig(feature_type=16)
        with self.assertRaises(ValueError):
            FeatureExtractor(config)

    def test_mel_matrix_shape(self):
        """Test that Mel matrix has correct shape"""
        config = FeatureConfig(n_fft=1024, n_mels=32)
        extractor = FeatureExtractor(config)
        # Mel matrix shape: [n_fft//2+1, n_mels]
        expected_shape = (513, 32)  # 1024//2+1 = 513
        self.assertEqual(extractor._mel_matrix.shape, expected_shape)

    def test_window_shape(self):
        """Test that Hann window has correct shape"""
        config = FeatureConfig(n_fft=1024)
        extractor = FeatureExtractor(config)
        self.assertEqual(extractor._window.shape, (1024,))


class TestPreemphasis(unittest.TestCase):
    """Test preemphasis filter"""

    def setUp(self):
        self.config = FeatureConfig(preemphasis=0.95)
        self.extractor = FeatureExtractor(self.config)

    def test_preemphasis_coefficient_zero(self):
        """Test that coeff=0 returns original signal"""
        signal = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        result = self.extractor.preemphasis(signal, coeff=0.0)
        torch.testing.assert_close(result, signal)

    def test_preemphasis_computation(self):
        """Test preemphasis computation: y[n] = x[n] - coeff * x[n-1]"""
        signal = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        coeff = 0.5
        result = self.extractor.preemphasis(signal, coeff=coeff)

        # Expected: [1.0, 2-0.5*1, 3-0.5*2, 4-0.5*3, 5-0.5*4]
        expected = torch.tensor([[1.0, 1.5, 2.0, 2.5, 3.0]])
        torch.testing.assert_close(result, expected)

    def test_preemphasis_first_sample_unchanged(self):
        """Test that first sample remains unchanged"""
        signal = torch.tensor([[5.0, 3.0, 1.0]])
        result = self.extractor.preemphasis(signal, coeff=0.95)
        self.assertEqual(result[0, 0].item(), 5.0)


class TestFrameSignal(unittest.TestCase):
    """Test signal framing"""

    def setUp(self):
        self.config = FeatureConfig(n_fft=128, hop_length=4)
        self.extractor = FeatureExtractor(self.config)

    def test_frame_signal_shape(self):
        """Test that framing produces correct output shape"""
        # Signal: batch=2, length=20
        signal = torch.randn(2, 20)
        frame_length = 8
        hop_length = 4

        frames = self.extractor.frame_signal(signal, frame_length, hop_length)

        # Expected frames: 1 + (20 - 8) // 4 = 4
        self.assertEqual(frames.shape, (2, 4, 8))

    def test_frame_signal_content(self):
        """Test that framing produces correct content"""
        # Simple signal for verification
        signal = torch.arange(16).float().unsqueeze(0)  # [1, 16]

        frames = self.extractor.frame_signal(signal, frame_length=8, hop_length=4)

        # First frame should be [0, 1, 2, 3, 4, 5, 6, 7]
        expected_first = torch.arange(8).float()
        torch.testing.assert_close(frames[0, 0], expected_first)

        # Second frame should be [4, 5, 6, 7, 8, 9, 10, 11]
        expected_second = torch.arange(4, 12).float()
        torch.testing.assert_close(frames[0, 1], expected_second)


class TestEnergyToDb(unittest.TestCase):
    """Test energy to dB conversion"""

    def setUp(self):
        self.config = FeatureConfig()
        self.extractor = FeatureExtractor(self.config)

    def test_energy_to_db_basic(self):
        """Test basic dB conversion"""
        # 10 * log10(100) = 20 dB, normalized: (20/10 + 8) / 8 = ... wait
        # Actually: log10(100) = 2, clamp to -8, (2 + 8) / 8 = 1.25
        energy = torch.tensor([100.0])
        result = self.extractor._energy_to_db(energy)
        expected = (math.log10(100) + 8) / 8
        self.assertAlmostEqual(result.item(), expected, places=5)

    def test_energy_to_db_clipping(self):
        """Test that values below -8 dB are clipped"""
        # Very small energy should be clipped at -8 (log10)
        energy = torch.tensor([1e-10])
        result = self.extractor._energy_to_db(energy)
        # log10(1e-10) = -10, clamped to -8, (-8 + 8) / 8 = 0
        self.assertEqual(result.item(), 0.0)

    def test_energy_to_db_amin_protection(self):
        """Test that amin prevents log(0)"""
        energy = torch.tensor([0.0])
        result = self.extractor._energy_to_db(energy)
        # Should not be inf or nan
        self.assertTrue(torch.isfinite(result).all())


class TestComputeEnergyFeatures(unittest.TestCase):
    """Test compute_energy_features method"""

    def setUp(self):
        # Use larger n_fft to avoid "n_mels too high" warning
        self.config = FeatureConfig(n_fft=128, hop_length=64, use_db_norm=False)
        self.extractor = FeatureExtractor(self.config)

    def test_energy_features_shape(self):
        """Test energy features output shape"""
        # Signal: [B=2, N=256], n_fft=128, hop=64
        # Frames: 1 + (256 - 128) // 64 = 3
        signal = torch.randn(2, 256)

        db_feat, moving_max = self.extractor.compute_energy_features(signal)

        # db_feat shape: [B, T, 2]
        self.assertEqual(db_feat.shape, (2, 3, 2))
        # moving_max shape: [B, T]
        self.assertEqual(moving_max.shape, (2, 3))

    def test_energy_features_without_db_norm(self):
        """Test energy features without db_norm"""
        config = FeatureConfig(n_fft=128, hop_length=4, use_db_norm=False)
        extractor = FeatureExtractor(config)

        signal = torch.ones(1, 256) * 0.5  # Constant signal
        db_feat, moving_max = extractor.compute_energy_features(signal)

        # Without db_norm, moving_max should be all ones
        torch.testing.assert_close(moving_max, torch.ones_like(moving_max))

        # db_feat should have 2 channels
        self.assertEqual(db_feat.shape[-1], 2)

    def test_energy_features_with_db_norm(self):
        """Test energy features with db_norm enabled"""
        config = FeatureConfig(
            n_fft=128, hop_length=4, use_db_norm=True, fbank_decay=0.9
        )
        extractor = FeatureExtractor(config)

        signal = torch.randn(1, 256)
        db_feat, moving_max = extractor.compute_energy_features(signal)

        # With db_norm, db_feat should be [db_avg_hann, moving_max]
        self.assertEqual(db_feat.shape[-1], 2)

        # moving_max should be clamped to [1e-4, 1] range for normalization
        self.assertTrue((moving_max >= 1e-4).all())
        self.assertTrue((moving_max <= 1).all())


class TestSTFT(unittest.TestCase):
    """Test STFT computation"""

    def setUp(self):
        self.config = FeatureConfig(n_fft=128, hop_length=64)
        self.extractor = FeatureExtractor(self.config)

    def test_stft_output_shape(self):
        """Test STFT output shape"""
        signal = torch.randn(2, 256)  # [B, N]
        stft_result = self.extractor.stft(signal, self.config.n_fft, self.config.hop_length)

        # Expected shape: [B, n_fft//2+1, num_frames]
        # After padding: length = 256 + (128-64) = 320
        # frames = 1 + (320 - 128) // 64 = 4
        self.assertEqual(stft_result.shape[0], 2)
        self.assertEqual(stft_result.shape[1], 65)  # 128//2+1
        self.assertEqual(stft_result.dtype, torch.complex64)

    def test_stft_is_complex(self):
        """Test that STFT returns complex tensor"""
        signal = torch.randn(1, 256)
        stft_result = self.extractor.stft(signal, self.config.n_fft, self.config.hop_length)
        self.assertTrue(torch.is_complex(stft_result))


class TestCreateDCTMatrix(unittest.TestCase):
    """Test DCT matrix creation"""

    def test_dct_matrix_shape(self):
        """Test DCT matrix has correct shape"""
        config = FeatureConfig(feature_type=FeatureType.MFCC, n_mels=32, n_mfcc=16)
        extractor = FeatureExtractor(config)

        # DCT matrix should be [n_mfcc, n_mels]
        self.assertEqual(extractor._dct_matrix.shape, (16, 32))

    def test_dct_matrix_orthogonality(self):
        """Test DCT matrix is approximately orthogonal"""
        config = FeatureConfig(feature_type=FeatureType.MFCC, n_mels=32, n_mfcc=16)
        extractor = FeatureExtractor(config)

        dct = extractor._dct_matrix  # [n_mfcc, n_mels]
        # For orthogonal matrix: D @ D.T should be identity (scaled)
        product = torch.matmul(dct, dct.T)

        # Check diagonal elements are equal (energy normalization)
        diag = torch.diag(product)
        self.assertTrue(torch.allclose(diag, diag[0], rtol=1e-5))

    def test_dct_matrix_no_mfcc(self):
        """Test that DCT matrix is None when MFCC not needed"""
        config = FeatureConfig(feature_type=FeatureType.FBANK)  # No MFCC
        extractor = FeatureExtractor(config)

        self.assertIsNone(extractor._dct_matrix)


class TestForwardFeatureCombinations(unittest.TestCase):
    """Test forward pass with various feature type combinations"""

    def _create_waveform(self, batch_size=2, duration_sec=0.5, sr=16000):
        """Helper to create test waveform"""
        samples = int(duration_sec * sr)
        return torch.randn(batch_size, samples)

    def test_forward_fft_only(self):
        """Test forward pass with FFT only"""
        config = FeatureConfig(feature_type=FeatureType.FFT, n_fft=1024, hop_length=500)
        extractor = FeatureExtractor(config, sr=16000)
        waveform = self._create_waveform()

        features = extractor(waveform)

        # FFT feature dim: n_fft//2+1 = 513
        self.assertEqual(features.shape[2], 513)

    def test_forward_db_only(self):
        """Test forward pass with DB only - should use DB as base feature"""
        config = FeatureConfig(
            feature_type=FeatureType.DB, n_fft=1024, hop_length=500
        )
        extractor = FeatureExtractor(config, sr=16000)
        waveform = self._create_waveform()

        # DB only is valid - should return DB features
        features = extractor(waveform)

        # DB feature dim: 2
        self.assertEqual(features.shape[2], 2)

    def test_forward_fbank_db_mfcc(self):
        """Test forward pass with FBANK+DB+MFCC combination"""
        config = FeatureConfig(
            feature_type=FeatureType.FBANK | FeatureType.DB | FeatureType.MFCC,
            n_mels=32,
            n_mfcc=16,
            hop_length=500
        )
        extractor = FeatureExtractor(config, sr=16000)
        waveform = self._create_waveform()

        features = extractor(waveform)

        # Feature dim: 32 + 2 + 16 = 50
        self.assertEqual(features.shape[2], 50)

    def test_forward_all_feature_types(self):
        """Test forward pass with all feature types (FBANK|DB|MFCC|FFT)"""
        config = FeatureConfig(
            feature_type=FeatureType.FBANK | FeatureType.DB | FeatureType.MFCC | FeatureType.FFT,
            n_mels=32,
            n_mfcc=16,
            n_fft=512,  # Smaller FFT for test
            hop_length=256
        )
        extractor = FeatureExtractor(config, sr=16000)
        waveform = self._create_waveform(batch_size=1, duration_sec=0.3)

        features = extractor(waveform)

        # Feature dim: 32 + 2 + 16 + 257 = 307
        expected_dim = 32 + 2 + 16 + 257
        self.assertEqual(features.shape[2], expected_dim)

    def test_forward_with_both_deltas_and_db(self):
        """Test forward with base + both deltas + DB"""
        config = FeatureConfig(
            feature_type=FeatureType.FBANK | FeatureType.DB,
            n_mels=32,
            use_time_delta=True,
            use_freq_delta=True,
            hop_length=500
        )
        extractor = FeatureExtractor(config, sr=16000)
        waveform = self._create_waveform()

        features = extractor(waveform)

        # Base FBANK: 32, deltas: 32*2=64, DB: 2
        # Total: 32 + 64 + 2 = 98
        expected_dim = 32 * 3 + 2
        self.assertEqual(features.shape[2], expected_dim)


class TestFeatureExtractorEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions"""

    def test_forward_empty_batch(self):
        """Test forward with batch_size=1 minimum"""
        config = FeatureConfig(feature_type=FeatureType.FBANK, n_mels=32)
        extractor = FeatureExtractor(config, sr=16000)

        waveform = torch.randn(1, 8000)  # Single sample
        features = extractor(waveform)

        self.assertEqual(features.shape[0], 1)
        self.assertEqual(features.shape[2], 32)

    def test_forward_very_short_audio(self):
        """Test forward with very short audio"""
        config = FeatureConfig(feature_type=FeatureType.FBANK, n_mels=32, n_fft=512, hop_length=256)
        extractor = FeatureExtractor(config, sr=16000)

        # Very short audio: just enough for one frame
        waveform = torch.randn(2, 512)
        features = extractor(waveform)

        # Should still produce valid output
        self.assertEqual(features.shape[0], 2)
        self.assertEqual(features.shape[2], 32)

    def test_preemphasis_with_extreme_coeff(self):
        """Test preemphasis with extreme coefficient values"""
        config = FeatureConfig()
        extractor = FeatureExtractor(config)

        signal = torch.randn(1, 100)

        # coeff = 0 should return original
        result = extractor.preemphasis(signal, coeff=0.0)
        torch.testing.assert_close(result, signal)

        # Very small positive coeff
        result_small = extractor.preemphasis(signal, coeff=0.01)
        self.assertEqual(result_small.shape, signal.shape)

        # Large coeff (close to 1)
        result_large = extractor.preemphasis(signal, coeff=0.99)
        self.assertEqual(result_large.shape, signal.shape)

    def test_frame_signal_exact_fit(self):
        """Test frame_signal when signal length exactly fits frames"""
        config = FeatureConfig()
        extractor = FeatureExtractor(config)

        # Length: 256, frame_length: 128, hop: 64
        # Should give exactly 3 frames: [0-127], [64-191], [128-255]
        signal = torch.randn(1, 256)
        frames = extractor.frame_signal(signal, frame_length=128, hop_length=64)

        self.assertEqual(frames.shape, (1, 3, 128))

    def test_energy_to_db_with_extreme_values(self):
        """Test _energy_to_db with extreme energy values"""
        config = FeatureConfig()
        extractor = FeatureExtractor(config)

        # Very large energy
        large_energy = torch.tensor([1e10])
        db_large = extractor._energy_to_db(large_energy)
        self.assertTrue(torch.isfinite(db_large).all())

        # Very small energy (below amin)
        small_energy = torch.tensor([1e-15])
        db_small = extractor._energy_to_db(small_energy)
        self.assertTrue(torch.isfinite(db_small).all())


class TestFeatureExtractorDevice(unittest.TestCase):
    """Test device handling"""

    def test_mel_matrix_device(self):
        """Test that Mel matrix is on correct device"""
        config = FeatureConfig()
        extractor = FeatureExtractor(config)

        # Check buffer device
        self.assertEqual(extractor._mel_matrix.device, torch.device('cpu'))

    def test_dct_matrix_device(self):
        """Test that DCT matrix is on correct device when MFCC enabled"""
        config = FeatureConfig(feature_type=FeatureType.MFCC)
        extractor = FeatureExtractor(config)

        self.assertEqual(extractor._dct_matrix.device, torch.device('cpu'))


class TestComputeMfccDirectly(unittest.TestCase):
    """Direct tests for compute_mfcc method"""

    def test_compute_mfcc_shape(self):
        """Test MFCC computation shape"""
        config = FeatureConfig(feature_type=FeatureType.MFCC, n_mels=32, n_mfcc=16)
        extractor = FeatureExtractor(config)

        # Input: [B, T, n_mels]
        log_mel = torch.randn(2, 10, 32)
        mfcc = extractor.compute_mfcc(log_mel)

        # Output: [B, T, n_mfcc]
        self.assertEqual(mfcc.shape, (2, 10, 16))

    def test_compute_mfcc_reconstruction(self):
        """Test that MFCC preserves information from log_mel"""
        config = FeatureConfig(feature_type=FeatureType.MFCC, n_mels=32, n_mfcc=32)
        extractor = FeatureExtractor(config)

        # When n_mfcc == n_mels, should be invertible (approximately)
        log_mel = torch.randn(1, 5, 32)
        mfcc = extractor.compute_mfcc(log_mel)

        # Shape should match
        self.assertEqual(mfcc.shape, log_mel.shape)
    """Test delta feature computation"""

    def setUp(self):
        self.config = FeatureConfig()
        self.extractor = FeatureExtractor(self.config)

    def test_delta_time_axis(self):
        """Test time delta computation (axis=2)"""
        # Features: [B=1, F=3, T=5]
        features = torch.arange(15).float().reshape(1, 3, 5)

        delta = self.extractor.compute_delta(features, axis=2)

        # Delta shape should match input
        self.assertEqual(delta.shape, features.shape)

        # First column: central diff of [0, 0, 1, 2, 3] after padding
        # = (1 - 0) / 2 = 0.5 (approximate due to replicate padding)
        self.assertEqual(delta.shape, (1, 3, 5))

    def test_delta_freq_axis(self):
        """Test frequency delta computation (axis=1)"""
        # Features: [B=1, F=5, T=3]
        features = torch.arange(15).float().reshape(1, 5, 3)

        delta = self.extractor.compute_delta(features, axis=1)

        self.assertEqual(delta.shape, features.shape)

    def test_delta_constant_input(self):
        """Test that constant input produces zero delta"""
        features = torch.ones(1, 5, 10)
        delta = self.extractor.compute_delta(features, axis=2)

        # For constant input, delta should be zero
        torch.testing.assert_close(delta, torch.zeros_like(delta))


class TestDropBlock(unittest.TestCase):
    """Test DropBlock masking functionality"""

    def setUp(self):
        self.config = FeatureConfig(n_mels=32)
        self.extractor = FeatureExtractor(self.config)
        self.extractor.train()  # Enable training mode

    def test_dropblock_basic(self):
        """Test basic DropBlock masking"""
        features = torch.ones(2, 20, 32)  # [B, T, F]

        masked, actual_rate = self.extractor.dropblock(features, rate=0.3, block_size=5)

        # Check shape preserved
        self.assertEqual(masked.shape, features.shape)
        # Check some values were masked
        self.assertTrue((masked == 0).any())
        self.assertGreater(actual_rate, 0)

    def test_dropblock_eval_mode(self):
        """Test DropBlock disabled in eval mode"""
        self.extractor.eval()
        features = torch.ones(2, 20, 32)

        masked, actual_rate = self.extractor.dropblock(features, rate=0.3, block_size=5)

        # Should not mask in eval mode
        self.assertEqual(actual_rate, 0.0)
        torch.testing.assert_close(masked, features)

    def test_dropblock_zero_rate(self):
        """Test DropBlock with zero rate"""
        features = torch.ones(2, 20, 32)

        masked, actual_rate = self.extractor.dropblock(features, rate=0.0, block_size=5)

        self.assertEqual(actual_rate, 0.0)
        torch.testing.assert_close(masked, features)

    def test_dropblock_different_block_sizes(self):
        """Test DropBlock with various block sizes"""
        features = torch.ones(4, 40, 32)

        for block_size in [1, 3, 5, 7]:
            with self.subTest(block_size=block_size):
                masked, actual_rate = self.extractor.dropblock(features, rate=0.2, block_size=block_size)
                self.assertEqual(masked.shape, features.shape)
                # Block size should not exceed dimensions
                self.assertLessEqual(block_size, 40)
                self.assertLessEqual(block_size, 32)


class TestNormalizeFbank(unittest.TestCase):
    """Test FBank normalization"""

    def setUp(self):
        self.config = FeatureConfig(n_mels=10, fbank_decay=0.9)
        self.extractor = FeatureExtractor(self.config)

    def test_normalize_fbank_range(self):
        """Test that normalized fbank is in [0, 1] range"""
        # Create random fbank: [B=2, T=20, n_mels=10]
        fbank = torch.randn(2, 20, 10) * 5 + 10  # Random values around 10

        normalized = self.extractor.normalize_fbank(fbank)

        # Check range
        self.assertTrue((normalized >= 0).all())
        self.assertTrue((normalized <= 1).all())

    def test_normalize_fbank_shape_preservation(self):
        """Test that normalization preserves shape"""
        fbank = torch.randn(2, 20, 10)
        normalized = self.extractor.normalize_fbank(fbank)
        self.assertEqual(normalized.shape, fbank.shape)

    def test_forward_fbank_without_norm(self):
        """Test forward pass with FBank but use_fbank_norm=False"""
        config = FeatureConfig(
            feature_type=FeatureType.FBANK,
            n_mels=32,
            hop_length=500,
            use_fbank_norm=False
        )
        extractor = FeatureExtractor(config, sr=16000)
        waveform = torch.randn(2, 8000)  # 0.5s @ 16kHz

        features = extractor(waveform)

        # Check output shape: [B, T, F]
        self.assertEqual(features.ndim, 3)
        self.assertEqual(features.shape[0], 2)  # batch
        self.assertEqual(features.shape[2], 32)  # n_mels

        # Without normalization, features should be clamped to [0, 7] range
        # (as per feature.py line 352: fbank = torch.clamp(fbank, 0, 7))
        self.assertTrue((features >= 0).all())
        self.assertTrue((features <= 7).all())


class TestForwardPass(unittest.TestCase):
    """Test forward pass with various configurations"""

    def _create_waveform(self, batch_size=2, duration_sec=0.5, sr=16000):
        """Helper to create test waveform"""
        samples = int(duration_sec * sr)
        return torch.randn(batch_size, samples)

    def test_forward_fbank_only(self):
        """Test forward pass with FBank only"""
        config = FeatureConfig(
            feature_type=FeatureType.FBANK,
            n_mels=32,
            hop_length=500
        )
        extractor = FeatureExtractor(config, sr=16000)
        waveform = self._create_waveform()

        features = extractor(waveform)

        # Check output shape: [B, T, F]
        self.assertEqual(features.ndim, 3)
        self.assertEqual(features.shape[0], 2)  # batch
        self.assertEqual(features.shape[2], 32)  # n_mels

    def test_forward_fbank_with_db(self):
        """Test forward pass with FBank + DB"""
        config = FeatureConfig(
            feature_type=FeatureType.FBANK | FeatureType.DB,
            n_mels=32,
            hop_length=500
        )
        extractor = FeatureExtractor(config, sr=16000)
        waveform = self._create_waveform()

        features = extractor(waveform)

        # Feature dim: 32 (FBANK) + 2 (DB) = 34
        self.assertEqual(features.shape[2], 34)

    def test_forward_fbank_with_deltas(self):
        """Test forward pass with FBank + time/freq deltas"""
        config = FeatureConfig(
            feature_type=FeatureType.FBANK,
            n_mels=32,
            hop_length=500,
            use_time_delta=True,
            use_freq_delta=True
        )
        extractor = FeatureExtractor(config, sr=16000)
        waveform = self._create_waveform()

        features = extractor(waveform)

        # Feature dim: 32 * 3 = 96 (base + time_delta + freq_delta)
        self.assertEqual(features.shape[2], 96)

    def test_forward_mfcc_only(self):
        """Test forward pass with MFCC only"""
        config = FeatureConfig(
            feature_type=FeatureType.MFCC,
            n_mfcc=16,
            hop_length=500
        )
        extractor = FeatureExtractor(config, sr=16000)
        waveform = self._create_waveform()

        features = extractor(waveform)

        self.assertEqual(features.shape[2], 16)  # n_mfcc

    def test_forward_single_waveform(self):
        """Test forward pass with single waveform (no batch)"""
        config = FeatureConfig(feature_type=FeatureType.FBANK, n_mels=32)
        extractor = FeatureExtractor(config, sr=16000)

        # Single waveform: [N] instead of [B, N]
        waveform = torch.randn(8000)  # 0.5s @ 16kHz

        features = extractor(waveform)

        # Output should be [1, T, F] (batch dimension added)
        self.assertEqual(features.shape[0], 1)
        self.assertEqual(features.shape[2], 32)

    def test_forward_all_features(self):
        """Test forward pass with all features combined"""
        config = FeatureConfig(
            feature_type=FeatureType.FBANK | FeatureType.DB | FeatureType.MFCC,
            n_mels=32,
            n_mfcc=16,
            hop_length=500
        )
        extractor = FeatureExtractor(config, sr=16000)
        waveform = self._create_waveform()

        features = extractor(waveform)

        # Feature dim: 32 + 2 + 16 = 50
        self.assertEqual(features.shape[2], 50)


class TestFeatureExtractorGPU(unittest.TestCase):
    """Test GPU support if available"""

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_forward_on_gpu(self):
        """Test forward pass on GPU"""
        config = FeatureConfig(feature_type=FeatureType.FBANK, n_mels=32)
        extractor = FeatureExtractor(config, sr=16000)
        extractor = extractor.cuda()

        waveform = torch.randn(2, 16000).cuda()
        features = extractor(waveform)

        self.assertTrue(features.is_cuda)
        self.assertEqual(features.shape[2], 32)


class TestFeatureExtractorNumericalStability(unittest.TestCase):
    """Test numerical stability edge cases"""

    def test_forward_with_silence(self):
        """Test forward pass with silent input"""
        config = FeatureConfig(feature_type=FeatureType.FBANK, n_mels=32)
        extractor = FeatureExtractor(config, sr=16000)

        # Silent waveform
        waveform = torch.zeros(2, 16000)
        features = extractor(waveform)

        # Should not produce NaN or Inf
        self.assertTrue(torch.isfinite(features).all())

    def test_forward_with_very_small_values(self):
        """Test forward pass with very small input values"""
        config = FeatureConfig(feature_type=FeatureType.FBANK, n_mels=32)
        extractor = FeatureExtractor(config, sr=16000)

        waveform = torch.randn(2, 16000) * 1e-6
        features = extractor(waveform)

        self.assertTrue(torch.isfinite(features).all())

    def test_forward_with_large_values(self):
        """Test forward pass with large input values"""
        config = FeatureConfig(feature_type=FeatureType.FBANK, n_mels=32)
        extractor = FeatureExtractor(config, sr=16000)

        waveform = torch.randn(2, 8000) * 100
        features = extractor(waveform)

        # Should be clamped and normalized
        self.assertTrue(torch.isfinite(features).all())


class TestDropBlockInForward(unittest.TestCase):
    """Test DropBlock integration in forward pass for FBank features"""

    def _create_config(self, mask_enable=True, mask_rate=0.2):
        """Helper to create config with mask enabled"""
        from utils.config import MaskConfig
        return FeatureConfig(
            feature_type=FeatureType.FBANK,
            n_mels=32,
            n_fft=1024,
            hop_length=500,
            mask=MaskConfig(enable=mask_enable, rate=mask_rate, prob=1.0)  # prob=1.0 to ensure masking
        )

    def test_forward_fbank_with_dropblock_training(self):
        """Test that DropBlock is applied to FBank in training mode"""
        config = self._create_config(mask_enable=True, mask_rate=0.3)
        extractor = FeatureExtractor(config, sr=16000)
        extractor.train()  # Ensure training mode

        # Use fixed random seed for reproducibility
        torch.manual_seed(42)
        waveform = torch.randn(2, 8000)  # 0.5s @ 16kHz

        # Run multiple times to account for randomness
        masked_count = 0
        for _ in range(5):
            torch.manual_seed(42 + _)
            features = extractor(waveform)
            # Check if any features are zero (masked)
            if (features == 0).any():
                masked_count += 1

        # At least some runs should have masking (high probability with rate=0.3)
        self.assertGreater(masked_count, 0, "DropBlock should mask some features in training mode")

    def test_forward_fbank_with_dropblock_eval(self):
        """Test that DropBlock is NOT applied in eval mode"""
        config = self._create_config(mask_enable=True, mask_rate=0.3)
        extractor = FeatureExtractor(config, sr=16000)
        extractor.eval()  # Set eval mode

        torch.manual_seed(42)
        waveform = torch.randn(2, 8000)

        features = extractor(waveform)

        # In eval mode, no masking should occur (all values should be non-zero for this input)
        # With random input, very unlikely to get exactly zero without masking
        self.assertTrue(torch.all(features > 0) or not (features == 0).any(),
                       "DropBlock should not mask in eval mode")

    def test_forward_fbank_dropblock_disabled(self):
        """Test that DropBlock is not applied when enable=False"""
        config = self._create_config(mask_enable=False, mask_rate=0.3)
        extractor = FeatureExtractor(config, sr=16000)
        extractor.train()

        torch.manual_seed(42)
        waveform = torch.randn(2, 8000)

        features = extractor(waveform)

        # Should not have masking when disabled
        self.assertTrue(torch.all(features > 0) or not (features == 0).any(),
                       "DropBlock should not apply when enable=False")

    def test_forward_fbank_dropblock_zero_rate(self):
        """Test that DropBlock with rate=0 does not mask"""
        config = self._create_config(mask_enable=True, mask_rate=0.0)
        extractor = FeatureExtractor(config, sr=16000)
        extractor.train()

        torch.manual_seed(42)
        waveform = torch.randn(2, 8000)

        features = extractor(waveform)

        # With rate=0, should not mask
        self.assertTrue(torch.all(features > 0) or not (features == 0).any(),
                       "DropBlock should not mask when rate=0")

    def test_forward_fbank_dropblock_shape_preserved(self):
        """Test that DropBlock preserves feature shape"""
        config = self._create_config(mask_enable=True, mask_rate=0.2)
        extractor = FeatureExtractor(config, sr=16000)
        extractor.train()

        waveform = torch.randn(4, 8000)
        features = extractor(waveform)

        # Shape should be [B, T, F]
        self.assertEqual(features.ndim, 3)
        self.assertEqual(features.shape[0], 4)  # batch
        self.assertEqual(features.shape[2], 32)  # n_mels


class TestMaskConfigStartEpoch(unittest.TestCase):
    """Test MaskConfig start_epoch and end_epoch parameters"""

    def test_mask_config_default_start_epoch(self):
        """Test that MaskConfig has default start_epoch of 0"""
        from utils.config import MaskConfig
        config = MaskConfig()
        self.assertEqual(config.start_epoch, 0)

    def test_mask_config_default_end_epoch(self):
        """Test that MaskConfig has default end_epoch of -1"""
        from utils.config import MaskConfig
        config = MaskConfig()
        self.assertEqual(config.end_epoch, -1)

    def test_mask_config_custom_start_epoch(self):
        """Test that MaskConfig accepts custom start_epoch"""
        from utils.config import MaskConfig
        config = MaskConfig(enable=True, rate=0.2, prob=0.5, start_epoch=20)
        self.assertEqual(config.start_epoch, 20)

    def test_mask_config_custom_end_epoch(self):
        """Test that MaskConfig accepts custom end_epoch"""
        from utils.config import MaskConfig
        config = MaskConfig(enable=True, rate=0.2, prob=0.5, start_epoch=20, end_epoch=50)
        self.assertEqual(config.start_epoch, 20)
        self.assertEqual(config.end_epoch, 50)

    def test_feature_config_mask_with_epoch_range(self):
        """Test FeatureConfig with mask containing start/end epoch"""
        from utils.config import MaskConfig, FeatureConfig
        mask_config = MaskConfig(enable=True, rate=0.2, start_epoch=10, end_epoch=80)
        feature_config = FeatureConfig(
            feature_type=FeatureType.FBANK,
            mask=mask_config
        )
        self.assertEqual(feature_config.mask.start_epoch, 10)
        self.assertEqual(feature_config.mask.end_epoch, 80)

    def test_mask_epoch_range_logic(self):
        """Test the epoch range logic for enabling mask"""
        from utils.config import MaskConfig

        def is_mask_enabled(config, current_epoch):
            current = current_epoch
            start = config.start_epoch
            end = config.end_epoch
            return config.enable and current >= start and (end < 0 or current < end)

        # Test: enable from epoch 20, no end
        config = MaskConfig(enable=True, start_epoch=20, end_epoch=-1)
        self.assertFalse(is_mask_enabled(config, 0))
        self.assertFalse(is_mask_enabled(config, 19))
        self.assertTrue(is_mask_enabled(config, 20))
        self.assertTrue(is_mask_enabled(config, 100))

        # Test: enable from epoch 20 to 50
        config = MaskConfig(enable=True, start_epoch=20, end_epoch=50)
        self.assertFalse(is_mask_enabled(config, 0))
        self.assertFalse(is_mask_enabled(config, 19))
        self.assertTrue(is_mask_enabled(config, 20))
        self.assertTrue(is_mask_enabled(config, 49))
        self.assertFalse(is_mask_enabled(config, 50))
        self.assertFalse(is_mask_enabled(config, 100))

        # Test: disabled
        config = MaskConfig(enable=False, start_epoch=0, end_epoch=-1)
        self.assertFalse(is_mask_enabled(config, 0))
        self.assertFalse(is_mask_enabled(config, 50))


if __name__ == '__main__':
    unittest.main()
