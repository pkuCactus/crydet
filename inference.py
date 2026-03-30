"""
Inference script for CryTransformer models

Supports single file inference and batch inference.

Usage:
    # Single file inference
    python inference.py --checkpoint checkpoints/best_model.pt --audio path/to/audio.wav

    # Batch inference
    python inference.py --checkpoint checkpoints/best_model.pt --audio_list audio_list/test.json
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from dataset.audio_reader import AudioReader
from dataset.feature import FeatureExtractor
from model import create_model
from utils.logger import setup_logger, setup_file_logger

# Initialize logger using utils.logger setup
LOGGER = setup_logger(rank=0, name=__name__)


def extract_audio_from_video(video_path: str, output_path: str, sample_rate: int = 16000) -> bool:
    """
    Extract audio from video file using ffmpeg.

    Args:
        video_path: Path to video file (mp4, avi, mov, etc.)
        output_path: Path to output audio file (wav)
        sample_rate: Target sample rate for extracted audio

    Returns:
        True if extraction succeeded, False otherwise
    """
    try:
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file if exists
            '-i', video_path,  # Input file
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM 16-bit little-endian
            '-ar', str(sample_rate),  # Sample rate
            '-ac', '1',  # Mono
            output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        LOGGER.error(f"Failed to extract audio from {video_path}: {e.stderr}")
        return False
    except FileNotFoundError:
        LOGGER.error("ffmpeg not found. Please install ffmpeg to process video files.")
        return False


def is_video_file(path: str) -> bool:
    """Check if file is a video file based on extension."""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
    return Path(path).suffix.lower() in video_extensions


class CryDetector:
    """
    Baby cry detector using CryTransformer model.

    Supports real-time inference with sliding window.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda',
        sample_rate: int = 16000,
        slice_len: float = 5.0,
        stride: float = 1.0,
        threshold: float = 0.5
    ):
        """
        Initialize cry detector.

        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on ('cuda' or 'cpu')
            sample_rate: Audio sample rate (default: 16000)
            slice_len: Length of each analysis window in seconds (default: 5.0)
            stride: Stride between consecutive windows in seconds (default: 1.0)
            threshold: Probability threshold for cry detection (default: 0.5)
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.sample_rate = sample_rate
        self.slice_len = slice_len
        self.stride = stride
        self.threshold = threshold

        # Load model
        LOGGER.info(f"Loading model from: {checkpoint_path}")
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.config = self.checkpoint['config']

        self.model = create_model(
            config=self.config.model,
            in_channels=self.config.feature.feature_dim,
            num_classes=self.config.model.num_classes
        )
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # Feature extractor
        self.feature_extractor = FeatureExtractor(self.config.feature)

        # Audio reader
        self.audio_reader = AudioReader(
            target_sr=sample_rate,
            cache_dir=None,
            force_mono=True
        )

        LOGGER.info(f"Model loaded. Device: {self.device}")

    def _extract_features(self, waveform: np.ndarray) -> torch.Tensor:
        """Extract features from waveform"""
        # Convert numpy array to torch tensor
        waveform_tensor = torch.from_numpy(waveform).float()
        # FeatureExtractor expects [B, N] or [N] format
        features = self.feature_extractor(waveform_tensor)
        # If single sample, add batch dimension
        if features.ndim == 2:
            features = features.unsqueeze(0)  # (1, T, F)
        return features.to(self.device)

    @torch.no_grad()
    def predict(self, waveform: np.ndarray) -> Dict:
        """
        Predict on a single audio segment.

        Args:
            waveform: Audio waveform (1D numpy array)

        Returns:
            Dictionary with prediction results:
                - 'is_cry': Boolean indicating if cry detected
                - 'cry_prob': Probability of cry (0-1)
                - 'confidence': Prediction confidence
        """
        features = self._extract_features(waveform)

        # Inference
        outputs = self.model(features)
        probs = F.softmax(outputs, dim=1)
        cry_prob = probs[0, 1].item()

        return {
            'is_cry': cry_prob > self.threshold,
            'cry_prob': cry_prob,
            'non_cry_prob': probs[0, 0].item(),
            'confidence': abs(cry_prob - 0.5) * 2  # Scale to 0-1
        }

    @torch.no_grad()
    def predict_file(self, audio_path: str, sliding_window: bool = True) -> Union[Dict, List[Dict]]:
        """
        Predict on an audio file or video file.

        Args:
            audio_path: Path to audio file (wav, mp3, etc.) or video file (mp4, avi, etc.)
            sliding_window: Whether to use sliding window for long audio (default: True)

        Returns:
            If sliding_window=False: Single prediction dict
            If sliding_window=True: List of prediction dicts for each window
        """
        # Check if input is a video file
        if is_video_file(audio_path):
            LOGGER.info(f"Detected video file: {audio_path}, extracting audio...")
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_path = tmp_file.name
            try:
                if not extract_audio_from_video(audio_path, tmp_path, self.sample_rate):
                    raise RuntimeError(f"Failed to extract audio from video: {audio_path}")
                waveform, sr = self.audio_reader.load(tmp_path)
            finally:
                # Clean up temp file
                Path(tmp_path).unlink(missing_ok=True)
        else:
            # Load audio directly
            waveform, sr = self.audio_reader.load(audio_path)

        duration = len(waveform) / sr

        LOGGER.debug(f"Audio loaded: {duration:.2f}s, sr={sr}")

        if not sliding_window or duration <= self.slice_len:
            # Single prediction for short audio
            result = self.predict(waveform)
            # Add time info for consistency
            result['start_time'] = 0.0
            result['end_time'] = duration
            return result

        # Sliding window for long audio
        slice_samples = int(self.slice_len * sr)
        stride_samples = int(self.stride * sr)

        results = []
        start = 0
        while start + slice_samples <= len(waveform):
            segment = waveform[start:start + slice_samples]
            pred = self.predict(segment)
            pred['start_time'] = start / sr
            pred['end_time'] = (start + slice_samples) / sr
            results.append(pred)
            start += stride_samples

        return results

    def detect_cry_regions(
        self,
        audio_path: str,
        min_duration: float = 1.0,
        merge_gap: float = 0.5
    ) -> List[Dict]:
        """
        Detect cry regions in an audio file.

        Args:
            audio_path: Path to audio file
            min_duration: Minimum duration for a cry region in seconds
            merge_gap: Merge regions separated by less than this gap

        Returns:
            List of detected cry regions:
                [{'start': float, 'end': float, 'confidence': float}, ...]
        """
        predictions = self.predict_file(audio_path, sliding_window=True)

        if isinstance(predictions, dict):
            predictions = [predictions]

        # Find cry regions
        cry_regions = []
        current_region = None

        for pred in predictions:
            if pred['is_cry']:
                if current_region is None:
                    current_region = {
                        'start': pred['start_time'],
                        'end': pred['end_time'],
                        'probs': [pred['cry_prob']]
                    }
                else:
                    # Check if we should merge
                    if pred['start_time'] - current_region['end'] <= merge_gap:
                        current_region['end'] = pred['end_time']
                        current_region['probs'].append(pred['cry_prob'])
                    else:
                        # Save current and start new
                        cry_regions.append(current_region)
                        current_region = {
                            'start': pred['start_time'],
                            'end': pred['end_time'],
                            'probs': [pred['cry_prob']]
                        }
            else:
                if current_region is not None:
                    cry_regions.append(current_region)
                    current_region = None

        if current_region is not None:
            cry_regions.append(current_region)

        # Filter by minimum duration and compute average confidence
        filtered_regions = []
        for region in cry_regions:
            duration = region['end'] - region['start']
            if duration >= min_duration:
                filtered_regions.append({
                    'start': region['start'],
                    'end': region['end'],
                    'duration': duration,
                    'confidence': np.mean(region['probs']),
                    'max_confidence': np.max(region['probs'])
                })

        return filtered_regions

    def benchmark(self, num_runs: int = 100, input_duration: float = 5.0) -> Dict:
        """
        Benchmark inference speed.

        Args:
            num_runs: Number of benchmark runs
            input_duration: Duration of test audio in seconds

        Returns:
            Dictionary with benchmark results
        """
        # Generate test audio
        samples = int(input_duration * self.sample_rate)
        test_audio = np.random.randn(samples).astype(np.float32) * 0.1

        # Warmup
        for _ in range(10):
            _ = self.predict(test_audio)

        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.time()
            _ = self.predict(test_audio)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.time() - start)

        times = np.array(times) * 1000  # ms

        return {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'p95_ms': np.percentile(times, 95),
            'p99_ms': np.percentile(times, 99),
            'rtf': np.mean(times) / (input_duration * 1000),  # Real-time factor
        }


def main():
    parser = argparse.ArgumentParser(description='CryTransformer Inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--audio', type=str, default=None, help='Path to audio file')
    parser.add_argument('--audio_list', type=str, default=None, help='Path to audio list JSON')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Detection threshold')
    parser.add_argument('--sliding_window', action='store_true', help='Use sliding window')
    parser.add_argument('--stride', type=float, default=1.0, help='Stride for sliding window')
    parser.add_argument('--detect_regions', action='store_true', help='Detect cry regions')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file')
    parser.add_argument('--log_file', type=str, default=None, help='Path to log file')
    args = parser.parse_args()

    # Reconfigure logging with file output if specified
    if args.log_file:
        global LOGGER
        LOGGER = setup_file_logger(args.log_file, rank=0, name=__name__)

    # Initialize detector
    detector = CryDetector(
        checkpoint_path=args.checkpoint,
        device=args.device,
        stride=args.stride,
        threshold=args.threshold
    )

    results = {}

    # Single file inference
    if args.audio:
        LOGGER.info(f"Processing: {args.audio}")

        if args.detect_regions:
            regions = detector.detect_cry_regions(args.audio)
            print(f"\nDetected {len(regions)} cry region(s):")
            for i, region in enumerate(regions):
                print(f"  {i+1}. {region['start']:.2f}s - {region['end']:.2f}s "
                      f"(duration: {region['duration']:.2f}s, confidence: {region['confidence']:.3f})")
            results['regions'] = regions
        else:
            result = detector.predict_file(args.audio, sliding_window=args.sliding_window)

            if isinstance(result, list):
                print(f"\nSliding window predictions ({len(result)} windows):")
                for r in result:
                    print(f"  {r['start_time']:.2f}s - {r['end_time']:.2f}s: "
                          f"Cry={r['is_cry']}, Prob={r['cry_prob']:.3f}")
            else:
                print(f"\nPrediction:")
                print(f"  Is Cry: {result['is_cry']}")
                print(f"  Cry Probability: {result['cry_prob']:.4f}")
                print(f"  Confidence: {result['confidence']:.4f}")

            results['prediction'] = result

    def _get_media_files(path: str) -> List[str]:
        """Get list of audio/video files from path (file or directory)."""
        path_obj = Path(path)
        if path_obj.is_file():
            return [str(path_obj)]
        elif path_obj.is_dir():
            # Recursively find all audio and video files
            media_exts = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.mp4', '.avi', '.mov', '.mkv', '.webm'}
            files = []
            for ext in media_exts:
                files.extend(path_obj.glob(f'**/*{ext}'))
            return sorted([str(f) for f in files])
        return []

    # Batch inference
    if args.audio_list:
        LOGGER.info(f"Processing audio list: {args.audio_list}")

        with open(args.audio_list, 'r') as f:
            data_dict = json.load(f)

        all_results = []
        total_correct = 0
        total_count = 0

        for label, paths in data_dict.items():
            # Skip non-cry duplicate count
            if label != 'cry':
                paths = paths[1:] if len(paths) > 0 and isinstance(paths[0], int) else paths

            # Expand directories to files
            all_files = []
            for path in paths:
                all_files.extend(_get_media_files(path))

            LOGGER.info(f"Processing {len(all_files)} files for label: {label}")

            for file_path in all_files[:5]:  # Process first 5 for demo
                result = detector.predict_file(file_path, sliding_window=False)
                predicted_label = 'cry' if result['is_cry'] else 'other'
                correct = (predicted_label == label)

                all_results.append({
                    'file': file_path,
                    'true_label': label,
                    'predicted_label': predicted_label,
                    'cry_prob': result['cry_prob'],
                    'correct': correct
                })

                total_correct += int(correct)
                total_count += 1

        accuracy = total_correct / total_count if total_count > 0 else 0
        print(f"\nBatch Results:")
        print(f"  Accuracy: {accuracy:.4f} ({total_correct}/{total_count})")

        results['batch'] = {
            'accuracy': accuracy,
            'predictions': all_results
        }

    # Benchmark
    if args.benchmark:
        LOGGER.info("Running benchmark...")
        bench = detector.benchmark()
        print("\nBenchmark Results:")
        print(f"  Mean: {bench['mean_ms']:.2f} ms")
        print(f"  P95:  {bench['p95_ms']:.2f} ms")
        print(f"  RTF:  {bench['rtf']:.4f} (real-time factor)")
        results['benchmark'] = bench

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        LOGGER.info(f"Results saved to: {args.output}")


if __name__ == '__main__':
    main()
