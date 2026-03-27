"""
Stub data generator for end-to-end testing.

Generates synthetic audio data and labels for testing train/evaluate pipelines
without requiring real audio files.
"""

import json
import os
import tempfile
import wave
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple


class StubAudioGenerator:
    """Generate synthetic audio files for testing."""

    def __init__(self, sample_rate: int = 16000, duration_range: Tuple[float, float] = (3.0, 8.0)):
        self.sample_rate = sample_rate
        self.duration_range = duration_range
        np.random.seed(42)

    def _generate_cry_like_signal(self, duration: float) -> np.ndarray:
        """Generate a signal that resembles baby cry (high frequency, periodic)."""
        samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, samples)

        # Cry-like: high pitch (~500-1000Hz) with harmonics and amplitude modulation
        base_freq = np.random.uniform(400, 800)
        harmonic_freq = base_freq * 2

        # Amplitude modulation (crying rhythm)
        mod_freq = np.random.uniform(2, 5)  # 2-5 cries per second
        modulator = 0.5 + 0.5 * np.sin(2 * np.pi * mod_freq * t)

        # Signal with harmonics
        signal = (
            np.sin(2 * np.pi * base_freq * t) * 0.6 +
            np.sin(2 * np.pi * harmonic_freq * t) * 0.3 +
            np.random.randn(samples) * 0.05  # Small noise
        ) * modulator

        return signal.astype(np.float32)

    def _generate_noise_signal(self, duration: float) -> np.ndarray:
        """Generate noise/silence for non-cry samples."""
        samples = int(duration * self.sample_rate)

        # Mix of white noise, pink noise, and silence
        noise_type = np.random.choice(['white', 'pink', 'silence', 'low_freq'])

        if noise_type == 'white':
            signal = np.random.randn(samples) * 0.1
        elif noise_type == 'pink':
            # Pink noise (1/f spectrum)
            freqs = np.fft.rfftfreq(samples)
            freqs[0] = 1  # Avoid division by zero
            phase = np.random.uniform(0, 2*np.pi, len(freqs))
            spectrum = (1 / np.sqrt(freqs)) * np.exp(1j * phase)
            signal = np.fft.irfft(spectrum, n=samples).real * 0.1
        elif noise_type == 'silence':
            signal = np.random.randn(samples) * 0.01  # Near silence
        else:  # low_freq
            # Low frequency rumble (not cry-like)
            t = np.linspace(0, duration, samples)
            signal = np.sin(2 * np.pi * np.random.uniform(50, 150) * t) * 0.1

        return signal.astype(np.float32)

    def generate_wav(self, path: str, is_cry: bool = False) -> str:
        """Generate a single WAV file."""
        duration = np.random.uniform(*self.duration_range)

        if is_cry:
            signal = self._generate_cry_like_signal(duration)
        else:
            signal = self._generate_noise_signal(duration)

        # Normalize to 16-bit range
        signal = (signal / np.max(np.abs(signal)) * 32767).astype(np.int16)

        # Write WAV file
        with wave.open(path, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(signal.tobytes())

        return path

    def generate_dataset(
        self,
        output_dir: str,
        num_cry: int = 10,
        num_other: int = 10,
        prefix: str = "stub"
    ) -> Dict[str, List[str]]:
        """Generate a complete dataset with cry and non-cry samples."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        cry_dir = output_path / "cry"
        other_dir = output_path / "other"
        cry_dir.mkdir(exist_ok=True)
        other_dir.mkdir(exist_ok=True)

        dataset = {"cry": [], "other": []}

        # Generate cry samples
        for i in range(num_cry):
            path = cry_dir / f"{prefix}_cry_{i:03d}.wav"
            self.generate_wav(str(path), is_cry=True)
            dataset["cry"].append(str(path))

        # Generate non-cry samples (first element is duplicate count for non-cry)
        for i in range(num_other):
            path = other_dir / f"{prefix}_other_{i:03d}.wav"
            self.generate_wav(str(path), is_cry=False)
            dataset["other"].append(str(path))

        return dataset


class StubDataManager:
    """Manage stub data for end-to-end testing."""

    def __init__(self, base_dir: str = None):
        self.base_dir = Path(base_dir) if base_dir else Path(tempfile.mkdtemp(prefix="crydet_test_"))
        self.generator = StubAudioGenerator()
        self.audio_list_path = None
        self.dataset_dict = None

    def create_train_val_test_split(
        self,
        train_cry: int = 20,
        train_other: int = 20,
        val_cry: int = 5,
        val_other: int = 5,
        test_cry: int = 5,
        test_other: int = 5
    ) -> Dict[str, str]:
        """Create complete train/val/test datasets and return paths to audio_list JSONs."""
        splits = {}

        # Create train dataset
        train_dir = self.base_dir / "train"
        train_data = self.generator.generate_dataset(train_dir, train_cry, train_other, "train")
        splits['train'] = self._create_audio_list_json(train_data, "train")

        # Create validation dataset
        val_dir = self.base_dir / "val"
        val_data = self.generator.generate_dataset(val_dir, val_cry, val_other, "val")
        splits['val'] = self._create_audio_list_json(val_data, "val")

        # Create test dataset
        test_dir = self.base_dir / "test"
        test_data = self.generator.generate_dataset(test_dir, test_cry, test_other, "test")
        splits['test'] = self._create_audio_list_json(test_data, "test")

        self.audio_list_path = splits
        return splits

    def _create_audio_list_json(self, dataset: Dict[str, List[str]], name: str) -> str:
        """Create audio_list JSON file from dataset dict."""
        # CryDataset expects label -> directory list mapping
        # Format: {"cry": ["/path/to/cry/dir"], "other": [1, "/path/to/other/dir"]}
        # where 1 is duplicate multiplier for non-cry
        cry_dir = str(self.base_dir / name / "cry")
        other_dir = str(self.base_dir / name / "other")

        audio_list = {
            "cry": [cry_dir],
            "other": [1, other_dir]  # Add duplicate count as first element for non-cry
        }

        json_path = self.base_dir / f"audio_list_{name}.json"
        with open(json_path, 'w') as f:
            json.dump(audio_list, f, indent=2)

        return str(json_path)

    def create_minimal_config(self, overrides: Dict = None) -> str:
        """Create a minimal config file for fast testing."""
        config = {
            "feature": {
                "feature_type": 3,
                "n_mels": 32,
                "n_mfcc": 16,
                "n_fft": 1024,
                "hop_length": 500,
                "fmin": 250,
                "fmax": 8000,
                "preemphasis": 0.95,
                "use_fbank_norm": True,
                "fbank_decay": 0.9,
                "use_db_norm": False,
                "use_time_delta": False,
                "use_freq_delta": False,
                "mask": {
                    "enable": False,
                    "rate": 0.2,
                    "prob": 0.5,
                    "start_epoch": 0,
                    "end_epoch": -1
                }
            },
            "dataset": {
                "audio_suffixes": [".wav"],
                "sample_rate": 16000,
                "slice_len": 3.0,
                "stride": 2.0,
                "cry_rate": 0.5,
                "cache_dir": str(self.base_dir / "audio_cache"),
                "force_mono": True
            },
            "augmentation": {
                "mixup": {
                    "cry_mix_prob": 0.0,
                    "cry_mix_rate_mean": 0.3,
                    "cry_mix_rate_std": 0.15,
                    "other_mix_prob": 0.0,
                    "mix_front_prob": 0.7
                },
                "noise": {
                    "prob": 0.0
                },
                "cry_aug_prob": 0.0,
                "other_aug_prob": 0.0,
                "other_reverse_prob": 0.0,
                "pitch_prob": 0.0,
                "reverb_prob": 0.0,
                "phaser_prob": 0.0,
                "echo_prob": 0.0,
                "gain_prob": 0.0,
                "time_stretch_prob": 0.0
            },
            "model": {
                "model_type": "transformer",
                "num_classes": 2,
                "d_model": 64,
                "n_heads": 2,
                "n_layers": 2,
                "d_ff": 128,
                "dropout": 0.1,
                "attention_dropout": 0.1,
                "max_seq_len": 200,
                "use_relative_pos": True,
                "attention_type": "depthwise",
                "ffn_type": "inverted_bottleneck",
                "pool_type": "mean",
                "label_smoothing": 0.1
            },
            "training": {
                "batch_size": 4,
                "num_workers": 0,
                "pin_memory": False,
                "num_epochs": 2,
                "grad_clip": 1.0,
                "early_stopping_patience": 5,
                "use_augmentation": False,
                "device": "cpu",
                "log_interval": 5,
                "val_interval": 1,
                "save_best_only": True,
                "run_dir": str(self.base_dir / "runs"),
                "seed": 42,
                "optimizer": {
                    "type": "adamw",
                    "lr": 0.001,
                    "weight_decay": 0.00001,
                    "betas": [0.9, 0.98],
                    "eps": 1e-8
                },
                "scheduler": {
                    "type": "cosine_warmup",
                    "warmup_epochs": 1,
                    "min_lr": 1e-6
                },
                "loss": {
                    "loss_type": "cross_entropy",
                    "label_smoothing": 0.1
                },
                "use_spec_augment": False,
                "use_ema": False
            }
        }

        if overrides:
            self._deep_update(config, overrides)

        config_path = self.base_dir / "test_config.yaml"

        # Convert to YAML format
        with open(config_path, 'w') as f:
            f.write(self._dict_to_yaml(config))

        return str(config_path)

    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """Recursively update nested dictionary."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    def _dict_to_yaml(self, data: Dict, indent: int = 0) -> str:
        """Convert dictionary to YAML string."""
        yaml_str = ""
        for key, value in data.items():
            if isinstance(value, dict):
                yaml_str += "  " * indent + f"{key}:\n"
                yaml_str += self._dict_to_yaml(value, indent + 1)
            elif isinstance(value, list):
                yaml_str += "  " * indent + f"{key}:\n"
                for item in value:
                    if isinstance(item, str):
                        yaml_str += "  " * (indent + 1) + f"- \"{item}\"\n"
                    else:
                        yaml_str += "  " * (indent + 1) + f"- {item}\n"
            elif isinstance(value, str):
                yaml_str += "  " * indent + f"{key}: \"{value}\"\n"
            else:
                yaml_str += "  " * indent + f"{key}: {value}\n"
        return yaml_str

    def cleanup(self):
        """Remove all generated files."""
        import shutil
        if self.base_dir.exists():
            shutil.rmtree(self.base_dir)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False


def create_mock_checkpoint(save_path: str, config_path: str = None) -> str:
    """Create a mock model checkpoint for testing inference/evaluation."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from utils.config import load_config
    from model import create_model

    if config_path:
        config = load_config(config_path)
    else:
        # Create minimal default config
        from utils.config import Config, FeatureConfig, ModelConfig, DatasetConfig
        config = Config(
            feature=FeatureConfig(n_mels=32, feature_type=3),
            model=ModelConfig(d_model=64, n_layers=2, n_heads=2),
            dataset=DatasetConfig()
        )

    # Create model with correct input dimension
    model = create_model(
        config=config.model,
        in_channels=config.feature.feature_dim,
        num_classes=config.model.num_classes
    )

    # Save checkpoint
    checkpoint = {
        'epoch': 1,
        'global_step': 100,
        'model_state_dict': model.state_dict(),
        'best_val_f1': 0.5,
        'config': config
    }

    torch.save(checkpoint, save_path)
    return save_path


if __name__ == "__main__":
    # Test stub data generation
    with StubDataManager() as manager:
        splits = manager.create_train_val_test_split(
            train_cry=5, train_other=5,
            val_cry=2, val_other=2,
            test_cry=2, test_other=2
        )
        config_path = manager.create_minimal_config()

        print(f"Created stub datasets:")
        print(f"  Train: {splits['train']}")
        print(f"  Val: {splits['val']}")
        print(f"  Test: {splits['test']}")
        print(f"  Config: {config_path}")

        # Verify files exist
        for split, path in splits.items():
            with open(path) as f:
                data = json.load(f)
            print(f"  {split}: {len(data['cry'])} cry, {len(data['other'])-1} other")
