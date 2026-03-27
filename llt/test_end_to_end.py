"""
End-to-end integration tests for CryTransformer pipeline.

Tests complete workflows:
- Training from scratch
- Resuming training from checkpoint
- Evaluation on test set
- Inference on single file and batch
- Export to various formats

Uses stub data (synthetic audio) so no real audio files are required.
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llt.stub_data import StubDataManager, create_mock_checkpoint
from utils.config import load_config

# Global conda environment setting (set via --conda-env argument)
CONDA_ENV = None


def get_conda_executable():
    """Find conda executable."""
    conda_exe = os.environ.get('CONDA_EXE')
    if conda_exe and Path(conda_exe).exists():
        return conda_exe

    import shutil
    conda_path = shutil.which('conda')
    if conda_path:
        return conda_path

    # Try default locations
    home = Path.home()
    possible_paths = [
        home / 'anaconda3' / 'bin' / 'conda',
        home / 'miniconda3' / 'bin' / 'conda',
        home / '.conda' / 'bin' / 'conda',
        Path('/opt/conda/bin/conda'),
        Path('/usr/local/bin/conda'),
    ]
    for path in possible_paths:
        if path.exists():
            return str(path)

    return None


def get_conda_env_path(conda_env: str) -> str:
    """Get the PATH for a specific conda environment."""
    conda_exe = get_conda_executable()
    if not conda_exe:
        return os.environ.get('PATH', '')

    # Get conda base directory from conda executable location
    conda_base = Path(conda_exe).parent.parent

    # Common paths for conda env binaries
    possible_env_paths = [
        conda_base / 'envs' / conda_env / 'bin',
        Path.home() / '.conda' / 'envs' / conda_env / 'bin',
        Path('/opt/conda/envs') / conda_env / 'bin',
    ]

    # Find the environment bin directory
    env_bin_path = None
    for path in possible_env_paths:
        if path.exists():
            env_bin_path = path
            break

    # If not found, try using conda info
    if env_bin_path is None:
        result = subprocess.run(
            [conda_exe, 'info', '--envs', '--json'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            import json
            envs_info = json.loads(result.stdout)
            for env in envs_info.get('envs', []):
                if conda_env in env or env.endswith(conda_env):
                    env_bin_path = Path(env) / 'bin'
                    break

    if env_bin_path and env_bin_path.exists():
        # Prepend env bin to current PATH
        current_path = os.environ.get('PATH', '')
        return f"{env_bin_path}:{current_path}"

    return os.environ.get('PATH', '')


def run_in_conda(cmd: list, **kwargs) -> subprocess.CompletedProcess:
    """Run command in specified conda environment if set."""
    global CONDA_ENV

    if CONDA_ENV:
        # Get the environment's PATH and use it for the subprocess
        env_path = get_conda_env_path(CONDA_ENV)
        env = kwargs.get('env', os.environ.copy())
        env['PATH'] = env_path
        kwargs['env'] = env

        # Run with the modified environment
        return subprocess.run(cmd, **kwargs)

    return subprocess.run(cmd, **kwargs)

# Suppress excessive logging during tests
logging.basicConfig(level=logging.WARNING)


class TestEndToEndPipeline(unittest.TestCase):
    """End-to-end tests for the complete CryTransformer pipeline."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests."""
        cls.test_dir = Path(tempfile.mkdtemp(prefix="crydet_e2e_"))
        cls.data_manager = StubDataManager(str(cls.test_dir / "data"))

        # Create datasets
        cls.splits = cls.data_manager.create_train_val_test_split(
            train_cry=10, train_other=10,
            val_cry=4, val_other=4,
            test_cry=4, test_other=4
        )

        # Create config
        cls.config_path = cls.data_manager.create_minimal_config({
            "training": {
                "num_epochs": 2,
                "batch_size": 4,
                "num_workers": 0,  # No multiprocessing in tests
                "save_best_only": True
            }
        })

        cls.checkpoint_dir = cls.test_dir / "checkpoints"
        cls.checkpoint_dir.mkdir(exist_ok=True)
        cls.checkpoint_path = None

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        cls.data_manager.cleanup()
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)

    def test_01_train_from_scratch(self):
        """Test training from scratch with stub data."""
        print("\n[TEST] Training from scratch...")

        cmd = [
            sys.executable, "train.py",
            "--config", self.config_path,
            "--train_list", self.splits['train'],
            "--val_list", self.splits['val'],
            "--epochs", "2",
            "--batch_size", "4"
        ]

        result = run_in_conda(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode != 0:
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")

        self.assertEqual(result.returncode, 0, "Training should complete successfully")

        # Check that checkpoint was created in a timestamped subdirectory
        runs_dir = Path(self.data_manager.base_dir) / "runs"
        checkpoint_files = list(runs_dir.glob("**/best_model.pt"))
        self.assertTrue(len(checkpoint_files) > 0,
                       "Training should create checkpoint file")

        print("[PASS] Training completed")

    def test_02_create_mock_checkpoint(self):
        """Test creating a mock checkpoint for inference/eval testing."""
        print("\n[TEST] Creating mock checkpoint...")

        checkpoint_path = str(self.checkpoint_dir / "mock_model.pt")
        create_mock_checkpoint(checkpoint_path, self.config_path)

        self.assertTrue(Path(checkpoint_path).exists(), "Checkpoint should be created")

        # Verify checkpoint can be loaded
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        self.assertIn('model_state_dict', checkpoint)
        self.assertIn('config', checkpoint)

        print("[PASS] Mock checkpoint created")
        self.checkpoint_path = checkpoint_path

    def test_03_evaluate_model(self):
        """Test evaluation with mock checkpoint."""
        print("\n[TEST] Evaluating model...")

        self.test_02_create_mock_checkpoint()

        cmd = [
            sys.executable, "evaluate.py",
            "--checkpoint", self.checkpoint_path,
            "--test_list", self.splits['test'],
            "--batch_size", "4",
            "--device", "cpu"
        ]

        result = run_in_conda(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode != 0:
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")

        self.assertEqual(result.returncode, 0, "Evaluation should complete successfully")

        # Check output contains expected metrics
        output = result.stdout + result.stderr
        self.assertIn("Accuracy", output or "Results", "Should report accuracy")

        print("[PASS] Evaluation completed")

    def test_04_inference_single_file(self):
        """Test inference on single audio file."""
        print("\n[TEST] Single file inference...")

        checkpoint_path = str(self.checkpoint_dir / "mock_model.pt")
        if not Path(checkpoint_path).exists():
            create_mock_checkpoint(checkpoint_path, self.config_path)

        # Get a test audio file
        with open(self.splits['test']) as f:
            test_data = json.load(f)
        # Get directory path and find an actual audio file
        cry_dir = Path(test_data['cry'][0])
        test_files = list(cry_dir.glob('*.wav'))
        if not test_files:
            self.fail("No audio files found in test directory")
        test_file = str(test_files[0])

        cmd = [
            sys.executable, "inference.py",
            "--checkpoint", checkpoint_path,
            "--audio", test_file,
            "--device", "cpu"
        ]

        result = run_in_conda(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")

        self.assertEqual(result.returncode, 0, "Inference should complete successfully")

        output = result.stdout + result.stderr
        self.assertIn("Is Cry", output or "Probability", "Should report prediction")

        print("[PASS] Single file inference completed")

    def test_05_inference_batch(self):
        """Test batch inference with audio list."""
        print("\n[TEST] Batch inference...")

        checkpoint_path = str(self.checkpoint_dir / "mock_model.pt")
        if not Path(checkpoint_path).exists():
            create_mock_checkpoint(checkpoint_path, self.config_path)

        cmd = [
            sys.executable, "inference.py",
            "--checkpoint", checkpoint_path,
            "--audio_list", self.splits['test'],
            "--device", "cpu"
        ]

        result = run_in_conda(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode != 0:
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")

        self.assertEqual(result.returncode, 0, "Batch inference should complete")

        print("[PASS] Batch inference completed")

    def test_06_export_pytorch(self):
        """Test exporting to PyTorch format."""
        print("\n[TEST] Exporting to PyTorch...")

        checkpoint_path = str(self.checkpoint_dir / "mock_model.pt")
        if not Path(checkpoint_path).exists():
            create_mock_checkpoint(checkpoint_path, self.config_path)

        output_path = str(self.test_dir / "export_pytorch.pt")

        cmd = [
            sys.executable, "export.py",
            "--checkpoint", checkpoint_path,
            "--format", "pytorch",
            "--output", output_path
        ]

        result = run_in_conda(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")

        self.assertEqual(result.returncode, 0, "Export should complete")
        self.assertTrue(Path(output_path).exists(), "Exported file should exist")

        print("[PASS] PyTorch export completed")

    def test_07_export_size_report(self):
        """Test model size report generation."""
        print("\n[TEST] Generating size report...")

        checkpoint_path = str(self.checkpoint_dir / "mock_model.pt")
        if not Path(checkpoint_path).exists():
            create_mock_checkpoint(checkpoint_path, self.config_path)

        cmd = [
            sys.executable, "export.py",
            "--checkpoint", checkpoint_path,
            "--format", "pytorch",
            "--size_report"
        ]

        result = run_in_conda(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")

        self.assertEqual(result.returncode, 0, "Size report should generate")

        output = result.stdout + result.stderr
        self.assertIn("Model Size Report", output or "Total Params", "Should show size report")

        print("[PASS] Size report generated")


class TestStubDataGeneration(unittest.TestCase):
    """Tests for stub data generation utilities."""

    def test_generate_wav_cry(self):
        """Test generating cry-like audio."""
        from llt.stub_data import StubAudioGenerator

        with tempfile.TemporaryDirectory() as tmpdir:
            generator = StubAudioGenerator()
            path = os.path.join(tmpdir, "test_cry.wav")
            generator.generate_wav(path, is_cry=True)

            self.assertTrue(os.path.exists(path))
            self.assertGreater(os.path.getsize(path), 0)

    def test_generate_wav_noise(self):
        """Test generating noise audio."""
        from llt.stub_data import StubAudioGenerator

        with tempfile.TemporaryDirectory() as tmpdir:
            generator = StubAudioGenerator()
            path = os.path.join(tmpdir, "test_noise.wav")
            generator.generate_wav(path, is_cry=False)

            self.assertTrue(os.path.exists(path))

    def test_generate_dataset(self):
        """Test generating complete dataset."""
        from llt.stub_data import StubAudioGenerator

        with tempfile.TemporaryDirectory() as tmpdir:
            generator = StubAudioGenerator()
            dataset = generator.generate_dataset(tmpdir, num_cry=5, num_other=5)

            self.assertEqual(len(dataset['cry']), 5)
            self.assertEqual(len(dataset['other']), 5)

            for path in dataset['cry'] + dataset['other']:
                self.assertTrue(os.path.exists(path))

    def test_stub_data_manager(self):
        """Test StubDataManager context manager."""
        with StubDataManager() as manager:
            splits = manager.create_train_val_test_split(
                train_cry=3, train_other=3,
                val_cry=2, val_other=2,
                test_cry=2, test_other=2
            )

            self.assertIn('train', splits)
            self.assertIn('val', splits)
            self.assertIn('test', splits)

            # Verify JSON files are valid
            for split, path in splits.items():
                with open(path) as f:
                    data = json.load(f)
                self.assertIn('cry', data)
                self.assertIn('other', data)

    def test_config_creation(self):
        """Test minimal config generation."""
        with StubDataManager() as manager:
            config_path = manager.create_minimal_config()
            self.assertTrue(os.path.exists(config_path))

            # Verify config can be loaded
            config = load_config(config_path)
            self.assertEqual(config.model.d_model, 64)  # Small model for testing
            self.assertEqual(config.training.num_epochs, 2)


class TestTrainingComponents(unittest.TestCase):
    """Unit tests for individual training components using stub data."""

    def test_dataset_loading(self):
        """Test CryDataset with stub audio."""
        from dataset.dataset import CryDataset
        from utils.config import DatasetConfig

        with StubDataManager() as manager:
            # Generate dataset directories
            dataset_dict = manager.generator.generate_dataset(
                str(manager.base_dir / "test"),
                num_cry=3, num_other=3
            )

            # Create data_dict in format expected by CryDataset
            data_dict = {
                "cry": [str(manager.base_dir / "test" / "cry")],
                "other": [1, str(manager.base_dir / "test" / "other")]
            }

            config = DatasetConfig(
                sample_rate=16000,
                slice_len=3.0,
                stride=2.0
            )

            dataset = CryDataset(
                data_dict=data_dict,
                config=config,
                aug_config=None
            )

            # Build schedule to initialize file_schedule_dict
            dataset.build_schedule(shuffle=False)

            self.assertGreater(len(dataset), 0)

    def test_feature_extraction(self):
        """Test feature extraction on stub audio."""
        from dataset.feature import FeatureExtractor
        from utils.config import FeatureConfig

        config = FeatureConfig(n_mels=32, feature_type=3)
        extractor = FeatureExtractor(config, sr=16000)

        # Generate synthetic waveform
        waveform = torch.randn(1, 16000)  # 1 second @ 16kHz
        features = extractor(waveform)

        self.assertEqual(features.ndim, 3)  # [B, T, F]
        self.assertEqual(features.shape[0], 1)  # Batch
        self.assertEqual(features.shape[2], config.feature_dim)  # Feature dim

    def test_model_forward(self):
        """Test model forward pass with stub features."""
        from model import create_model
        from utils.config import ModelConfig, FeatureConfig

        feature_config = FeatureConfig(n_mels=32, feature_type=3)
        model_config = ModelConfig(d_model=64, n_layers=2, n_heads=2)

        model = create_model(
            config=model_config,
            in_channels=feature_config.feature_dim,
            num_classes=2
        )

        # Forward pass with random features
        batch_size = 2
        seq_len = 50
        x = torch.randn(batch_size, seq_len, feature_config.feature_dim)
        output = model(x)

        self.assertEqual(output.shape, (batch_size, 2))


def run_quick_smoke_test(conda_env: str = None):
    """Run a quick smoke test without unittest framework."""
    global CONDA_ENV
    if conda_env:
        CONDA_ENV = conda_env

    print("=" * 60)
    print("CRYDET SMOKE TEST")
    if CONDA_ENV:
        print(f"  Conda env: {CONDA_ENV}")
    print("=" * 60)

    with StubDataManager() as manager:
        print("\n1. Creating stub data...")
        splits = manager.create_train_val_test_split(
            train_cry=4, train_other=4,
            val_cry=2, val_other=2,
            test_cry=2, test_other=2
        )
        config_path = manager.create_minimal_config()
        print(f"   Data: {manager.base_dir}")

        print("\n2. Creating mock checkpoint...")
        checkpoint_path = str(manager.base_dir / "model.pt")
        create_mock_checkpoint(checkpoint_path, config_path)
        print(f"   Checkpoint: {checkpoint_path}")

        print("\n3. Testing evaluation...")
        cmd = [
            sys.executable, "evaluate.py",
            "--checkpoint", checkpoint_path,
            "--test_list", splits['test'],
            "--batch_size", "2",
            "--device", "cpu"
        ]
        result = run_in_conda(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            print("   [PASS] Evaluation")
        else:
            print(f"   [FAIL] Evaluation: {result.stderr[:200]}")

        print("\n4. Testing inference...")
        with open(splits['test']) as f:
            test_data = json.load(f)
        # Get directory path and find an actual audio file
        cry_dir = Path(test_data['cry'][0])
        test_files = list(cry_dir.glob('*.wav'))
        if not test_files:
            print("   [FAIL] No audio files found in test directory")
            return
        test_file = str(test_files[0])

        cmd = [
            sys.executable, "inference.py",
            "--checkpoint", checkpoint_path,
            "--audio", test_file,
            "--device", "cpu"
        ]
        result = run_in_conda(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("   [PASS] Inference")
        else:
            print(f"   [FAIL] Inference: {result.stderr[:200]}")

        print("\n5. Testing export...")
        cmd = [
            sys.executable, "export.py",
            "--checkpoint", checkpoint_path,
            "--format", "pytorch",
            "--output", str(manager.base_dir / "export.pt")
        ]
        result = run_in_conda(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("   [PASS] Export")
        else:
            print(f"   [FAIL] Export: {result.stderr[:200]}")

    print("\n" + "=" * 60)
    print("SMOKE TEST COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='End-to-end tests for CryTransformer')
    parser.add_argument('--smoke', action='store_true', help='Run quick smoke test only')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--conda-env', type=str, default=None,
                        help='Conda environment name to run subprocess commands in')
    args = parser.parse_args()

    # Set global conda environment from argument or environment variable
    if args.conda_env:
        CONDA_ENV = args.conda_env
    elif os.environ.get('CONDA_ENV'):
        CONDA_ENV = os.environ.get('CONDA_ENV')

    if CONDA_ENV:
        print(f"Using conda environment: {CONDA_ENV}")

    if args.smoke:
        run_quick_smoke_test(CONDA_ENV)
    else:
        # Run full unittest suite
        verbosity = 2 if args.verbose else 1
        unittest.main(verbosity=verbosity, exit=False)
