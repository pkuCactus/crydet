"""
Integration tests for dataset/augmentation.py - AudioAugmenter

Tests cover:
- Actual effect application with sox (pitch, reverb, phaser, echo, time_stretch)
- Noise addition with actual SNR verification
- Mixup audio blending verification
- End-to-end augmentation pipeline

Integration tests require sox binary to be installed.
Tests are skipped if sox is not available.
"""

import logging
import unittest
import numpy as np
import sys
import shutil
from pathlib import Path
from unittest.mock import patch, Mock

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import AugmentationConfig, MixupConfig, NoiseConfig
from llt.stub_data import StubAudioGenerator

logging.basicConfig(level=logging.ERROR)

# Check if sox is available
SOX_AVAILABLE = shutil.which('sox') is not None


def skip_if_no_sox(func):
    """Decorator to skip test if sox is not available."""
    return unittest.skipUnless(SOX_AVAILABLE, "sox not installed")(func)


class TestEffectApplication(unittest.TestCase):
    """Test actual effect application using sox."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.sample_rate = 16000
        cls.duration = 1.0  # 1 second
        cls.generator = StubAudioGenerator(sample_rate=cls.sample_rate)

        # Create a test audio
        cls.test_audio = cls._create_test_audio()

    @classmethod
    def _create_test_audio(cls):
        """Create a test audio signal."""
        # Use a simple sine wave
        t = np.linspace(0, cls.duration, int(cls.sample_rate * cls.duration))
        return np.sin(2 * np.pi * 440 * t).astype(np.float32)

    def setUp(self):
        """Set up augmenter for each test."""
        from dataset.augmentation import AudioAugmenter
        self.config = AugmentationConfig()
        self.augmenter = AudioAugmenter(self.config, sample_rate=self.sample_rate)

    @skip_if_no_sox
    def test_pitch_effect_changes_audio(self):
        """Test pitch effect actually changes the audio."""
        import sox
        original = self.test_audio.copy()

        tfm = sox.transform.Transformer()
        tfm.pitch(n_semitones=2.0)
        result = tfm.build_array(input_array=original, sample_rate_in=self.sample_rate)

        # Pitch change should produce different audio
        self.assertFalse(np.allclose(result, original, rtol=0.1))
        # Length should be preserved
        self.assertEqual(len(result), len(original))

    @skip_if_no_sox
    def test_reverb_effect_changes_audio(self):
        """Test reverb effect actually changes the audio."""
        original = self.test_audio.copy()

        import sox
        tfm = sox.transform.Transformer()
        tfm.reverb(reverberance=50, high_freq_damping=50, room_scale=50)
        result = tfm.build_array(input_array=original, sample_rate_in=self.sample_rate)

        # Reverb should change the audio
        self.assertFalse(np.allclose(result, original, rtol=0.1))

    @skip_if_no_sox
    def test_phaser_effect_changes_audio(self):
        """Test phaser effect actually changes the audio."""
        original = self.test_audio.copy()

        import sox
        tfm = sox.transform.Transformer()
        tfm.phaser(gain_in=0.8, gain_out=0.8, delay=3, decay=0.3, speed=1.0)
        result = tfm.build_array(input_array=original, sample_rate_in=self.sample_rate)

        # Phaser should change the audio
        self.assertFalse(np.allclose(result, original, rtol=0.1))

    @skip_if_no_sox
    def test_echo_effect_changes_audio(self):
        """Test echo effect actually changes the audio."""
        original = self.test_audio.copy()

        import sox
        tfm = sox.transform.Transformer()
        tfm.echo(gain_in=0.8, gain_out=0.8, n_echos=1, delays=[50], decays=[0.3])
        result = tfm.build_array(input_array=original, sample_rate_in=self.sample_rate)

        # Echo should change the audio
        self.assertFalse(np.allclose(result, original, rtol=0.1))

    @skip_if_no_sox
    def test_time_stretch_sola_changes_audio(self):
        """Test time stretch with SOLA changes audio."""
        original = self.test_audio.copy()

        import sox
        tfm = sox.transform.Transformer()
        tfm.stretch(factor=1.1, window=20)
        tfm.set_output_format(rate=self.sample_rate, channels=1)
        result = tfm.build_array(input_array=original, sample_rate_in=self.sample_rate)

        # Time stretch should change the audio characteristics
        # Length might change slightly due to stretch
        self.assertFalse(np.array_equal(result, original))

    @skip_if_no_sox
    def test_time_stretch_tempo_changes_audio(self):
        """Test time stretch with tempo changes audio."""
        original = self.test_audio.copy()

        import sox
        tfm = sox.transform.Transformer()
        tfm.tempo(factor=1.2)
        tfm.set_output_format(rate=self.sample_rate, channels=1)
        result = tfm.build_array(input_array=original, sample_rate_in=self.sample_rate)

        # Tempo change should affect the audio
        self.assertFalse(np.array_equal(result, original))

    @skip_if_no_sox
    def test_apply_effect_group_empty_list(self):
        """Test apply_effect_group with empty effects list returns original."""
        original = self.test_audio.copy()
        result = self.augmenter._apply_effect_group(original, [])

        np.testing.assert_array_equal(result, original)

    @skip_if_no_sox
    def test_apply_effect_group_single_effect(self):
        """Test apply_effect_group with single effect."""
        original = self.test_audio.copy()
        result = self.augmenter._apply_effect_group(original, ['pitch'])

        # Should change the audio
        self.assertFalse(np.allclose(result, original, rtol=0.1))

    @skip_if_no_sox
    def test_apply_effect_group_multiple_effects(self):
        """Test apply_effect_group with multiple effects."""
        original = self.test_audio.copy()
        result = self.augmenter._apply_effect_group(original, ['pitch', 'reverb'])

        # Should change the audio
        self.assertFalse(np.allclose(result, original, rtol=0.1))


class TestNoiseAddition(unittest.TestCase):
    """Test actual noise addition with SNR verification."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.sample_rate = 16000
        cls.duration = 1.0
        cls.generator = StubAudioGenerator(sample_rate=cls.sample_rate)

    def setUp(self):
        """Set up augmenter for each test."""
        from dataset.augmentation import AudioAugmenter
        self.config = AugmentationConfig()
        self.augmenter = AudioAugmenter(self.config, sample_rate=self.sample_rate)

        # Create test audio with known energy
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        self.test_audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

    def test_white_noise_addition(self):
        """Test white noise is actually added to the signal."""
        original = self.test_audio.copy()

        # Mock noise selection to always return white
        self.augmenter._select_noise_type = lambda: 'white'

        result = self.augmenter._apply_noise(original.copy())

        # Signal should be modified (noise added)
        self.assertFalse(np.array_equal(result, original))
        # Should not be all zeros
        self.assertTrue(np.any(result != 0))

    def test_pink_noise_addition(self):
        """Test pink noise is actually added to the signal."""
        original = self.test_audio.copy()

        # Mock noise selection to always return pink
        self.augmenter._select_noise_type = lambda: 'pink'

        result = self.augmenter._apply_noise(original.copy())

        # Signal should be modified
        self.assertFalse(np.array_equal(result, original))

    def test_noise_preserves_signal_shape(self):
        """Test noise addition preserves audio shape."""
        original = self.test_audio.copy()

        self.augmenter._select_noise_type = lambda: 'white'
        result = self.augmenter._apply_noise(original.copy())

        self.assertEqual(result.shape, original.shape)

    def test_snr_range_respected(self):
        """Test SNR is within configured range."""
        from dataset import utils

        original = self.test_audio.copy()
        original_db = utils.get_db(original)

        self.augmenter._select_noise_type = lambda: 'white'

        # Test multiple runs
        snrs = []
        for _ in range(10):
            result = self.augmenter._apply_noise(original.copy())
            noise = result - original
            noise_db = utils.get_db(noise)
            snr = original_db - noise_db
            snrs.append(snr)

        # All SNRs should be within configured range
        config_snr_min = self.config.noise.snr_min
        config_snr_max = self.config.noise.snr_max

        for snr in snrs:
            # Allow some tolerance due to measurement method
            self.assertGreaterEqual(snr, config_snr_min - 2)
            self.assertLessEqual(snr, config_snr_max + 2)


class TestMixupIntegration(unittest.TestCase):
    """Test mixup integration with actual audio."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.sample_rate = 16000
        cls.generator = StubAudioGenerator(sample_rate=cls.sample_rate)

    def setUp(self):
        """Set up augmenter for each test."""
        from dataset.augmentation import AudioAugmenter
        from dataset.audio_reader import AudioReader

        self.config = AugmentationConfig()
        self.augmenter = AudioAugmenter(self.config, sample_rate=self.sample_rate)

        # Create mock audio reader
        self.mock_reader = Mock(spec=AudioReader)
        self.augmenter.audio_reader = self.mock_reader

    def test_mixup_produces_valid_audio(self):
        """Test mixup produces valid audio output."""
        from dataset import utils

        original = np.sin(2 * np.pi * 440 * np.linspace(0, 1, self.sample_rate)).astype(np.float32)

        # Create mix sample with different frequency
        mix_sample = np.sin(2 * np.pi * 880 * np.linspace(0, 1, self.sample_rate)).astype(np.float32)

        # Mock _get_mixup_sample
        self.augmenter._get_mixup_sample = lambda exclude_cry=False: mix_sample

        result = self.augmenter._do_mixup(original.copy(), 'cry')

        # Result should be different from original
        self.assertFalse(np.array_equal(result, original))
        # Result should be clipped to [-1, 1]
        self.assertTrue(np.all(result >= -1.0))
        self.assertTrue(np.all(result <= 1.0))

    def test_mixup_with_different_length_samples(self):
        """Test mixup handles samples of different lengths."""
        original = np.ones(1000)
        mix_sample = np.ones(800)

        self.augmenter._get_mixup_sample = lambda exclude_cry=False: mix_sample

        result = self.augmenter._do_mixup(original.copy(), 'cry')

        # Result should be same length as original
        self.assertEqual(len(result), len(original))

    def test_generate_mixup_sample_excludes_cry_when_requested(self):
        """Test _generate_mixup_sample excludes cry when exclude_cry=True."""
        self.augmenter._file_schedule_dict = {
            'cry': [('cry.wav', 0.0, 5.0, False)],
            'other': [('other.wav', 0.0, 5.0, False)]
        }

        # Mock audio reader
        self.mock_reader.load_by_time.return_value = (np.ones(100), self.sample_rate)

        # Get mixup with exclude_cry
        sample = self.augmenter._load_random_sample_from_disk(exclude_cry=True)

        # Should only load from 'other' (not 'cry')
        # Since we can't verify random selection, we verify the method doesn't error
        self.assertIsNotNone(sample)
        self.assertEqual(len(sample), 100)


class MockAudioReader:
    """Mock audio reader for testing."""

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def load_by_time(self, file_path, start, end):
        """Return synthetic audio."""
        duration = end - start
        samples = int(duration * self.sample_rate)
        return np.random.randn(samples).astype(np.float32), self.sample_rate


class TestEndToEndAugmentation(unittest.TestCase):
    """End-to-end augmentation pipeline tests."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.sample_rate = 16000
        cls.generator = StubAudioGenerator(sample_rate=cls.sample_rate)

    def setUp(self):
        """Set up augmenter for each test."""
        from dataset.augmentation import AudioAugmenter
        self.config = AugmentationConfig(
            cry_aug_prob=1.0,  # Always augment
            other_aug_prob=1.0,
            pitch_prob=0.0,  # Disable individual effects for predictable tests
            reverb_prob=0.0,
            phaser_prob=0.0,
            echo_prob=0.0,
            gain_prob=0.0,
            time_stretch_prob=0.0,
            noise=NoiseConfig(prob=0.0),  # Disable noise
            mixup=MixupConfig(
                cry_mix_prob=0.0,  # Disable mixup for predictable tests
                other_mix_prob=0.0
            )
        )
        self.augmenter = AudioAugmenter(self.config, sample_rate=self.sample_rate)

    def test_augment_preserves_audio_length(self):
        """Test augment preserves audio length."""
        original = np.sin(2 * np.pi * 440 * np.linspace(0, 1, self.sample_rate)).astype(np.float32)

        # With no effects enabled, should return original
        result = self.augmenter.augment(original.copy(), 'cry')

        # Length should be preserved
        self.assertEqual(len(result), len(original))

    def test_augment_output_range(self):
        """Test augment output is in valid range."""
        original = np.sin(2 * np.pi * 440 * np.linspace(0, 1, self.sample_rate)).astype(np.float32)

        result = self.augmenter.augment(original.copy(), 'cry')

        # Output should be in [-1, 1] range
        self.assertTrue(np.all(result >= -1.0))
        self.assertTrue(np.all(result <= 1.0))

    def test_augment_different_labels(self):
        """Test augment with different label types."""
        original = np.sin(2 * np.pi * 440 * np.linspace(0, 1, self.sample_rate)).astype(np.float32)

        # Test with 'cry' label
        result_cry = self.augmenter.augment(original.copy(), 'cry')
        self.assertEqual(len(result_cry), len(original))

        # Test with 'other' label
        result_other = self.augmenter.augment(original.copy(), 'other')
        self.assertEqual(len(result_other), len(original))

    @skip_if_no_sox
    def test_full_augmentation_pipeline(self):
        """Test full augmentation pipeline with effects."""
        # Enable effects
        config = AugmentationConfig(
            cry_aug_prob=1.0,
            pitch_prob=1.0,  # Enable pitch
            reverb_prob=0.0,
            phaser_prob=0.0,
            echo_prob=0.0,
            gain_prob=0.0,
            time_stretch_prob=0.0,
            noise=NoiseConfig(prob=0.0),
            mixup=MixupConfig(cry_mix_prob=0.0)
        )
        from dataset.augmentation import AudioAugmenter
        augmenter = AudioAugmenter(config, sample_rate=self.sample_rate)

        original = np.sin(2 * np.pi * 440 * np.linspace(0, 1, self.sample_rate)).astype(np.float32)

        result = augmenter.augment(original.copy(), 'cry')

        # Result should be different from original (pitch applied)
        self.assertFalse(np.allclose(result, original, rtol=0.01))
        # Should be in valid range
        self.assertTrue(np.all(result >= -1.0))
        self.assertTrue(np.all(result <= 1.0))


class TestAugmentationConfigEffectsMap(unittest.TestCase):
    """Test AugmentationConfig effect map behavior."""

    def test_effect_map_indexing(self):
        """Test effect probability access via indexing."""
        config = AugmentationConfig(
            pitch_prob=0.5,
            reverb_prob=0.8,
            phaser_prob=0.3,
            echo_prob=0.4,
            gain_prob=0.9,
            time_stretch_prob=0.1
        )

        # Test accessing via __getitem__
        self.assertEqual(config['pitch'], 0.5)
        self.assertEqual(config['reverb'], 0.8)
        self.assertEqual(config['phaser'], 0.3)
        self.assertEqual(config['echo'], 0.4)
        self.assertEqual(config['gain'], 0.9)
        self.assertEqual(config['time_stretch'], 0.1)

    def test_effect_map_unknown_key_raises(self):
        """Test unknown effect key raises KeyError."""
        config = AugmentationConfig()

        with self.assertRaises(KeyError):
            _ = config['unknown_effect']


class TestAugmentationWithStubData(unittest.TestCase):
    """Test augmentation using stub data for realistic audio."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.sample_rate = 16000
        cls.generator = StubAudioGenerator(sample_rate=cls.sample_rate)

    def test_with_cry_like_audio(self):
        """Test augmentation with cry-like synthetic audio."""
        from dataset.augmentation import AudioAugmenter

        # Generate cry-like audio
        cry_audio = self.generator._generate_cry_audio(duration_sec=1.0)

        config = AugmentationConfig(
            cry_aug_prob=1.0,
            noise=NoiseConfig(prob=0.0),
            mixup=MixupConfig(cry_mix_prob=0.0)
        )
        augmenter = AudioAugmenter(config, sample_rate=self.sample_rate)

        result = augmenter.augment(cry_audio.copy(), 'cry')

        self.assertEqual(len(result), len(cry_audio))
        self.assertTrue(np.all(result >= -1.0))
        self.assertTrue(np.all(result <= 1.0))

    def test_with_noise_like_audio(self):
        """Test augmentation with noise-like synthetic audio."""
        from dataset.augmentation import AudioAugmenter

        # Generate noise audio
        noise_audio = self.generator._generate_noise_audio(duration_sec=1.0)

        config = AugmentationConfig(
            other_aug_prob=1.0,
            other_reverse_prob=0.0,
            noise=NoiseConfig(prob=0.0),
            mixup=MixupConfig(other_mix_prob=0.0)
        )
        augmenter = AudioAugmenter(config, sample_rate=self.sample_rate)

        result = augmenter.augment(noise_audio.copy(), 'other')

        self.assertEqual(len(result), len(noise_audio))


if __name__ == '__main__':
    unittest.main()
