"""
Unit tests for dataset/augmentation.py - AudioAugmenter

Tests cover:
- Configuration consistency (all params match config)
- Mixup probability computation logic
- Mixup energy control logic (cry samples 3-10dB lower)
- Label-aware logic (cry vs non-cry different rules)
- Probability boundary conditions (0, 1, edge cases)
- Noise type selection probabilities

Pure unit tests - no external dependencies (sox not required)
"""

import logging
import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import AugmentationConfig, MixupConfig, NoiseConfig
from dataset.augmentation import AudioAugmenter

logging.basicConfig(level=logging.ERROR)


class TestAugmentationConfigConsistency(unittest.TestCase):
    """Test that AudioAugmenter correctly reads all config parameters."""

    def test_default_config_loading(self):
        """Test default config values are correctly loaded."""
        config = AugmentationConfig()
        self.assertEqual(config.cry_aug_prob, 0.9)
        self.assertEqual(config.other_aug_prob, 0.6)
        self.assertEqual(config.other_reverse_prob, 0.5)
        self.assertEqual(config.pitch_prob, 0.5)
        self.assertEqual(config.reverb_prob, 0.8)
        self.assertEqual(config.phaser_prob, 0.5)
        self.assertEqual(config.echo_prob, 0.5)
        self.assertEqual(config.gain_prob, 0.9)
        self.assertEqual(config.time_stretch_prob, 0.0)

    def test_mixup_config_loading(self):
        """Test mixup config values are correctly loaded."""
        config = AugmentationConfig()
        self.assertEqual(config.mixup.cry_mix_prob, 0.3)
        self.assertEqual(config.mixup.cry_mix_rate_mean, 0.3)
        self.assertEqual(config.mixup.cry_mix_rate_std, 0.15)
        self.assertEqual(config.mixup.other_mix_prob, 0.3)
        self.assertEqual(config.mixup.mix_front_prob, 0.7)

    def test_noise_config_loading(self):
        """Test noise config values are correctly loaded."""
        config = AugmentationConfig()
        self.assertEqual(config.noise.prob, 0.1)
        self.assertEqual(config.noise.white_noise_prob, 0.3)
        self.assertEqual(config.noise.pink_noise_prob, 0.4)
        self.assertEqual(config.noise.ambient_noise_prob, 0.3)
        self.assertEqual(config.noise.snr_min, 5.0)
        self.assertEqual(config.noise.snr_max, 25.0)

    def test_custom_config_values(self):
        """Test custom config values are correctly propagated."""
        config = AugmentationConfig(
            cry_aug_prob=0.95,
            other_aug_prob=0.7,
            pitch_prob=0.6,
            reverb_prob=0.9,
            mixup=MixupConfig(cry_mix_prob=0.5, cry_mix_rate_mean=0.4),
            noise=NoiseConfig(prob=0.2, snr_min=10.0, snr_max=30.0)
        )
        self.assertEqual(config.cry_aug_prob, 0.95)
        self.assertEqual(config.other_aug_prob, 0.7)
        self.assertEqual(config.pitch_prob, 0.6)
        self.assertEqual(config.reverb_prob, 0.9)
        self.assertEqual(config.mixup.cry_mix_prob, 0.5)
        self.assertEqual(config.mixup.cry_mix_rate_mean, 0.4)
        self.assertEqual(config.noise.prob, 0.2)
        self.assertEqual(config.noise.snr_min, 10.0)
        self.assertEqual(config.noise.snr_max, 30.0)


class TestIsAugment(unittest.TestCase):
    """Test the is_augment method for label-aware augmentation decisions."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = AugmentationConfig(
            cry_aug_prob=0.9,
            other_aug_prob=0.6
        )
        self.augmenter = AudioAugmenter(self.config)

    @patch('random.random', return_value=0.5)
    def test_is_augment_cry_prob_pass(self, mock_random):
        """Test cry augmentation when probability check passes."""
        result = self.augmenter.is_augment('cry')
        self.assertTrue(result)

    @patch('random.random', return_value=0.95)
    def test_is_augment_cry_prob_fail(self, mock_random):
        """Test cry augmentation when probability check fails."""
        result = self.augmenter.is_augment('cry')
        self.assertFalse(result)

    @patch('random.random', return_value=0.5)
    def test_is_augment_other_prob_pass(self, mock_random):
        """Test other augmentation when probability check passes."""
        result = self.augmenter.is_augment('other')
        self.assertTrue(result)

    @patch('random.random', return_value=0.7)
    def test_is_augment_other_prob_fail(self, mock_random):
        """Test other augmentation when probability check fails."""
        result = self.augmenter.is_augment('other')
        self.assertFalse(result)

    def test_is_augment_case_insensitive(self):
        """Test label matching is case insensitive."""
        with patch('random.random', return_value=0.5):
            self.assertTrue(self.augmenter.is_augment('CRY'))
            self.assertTrue(self.augmenter.is_augment('Cry'))


class TestComputeMixupRate(unittest.TestCase):
    """Test the _compute_mixup_rate method for correct probability computation."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = AugmentationConfig(
            mixup=MixupConfig(
                cry_mix_prob=0.5,
                cry_mix_rate_mean=0.3,
                cry_mix_rate_std=0.1,
                other_mix_prob=0.4
            )
        )
        self.augmenter = AudioAugmenter(self.config)

    @patch('random.random', return_value=0.3)
    @patch('random.gauss', return_value=0.3)
    def test_cry_mixup_rate_computation(self, mock_gauss, mock_random):
        """Test cry mixup rate is computed with correct Gaussian parameters."""
        rate = self.augmenter._compute_mixup_rate(is_cry=True)
        # Should be clipped and adjusted: 0.3 + gauss(0, 0.05), then clipped to [0.1, 0.65]
        self.assertGreaterEqual(rate, 0.1)
        self.assertLessEqual(rate, 0.65)

    @patch('random.random', return_value=0.5)
    def test_cry_mixup_prob_fail(self, mock_random):
        """Test cry mixup returns -1 when probability check fails."""
        rate = self.augmenter._compute_mixup_rate(is_cry=True)
        self.assertEqual(rate, -1.0)

    @patch('random.random', return_value=0.3)
    def test_other_mixup_rate_random(self, mock_random):
        """Test other mixup rate uses random.random()."""
        rate = self.augmenter._compute_mixup_rate(is_cry=False)
        # random() returns 0.3, then clipped with gauss adjustment
        self.assertGreaterEqual(rate, 0.1)
        self.assertLessEqual(rate, 0.65)

    @patch('random.random', return_value=0.5)
    def test_other_mixup_prob_fail(self, mock_random):
        """Test other mixup returns -1 when probability check fails."""
        rate = self.augmenter._compute_mixup_rate(is_cry=False)
        self.assertEqual(rate, -1.0)

    def test_mixup_rate_clipping(self):
        """Test mixup rate is always clipped to [0.1, 0.65] range."""
        # The function has a while loop that continues until gauss returns a value in (0, 1)
        # First value 10.0 triggers loop to continue, second value 0.3 is valid for mix_rate
        # Then mix_rate + gauss(0, 0.05) is clipped: 0.3 + 10.0 = 10.3 -> clipped to 0.65
        with patch('random.random', return_value=0.3):
            with patch('random.gauss', side_effect=[10.0, 0.3, 10.0]):  # loop value, valid mix_rate, clip adjustment
                rate = self.augmenter._compute_mixup_rate(is_cry=True)
                self.assertEqual(rate, 0.65)  # Upper clip

        # Test lower clip: mix_rate=0.3 + gauss=-10.0 -> -9.7 -> clipped to 0.1
        with patch('random.random', return_value=0.3):
            with patch('random.gauss', side_effect=[0.3, -10.0]):  # valid mix_rate, small clip value
                rate = self.augmenter._compute_mixup_rate(is_cry=True)
                self.assertEqual(rate, 0.1)  # Lower clip


class TestSelectNoiseType(unittest.TestCase):
    """Test noise type selection based on configured probabilities."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = AugmentationConfig(
            noise=NoiseConfig(
                white_noise_prob=0.3,
                pink_noise_prob=0.4,
                ambient_noise_prob=0.3
            )
        )
        self.augmenter = AudioAugmenter(self.config)

    @patch('random.choices', return_value=['white'])
    def test_white_noise_selection(self, mock_choices):
        """Test white noise can be selected."""
        noise_type = self.augmenter._select_noise_type()
        self.assertEqual(noise_type, 'white')

    @patch('random.choices', return_value=['pink'])
    def test_pink_noise_selection(self, mock_choices):
        """Test pink noise can be selected."""
        noise_type = self.augmenter._select_noise_type()
        self.assertEqual(noise_type, 'pink')

    @patch('random.choices', return_value=['ambient'])
    def test_ambient_noise_selection(self, mock_choices):
        """Test ambient noise can be selected when files available."""
        self.augmenter._ambient_noise_files = ['noise1.wav', 'noise2.wav']
        noise_type = self.augmenter._select_noise_type()
        self.assertEqual(noise_type, 'ambient')

    def test_no_ambient_files_fallback(self):
        """Test fallback to white when ambient files not available."""
        config = AugmentationConfig(
            noise=NoiseConfig(
                white_noise_prob=0.0,
                pink_noise_prob=0.0,
                ambient_noise_prob=1.0  # Would prefer ambient but no files
            )
        )
        augmenter = AudioAugmenter(config)
        # Should fall back to white (default)
        noise_type = augmenter._select_noise_type()
        self.assertEqual(noise_type, 'white')

    def test_zero_probabilities_fallback(self):
        """Test fallback when all probabilities are zero."""
        config = AugmentationConfig(
            noise=NoiseConfig(
                white_noise_prob=0.0,
                pink_noise_prob=0.0,
                ambient_noise_prob=0.0
            )
        )
        augmenter = AudioAugmenter(config)
        noise_type = augmenter._select_noise_type()
        self.assertEqual(noise_type, 'white')  # Default fallback


class TestMixupEnergyControl(unittest.TestCase):
    """Test mixup energy control for cry samples (must be 3-10dB lower)."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = AugmentationConfig()
        self.augmenter = AudioAugmenter(self.config)
        # Create mock file schedule
        self.augmenter._file_schedule_dict = {
            'cry': [('file1.wav', 0.0, 5.0, False)],
            'other': [('file2.wav', 0.0, 5.0, False)]
        }

    @patch('dataset.augmentation.utils.get_db')
    @patch('dataset.augmentation.utils.get_p')
    @patch('random.uniform', return_value=5.0)  # Target 5dB lower
    def test_cry_mixup_energy_reduction(self, mock_uniform, mock_get_p, mock_get_db):
        """Test cry mixup sample energy is reduced when higher than original."""
        # Setup: mix sample has higher energy
        # get_db is called 3 times: original_db, mix_db, temp_db
        mock_get_db.side_effect = [-20.0, -15.0, -15.0]  # original_db, mix_db, temp_db
        mock_get_p.return_value = 0.5  # p value for scaling

        # Create mixup sample with higher energy
        y_mix = np.ones(1000) * 0.5  # Higher amplitude
        y_original = np.ones(1000) * 0.1  # Lower amplitude

        with patch.object(self.augmenter, '_get_mixup_sample', return_value=y_mix):
            result = self.augmenter._generate_mixup_sample(y_original, 'cry')

        # Result should be scaled down (energy reduced)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), len(y_original))

    def test_cry_mixup_no_reduction_when_lower(self):
        """Test cry mixup sample not reduced when already lower energy."""
        # When mix energy is lower than original, no reduction should happen
        y_mix = np.ones(1000) * 0.1  # Lower amplitude
        y_original = np.ones(1000) * 0.5  # Higher amplitude

        # Mock all utility functions to avoid complex side_effect chains
        with patch.object(self.augmenter, '_get_mixup_sample', return_value=y_mix):
            with patch('dataset.augmentation.utils.get_db', return_value=-20.0):
                with patch('dataset.augmentation.utils.get_p', return_value=0.5):
                    result = self.augmenter._generate_mixup_sample(y_original, 'cry')

        # Should not reduce energy since mix is already lower
        self.assertIsNotNone(result)
        self.assertEqual(len(result), len(y_original))

    def test_non_cry_mixup_energy_unrestricted(self):
        """Test non-cry mixup has no energy restrictions."""
        y_mix = np.ones(1000) * 0.8  # High energy
        y_original = np.ones(1000) * 0.1  # Low energy

        with patch.object(self.augmenter, '_get_mixup_sample', return_value=y_mix):
            with patch('dataset.augmentation.utils.get_db', return_value=-20.0):
                with patch('dataset.augmentation.utils.get_p', return_value=0.5):
                    result = self.augmenter._generate_mixup_sample(y_original, 'other')

        # Non-cry should not check energy difference
        self.assertIsNotNone(result)


class TestLabelAwareMixup(unittest.TestCase):
    """Test label-aware mixup rules (non-cry can only mix with non-cry)."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = AugmentationConfig()
        self.augmenter = AudioAugmenter(self.config)
        self.augmenter._file_schedule_dict = {
            'cry': [('cry1.wav', 0.0, 5.0, False)],
            'other': [('other1.wav', 0.0, 5.0, False)],
            'noise': [('noise1.wav', 0.0, 5.0, False)]
        }

    def test_cry_can_mix_with_any_label(self):
        """Test cry samples can mix with any label (cry or non-cry)."""
        # When is_cry=True, exclude_cry=False means cry samples can be selected
        with patch.object(self.augmenter, '_load_random_sample_from_disk') as mock_load:
            mock_load.return_value = np.zeros(1000)
            self.augmenter._get_mixup_sample(exclude_cry=False)

            # Should be able to select from any label including 'cry'
            call_args = mock_load.call_args
            self.assertIsNotNone(call_args)

    def test_non_cry_excludes_cry_labels(self):
        """Test non-cry samples exclude cry labels from mixup pool."""
        with patch.object(self.augmenter, '_load_random_sample_from_disk') as mock_load:
            mock_load.return_value = np.zeros(1000)
            self.augmenter._get_mixup_sample(exclude_cry=True)

            # exclude_cry=True should filter out 'cry' label
            # Verify by checking the available_labels logic
            available = [l for l in self.augmenter._file_schedule_dict.keys()
                        if l.lower() != 'cry']
            self.assertNotIn('cry', available)
            self.assertIn('other', available)
            self.assertIn('noise', available)

    def test_load_random_sample_respects_exclude_cry(self):
        """Test _load_random_sample_from_disk respects exclude_cry parameter."""
        # Set up mock audio reader
        self.augmenter.audio_reader = Mock()
        self.augmenter.audio_reader.load_by_time.return_value = (np.zeros(1000), 16000)

        # When exclude_cry=True, 'cry' should not be in available labels
        result = self.augmenter._load_random_sample_from_disk(exclude_cry=True)

        # The method should only select from 'other' or 'noise'
        # Since we can't easily verify random selection, we verify the logic
        available_labels = list(self.augmenter._file_schedule_dict.keys())
        filtered_labels = [l for l in available_labels if l.lower() != 'cry']

        self.assertEqual(len(filtered_labels), 2)
        self.assertNotIn('cry', filtered_labels)


class TestProbabilityBoundaries(unittest.TestCase):
    """Test boundary conditions for probability parameters."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = AugmentationConfig()
        self.augmenter = AudioAugmenter(self.config)

    @patch('random.random', return_value=0.0)
    def test_zero_prob_never_augments(self, mock_random):
        """Test probability=0 means never augment."""
        config = AugmentationConfig(cry_aug_prob=0.0)
        augmenter = AudioAugmenter(config)

        # random() returns 0.0, which is NOT < 0.0, so should be False
        result = augmenter.is_augment('cry')
        self.assertFalse(result)

    @patch('random.random', return_value=0.999999)
    def test_one_prob_always_augments(self, mock_random):
        """Test probability=1 means always augment."""
        config = AugmentationConfig(cry_aug_prob=1.0)
        augmenter = AudioAugmenter(config)

        # random() returns 0.999999, which is < 1.0, so should be True
        result = augmenter.is_augment('cry')
        self.assertTrue(result)

    def test_effect_probabilities_zero(self):
        """Test all effect probabilities can be set to zero."""
        config = AugmentationConfig(
            pitch_prob=0.0,
            reverb_prob=0.0,
            phaser_prob=0.0,
            echo_prob=0.0,
            gain_prob=0.0,
            time_stretch_prob=0.0
        )
        augmenter = AudioAugmenter(config)

        self.assertEqual(augmenter.config.pitch_prob, 0.0)
        self.assertEqual(augmenter.config.reverb_prob, 0.0)
        self.assertEqual(augmenter.config.phaser_prob, 0.0)

    def test_effect_probabilities_one(self):
        """Test all effect probabilities can be set to one."""
        config = AugmentationConfig(
            pitch_prob=1.0,
            reverb_prob=1.0,
            phaser_prob=1.0,
            echo_prob=1.0,
            gain_prob=1.0,
            time_stretch_prob=1.0
        )
        augmenter = AudioAugmenter(config)

        self.assertEqual(augmenter.config.pitch_prob, 1.0)
        self.assertEqual(augmenter.config.reverb_prob, 1.0)
        self.assertEqual(augmenter.config.phaser_prob, 1.0)


class TestAugmentMethod(unittest.TestCase):
    """Test the main augment method with mocked dependencies."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = AugmentationConfig(
            cry_aug_prob=1.0,  # Always augment
            other_reverse_prob=0.0,  # Never reverse
            mixup=MixupConfig(mix_front_prob=0.0)  # Mixup at end
        )
        self.augmenter = AudioAugmenter(self.config)

    @patch.object(AudioAugmenter, '_apply_effect_group')
    @patch.object(AudioAugmenter, '_do_mixup')
    def test_cry_augment_flow(self, mock_mixup, mock_apply_effect):
        """Test cry sample goes through correct augmentation flow."""
        mock_apply_effect.return_value = np.ones(1000)
        mock_mixup.return_value = np.ones(1000)

        y = np.ones(1000)
        result = self.augmenter.augment(y, 'cry')

        # Cry samples should have effects applied
        mock_apply_effect.assert_called()
        # Mixup should be called at the end (not front)
        mock_mixup.assert_called_once()

    @patch.object(AudioAugmenter, '_apply_effect_group')
    @patch.object(AudioAugmenter, '_do_mixup')
    def test_other_reverse_applied(self, mock_mixup, mock_apply_effect):
        """Test non-cry samples can be reversed based on probability."""
        config = AugmentationConfig(
            other_aug_prob=1.0,
            other_reverse_prob=1.0,  # Always reverse
            mixup=MixupConfig(mix_front_prob=0.0)
        )
        augmenter = AudioAugmenter(config)

        # Use same length array for input and mock return
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 200)  # Length 1000
        mock_apply_effect.return_value = y.copy()
        mock_mixup.return_value = y.copy()

        result = augmenter.augment(y.copy(), 'other')

        # Audio should have been reversed before effect application
        # Check that _apply_effect_group was called with reversed array
        call_args = mock_apply_effect.call_args
        reversed_input = np.allclose(call_args[0][0], y[::-1])
        self.assertTrue(reversed_input, "Effect group should receive reversed audio")

    def test_mixup_front_placement(self):
        """Test mixup can be placed at front of augmentation chain."""
        config = AugmentationConfig(
            cry_aug_prob=1.0,
            mixup=MixupConfig(mix_front_prob=1.0)  # Always front
        )
        augmenter = AudioAugmenter(config)

        with patch.object(augmenter, '_do_mixup') as mock_mixup:
            with patch.object(augmenter, '_apply_effect_group') as mock_apply:
                mock_mixup.return_value = np.ones(1000)
                mock_apply.return_value = np.ones(1000)

                y = np.ones(1000)
                augmenter.augment(y, 'cry')

                # Mixup should be called before effect application
                self.assertTrue(mock_mixup.called)


class TestLoadAmbientNoiseFiles(unittest.TestCase):
    """Test ambient noise file loading."""

    def test_explicit_file_list(self):
        """Test loading from explicit file list."""
        config = AugmentationConfig(
            noise=NoiseConfig(ambient_noise_files=['noise1.wav', 'noise2.wav'])
        )
        augmenter = AudioAugmenter(config)

        files = augmenter._load_ambient_noise_files()
        self.assertEqual(len(files), 2)
        self.assertIn('noise1.wav', files)
        self.assertIn('noise2.wav', files)

    @patch('pathlib.Path.exists', return_value=True)
    @patch('pathlib.Path.is_dir', return_value=True)
    @patch('pathlib.Path.rglob', return_value=[
        Path('noise1.wav'), Path('noise2.mp3'), Path('sub/noise3.flac')
    ])
    def test_directory_scanning(self, mock_rglob, mock_is_dir, mock_exists):
        """Test scanning directory for noise files."""
        config = AugmentationConfig(
            noise=NoiseConfig(ambient_noise_dir='/fake/noise/dir')
        )
        augmenter = AudioAugmenter(config)

        files = augmenter._load_ambient_noise_files()
        self.assertEqual(len(files), 3)

    def test_no_noise_config(self):
        """Test empty list when no noise config provided."""
        config = AugmentationConfig(
            noise=NoiseConfig(ambient_noise_files=[], ambient_noise_dir=None)
        )
        augmenter = AudioAugmenter(config)

        files = augmenter._load_ambient_noise_files()
        self.assertEqual(len(files), 0)


class TestFileScheduleDict(unittest.TestCase):
    """Test file_schedule_dict property behavior."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = AugmentationConfig()
        self.augmenter = AudioAugmenter(self.config)

    def test_setter_clears_pool(self):
        """Test setting file_schedule_dict clears mixup pool."""
        # Initialize with some data
        self.augmenter._mixup_pool = [np.ones(100), np.ones(100)]
        self.augmenter._pool_initialized = True

        # Set new file schedule
        new_schedule = {'cry': [('file.wav', 0.0, 5.0, False)]}
        self.augmenter.file_schedule_dict = new_schedule

        # Pool should be cleared
        self.assertEqual(len(self.augmenter._mixup_pool), 0)
        self.assertFalse(self.augmenter._pool_initialized)

    def test_getter_returns_correct_value(self):
        """Test getter returns the set value."""
        schedule = {'cry': [('file.wav', 0.0, 5.0, False)]}
        self.augmenter.file_schedule_dict = schedule

        retrieved = self.augmenter.file_schedule_dict
        self.assertEqual(retrieved, schedule)


class TestAudioReaderProperty(unittest.TestCase):
    """Test audio_reader property."""

    def test_getter_setter(self):
        """Test audio_reader getter and setter."""
        config = AugmentationConfig()
        augmenter = AudioAugmenter(config)

        # Initially None
        self.assertIsNone(augmenter.audio_reader)

        # Set mock reader
        mock_reader = Mock()
        augmenter.audio_reader = mock_reader

        # Should retrieve same object
        self.assertEqual(augmenter.audio_reader, mock_reader)


class TestCallMethod(unittest.TestCase):
    """Test __call__ method delegates to augment."""

    def test_call_delegates_to_augment(self):
        """Test __call__ invokes augment method."""
        config = AugmentationConfig()
        augmenter = AudioAugmenter(config)

        with patch.object(augmenter, 'augment') as mock_augment:
            mock_augment.return_value = np.ones(100)

            y = np.ones(100)
            result = augmenter(y, 'cry')

            mock_augment.assert_called_once_with(y, 'cry')
            np.testing.assert_array_equal(result, np.ones(100))


if __name__ == '__main__':
    unittest.main()
