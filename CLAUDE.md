# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Baby Cry Detection - Audio classification system to detect whether audio contains baby cry sounds.

## Architecture

```
crydet/
├── config.py              # Configuration dataclasses (FeatureConfig, DatasetConfig, AugmentationConfig, ModelConfig, TrainingConfig)
├── configs/
│   └── default.yaml       # Default YAML configuration
├── dataset/
│   ├── audio_reader.py    # Audio loading with resampling and caching (mtime-based cache, librosa resampling)
│   ├── dataset.py         # CryDataset with low-energy cry filtering
│   ├── augmentation.py    # AudioAugmenter with label-aware mixup and sox effects
│   ├── sampler.py         # CrySampler for balanced cry/non-cry sampling
│   ├── utils.py           # Audio utilities (get_db, gain, add_noise, pad_pcm)
│   └── feature.py         # Feature extraction (placeholder)
└── examples/
    └── data_loading.py    # Example: DataLoader creation and sample saving
```

## Data Flow

1. **Audio List JSON** (`audio_list/*.json`): Defines label → directory mappings
   ```json
   {"cry": ["/path/to/cry/audio"], "other": [1, "/path/to/other1", "/path/to/other2"]}
   ```
   - First element for non-cry labels is a duplicate count multiplier

2. **CryDataset** → builds `file_schedule_dict` → filtered by energy threshold

3. **CrySampler** → yields `(label, file_idx)` tuples based on `cry_rate`

4. **AudioAugmenter** → applies mixup (label-aware) + sox effects

## Key Configuration

| Config | Purpose |
|--------|---------|
| `FeatureConfig` | fbank/mfcc extraction, delta features |
| `DatasetConfig` | sample_rate (16kHz), slice_len (5s), cry_rate (0.5) |
| `AugmentationConfig` | mixup params, effect probabilities |
| `cry_min_energy_db` | Dataset param, filters low-energy cry samples (default: -40 dB) |

## Mixup Rules

- **Cry samples**: Mixup sample energy must be 3-10 dB lower than original
- **Non-cry samples**: Mixup can only use non-cry labels

## Common Commands

```bash
# Run data loading example
cd examples && python data_loading.py

# Install dependencies
pip install -r requirements.txt

# Note: sox binary required for audio augmentation effects
```

## Performance Optimizations

- Audio cache uses file mtime (not MD5 hash) for cache validation
- File info cache uses pickle (not JSON) for faster serialization
- Librosa resampling (faster than scipy.signal.resample)
