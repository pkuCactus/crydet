# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Baby Cry Detection - Audio classification system to detect whether audio contains baby cry sounds using Transformer-based architecture.

## Architecture

```
crydet/
├── config.py              # Configuration dataclasses (FeatureConfig, DatasetConfig, AugmentationConfig, ModelConfig, TrainingConfig)
├── configs/
│   ├── default.yaml       # Default YAML configuration
│   ├── model_large.yaml   # High-performance config (d_model=512, n_layers=12)
│   ├── model_medium.yaml  # Edge device config (d_model=256, n_layers=6)
│   └── model_tiny.yaml    # MCU config (d_model=128, n_layers=3)
├── model/
│   ├── transformer.py     # CryTransformer with Linear projection (NOT Conv1d)
│   ├── layers.py          # Attention variants (standard/linear/depthwise), FFN variants
│   ├── loss.py            # FocalLoss, LabelSmoothingCrossEntropy, CombinedLoss
│   └── variants.py        # Model factory functions, MODEL_CONFIGS presets
├── dataset/
│   ├── audio_reader.py    # Audio loading with resampling and caching
│   ├── dataset.py         # CryDataset with low-energy cry filtering
│   ├── augmentation.py    # AudioAugmenter with label-aware mixup
│   ├── feature.py         # FeatureExtractor with FBank + delta features
│   ├── sampler.py         # CrySampler and DistributedCrySampler
│   └── utils.py           # Audio utilities
├── train.py               # Training script with DDP support
├── train_ddp.sh           # Multi-GPU training launcher
├── evaluate.py            # Evaluation script
├── export.py              # Model export (ONNX, TorchScript, etc.)
└── examples/
    ├── model_demo.py      # Model creation and benchmarking demo
    └── data_loading.py    # DataLoader usage example
```

## Model Architecture

**CryTransformer** uses Linear projection (NOT Conv1d patch embedding):

```
Input: [B, T, F] where B=batch, T=time_frames(157), F=feature_dim
    ↓
Linear Projection: F → d_model (F can be 64/128/192 with delta features)
    ↓
Positional Encoding: Sinusoidal or relative
    ↓
Transformer Encoder × N layers
    ↓
Global Average Pooling
    ↓
Classifier: d_model → num_classes
```

**Key Design Points:**
- Input format: `[B, T, F]` (batch, time_frames, feature_dim)
- Time dimension T=157 for 5s audio @ 16kHz with hop_length=512
- Feature dim F: 64 (base), 128 (+time delta), 192 (+freq delta)
- Linear projection replaces Conv1d patch embedding
- Sequence length preserved (no downsampling)

## Data Flow

### Data Loading Pipeline

```
Audio List JSON
    ↓ (labels → directories)
CryDataset (Dataset.__init__)
    ↓ (scan files, build schedule)
file_schedule_dict
    ↓ (filtered by energy threshold)
CrySampler
    ↓ (yields balanced cry/non-cry indices)
Dataset.__getitem__(index)
    ↓ (load audio → augment → extract features)
Features [T, F] + Label
    ↓
collate_fn (stack to [B, T, F])
    ↓
DataLoader → Model
```

### Step-by-Step

1. **Audio List JSON** (`audio_list/*.json`): Defines label → directory mappings
   ```json
   {"cry": ["/path/to/cry/audio"], "other": [1, "/path/to/other1", "/path/to/other2"]}
   ```
   - First element for non-cry labels is a duplicate count multiplier

2. **CryDataset initialization** (`dataset/dataset.py`):
   - Scans directories → builds `file_schedule_dict[label]`
   - Each entry: `(file_path, start_time, duration, need_pad)`
   - Filters low-energy cry samples (< cry_min_energy_db)

3. **CrySampler** (`dataset/sampler.py`):
   - Balances cry/non-cry sampling based on `cry_rate`
   - Yields `(label, file_idx)` tuples
   - Automatically cycles through samples

   **DistributedCrySampler**: Distributed version for DDP training
   - Wraps CrySampler for multi-GPU distributed training
   - Handles data partitioning across replicas

4. **Dataset.__getitem__**:
   - Loads audio segment from file
   - Applies augmentation (if training)
   - Extracts features via `FeatureExtractor.extract_with_deltas()`
   - Returns `(features, label)` where features shape is `[T, F]`

5. **collate_fn**:
   - Stacks features to `[B, T, F]` batch tensor
   - Converts labels to indices (cry=1, other=0)

**Note:** Feature extraction happens in the Dataset (not Trainer), enabling:
- Feature caching capability
- Multi-processing with `num_workers`
- Cleaner separation of concerns

### Key Data Structures

| Component | Key Structure | Description |
|-----------|---------------|-------------|
| `file_schedule_dict` | `{label: [(path, start, dur, pad), ...]}` | All audio segments per label |
| `CrySampler` | yields `(label, idx)` | Balanced sampling indices |
| `Dataset.__getitem__` | returns `(features, label)` | `features` shape: `[T, F]` |
| `collate_fn` | returns `(batch_features, batch_labels)` | `batch_features` shape: `[B, T, F]` |

## Feature Extraction

**FeatureConfig parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_mels` | 64 | Number of mel filters |
| `n_fft` | 1024 | FFT window size |
| `hop_length` | 512 | Frame shift (32ms @ 16kHz) |
| `fmin` | 250 | Min frequency for mel filter |
| `fmax` | 8000 | Max frequency for mel filter |
| `use_delta` | false | Add time delta features (+64 dim) |
| `use_freq_delta` | false | Add frequency delta features (+64 dim) |

**Output dimensions:**
- Base: `[T, 64]`
- +delta: `[T, 128]`
- +freq_delta: `[T, 192]`

## Model Configuration

**Model size controlled by layer parameters:**

| Config | d_model | n_layers | n_heads | d_ff | Params | Use Case |
|--------|---------|----------|---------|------|--------|----------|
| Large | 512 | 12 | 8 | 2048 | ~40M | Server/Cloud |
| Medium | 256 | 6 | 4 | 1024 | ~5M | Edge device |
| Tiny | 128 | 3 | 2 | 256 | ~600K | MCU |
| Nano | 64 | 2 | 2 | 128 | ~120K | Ultra-low power |

**Auto-selection (when type='auto'):**
- Large models (d_model*n_layers >= 3000): standard attention + standard FFN
- Small models: depthwise attention + inverted bottleneck FFN

## Mixup Rules

- **Cry samples**: Mixup sample energy must be 3-10 dB lower than original
- **Non-cry samples**: Mixup can only use non-cry labels

## Training

```bash
# Single GPU
python train.py --config configs/model_medium.yaml --train_list audio_list/train.json

# Multi-GPU DDP
./train_ddp.sh --ngpu 4 --config configs/model_medium.yaml --train_list audio_list/train.json
```

**Key Training Features:**
- DDP (DistributedDataParallel) multi-GPU support
- Automatic mixed precision (AMP)
- SpecAugment for spectrogram augmentation
- Label smoothing and Focal Loss
- SWA (Stochastic Weight Averaging)
- Early stopping with patience

## Common Commands

```bash
# Run model demo
python examples/model_demo.py --mode all

# Run data loading example
cd examples && python data_loading.py

# Export model
python export.py --checkpoint checkpoints/best_model.pt --format onnx --quantize int8

# Install dependencies
pip install -r requirements.txt

# Note: sox binary required for audio augmentation effects
```

## Performance Optimizations

- Audio cache uses file mtime (not MD5 hash) for cache validation
- File info cache uses pickle (not JSON) for faster serialization
- Librosa resampling (faster than scipy.signal.resample)
- DDP with DistributedCrySampler for multi-GPU training
