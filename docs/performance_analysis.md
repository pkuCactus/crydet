# 深度性能优化分析报告

## 当前架构瓶颈分析

### 1. 数据加载流水线（最严重瓶颈）

```
当前流程（CPU 密集型）:
DataLoader Worker
  ↓ 读取 WAV 文件 (I/O)
  ↓ sox 音频增强（外部进程！阻塞！）
  ↓ mixup 加载（额外的磁盘读取）
  ↓ 返回 numpy array
  ↓ collate_fn (CPU 内存拷贝)
  ↓ .to(device, non_blocking=True)
  ↓ GPU 特征提取 (STFT/Mel)
  ↓ GPU 模型训练
```

**问题识别：**

| 位置 | 问题 | 影响 |
|------|------|------|
| `augmentation.py:318-321` | sox 是外部进程调用，GIL 阻塞 | 每个样本增加 50-200ms |
| `augmentation.py:410` | mixup 时额外磁盘读取 | 随机 I/O 延迟 |
| `dataset.py:88-97` | 音频加载在 `__getitem__` | 无法预取 |
| `feature.py:366-447` | 特征提取在训练循环 | GPU 等待 CPU |

### 2. 内存拷贝和格式转换

**当前问题：**
- `collate_fn` 使用 numpy 堆叠，再转 Tensor
- `feature.py` 使用 `register_buffer` 但 mel 矩阵在每次前向都传到 device
- 数据类型 float64 可能在某些地方隐式转换

### 3. 模型层面优化空间

- 没有使用 `torch.compile()` (PyTorch 2.0+)
- Attention 计算使用 `.contiguous()` 导致内存重排
- LayerNorm 在前向中重复计算

---

## 优化方案（按 ROI 排序）

### 方案 1: 预计算特征并缓存（最高 ROI）

**思路：** 特征提取是确定性的，可以提前计算好缓存到磁盘。

```python
# 新增：dataset/precomputed_dataset.py
class PrecomputedFeatureDataset(Dataset):
    """
    预计算特征的数据集
    - 第一次运行时提取特征并保存为 .npy
    - 后续直接从磁盘加载特征
    - 数据增强仍可在特征域进行 (SpecAugment)
    """
    def __init__(self, data_dict, config, cache_dir='./feature_cache'):
        # 检查缓存是否存在
        # 不存在：提取特征并保存
        # 存在：直接加载
```

**预期收益：**
- 训练速度提升 5-10x
- GPU 利用率从 0% 提升到 90%+
- 首次 epoch 有额外开销，但后续 epoch 极快

**存储估算：**
- 每个样本：157 frames × 64 dims × 4 bytes ≈ 40 KB
- 10万样本：约 4 GB（可接受）

---

### 方案 2: 异步音频增强（高 ROI）

**当前问题：** sox 是同步阻塞调用。

**优化方案：**
1. 使用 `torchaudio` 替代 sox（纯 PyTorch，GPU 可加速）
2. 或使用多进程预取增强后的样本

```python
# 改进：augmentation.py
# 使用 torchaudio 替代 sox
def _apply_effect_group_torchaudio(self, y: torch.Tensor, effects: List[str]) -> torch.Tensor:
    """使用 torchaudio 进行 GPU 加速的音频增强"""
    y_tensor = torch.from_numpy(y).to(self.device)
    for effect in effects:
        if effect == 'pitch':
            y_tensor = torchaudio.functional.pitch_shift(y_tensor, ...)
    return y_tensor.cpu().numpy()
```

---

### 方案 3: 使用 torch.compile()（中等 ROI，PyTorch 2.0+）

```python
# train.py
model = create_model(...)
if hasattr(torch, 'compile'):
    model = torch.compile(model, mode='max-autotune')
```

**预期收益：** 20-50% 速度提升（取决于模型大小）

---

### 方案 4: 优化 Attention 内存访问模式

**当前问题：** `layers.py:137`
```python
output = output.transpose(1, 2).contiguous().view(...)
```

**优化：** 使用 `einops` 或优化 reshape 顺序避免 `.contiguous()`

---

### 方案 5: 梯度累积 + 更大有效批次

```python
# utils/config.py 新增
@dataclass
class TrainingConfig:
    gradient_accumulation_steps: int = 1  # 新增

# train.py
for i, batch in enumerate(train_loader):
    loss = model(batch) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**收益：** 更大 batch size 效果，不增加内存

---

### 方案 6: 混合精度优化（已有但可改进）

检查 `torch.amp` 是否在所有地方启用。

---

### 方案 7: 验证流程优化

**当前问题：** `_validate()` 每次都要应用/恢复 EMA

**优化：** 如果不需要中间验证，可以只在最后验证一次

---

### 方案 8: 使用 FastDataLoader（自定义）

```python
# 优化后的 DataLoader，使用共享内存
class FastDataLoader:
    """
    使用 pinned memory + prefetch + async transfer
    """
    def __init__(self, dataset, batch_size, num_workers):
        self.prefetch_queue = Queue(maxsize=prefetch_factor * num_workers)
        # 后台线程预取到 GPU
```

---

## 推荐的实施顺序

### Phase 1: 立即可做（低风险，高收益）

1. **预计算特征缓存** - 收益最大
2. **torch.compile()** - 一行代码，20% 提升
3. **检查 cuDNN benchmark** - 确保已启用

### Phase 2: 短期（中等风险，高收益）

4. **torchaudio 替代 sox** - 移除外部进程开销
5. **梯度累积** - 更好的 batch norm 统计
6. **优化 collate_fn** - 减少内存拷贝

### Phase 3: 中期（需要测试）

7. **CUDA Graphs** - 消除 kernel launch 开销
8. **分布式数据并行优化** - 重叠 all-reduce

---

## 具体实施：特征预计算方案

### 实现代码

```python
# dataset/precomputed_dataset.py
import os
import pickle
import hashlib
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset


class PrecomputedCryDataset(Dataset):
    """
    预计算特征的数据集实现

    缓存结构：
    feature_cache/
    ├── config_hash.pkl      # 配置哈希，用于验证缓存有效性
    ├── cry/
    │   ├── file1_0.npy     # 格式: {filename}_{slice_idx}.npy
    │   └── file2_0.npy
    └── other/
        ├── file3_0.npy
        └── file4_0.npy
    """

    def __init__(self, base_dataset, feature_extractor, config,
                 cache_dir='./feature_cache', force_recompute=False):
        self.base_dataset = base_dataset
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 生成配置哈希（特征配置改变时重新计算）
        config_dict = {
            'feature': config.feature.__dict__,
            'sample_rate': config.dataset.sample_rate,
            'slice_len': config.dataset.slice_len,
        }
        self.config_hash = hashlib.md5(
            pickle.dumps(config_dict)
        ).hexdigest()[:12]

        # 检查缓存有效性
        hash_file = self.cache_dir / 'config_hash.pkl'
        if not force_recompute and hash_file.exists():
            with open(hash_file, 'rb') as f:
                cached_hash = pickle.load(f)
            if cached_hash != self.config_hash:
                print(f"Feature config changed, recomputing cache...")
                force_recompute = True

        # 预计算特征
        if force_recompute or not hash_file.exists():
            self._precompute_features(feature_extractor)
            with open(hash_file, 'wb') as f:
                pickle.dump(self.config_hash, f)

        # 加载缓存索引
        self._load_cache_index()

    def _precompute_features(self, feature_extractor):
        """预计算所有特征"""
        feature_extractor.eval()
        device = next(feature_extractor.parameters()).device

        print("Precomputing features...")
        for idx in tqdm(range(len(self.base_dataset))):
            waveform, label = self.base_dataset[idx]

            # 提取特征
            with torch.no_grad():
                wave_tensor = torch.from_numpy(waveform).unsqueeze(0).to(device)
                features = feature_extractor(wave_tensor).cpu().numpy()[0]

            # 保存
            label_dir = self.cache_dir / label
            label_dir.mkdir(exist_ok=True)
            cache_path = label_dir / f"sample_{idx}.npy"
            np.save(cache_path, features)

        print(f"Features cached to {self.cache_dir}")

    def __getitem__(self, idx):
        # 直接加载预计算的特征
        cache_path, label = self.cache_index[idx]
        features = np.load(cache_path)

        # 可选：在特征域进行数据增强 (SpecAugment)
        if self.training and self.augmenter:
            features = self.augmenter.augment_features(features)

        return features, label

    def __len__(self):
        return len(self.cache_index)
```

### 使用方式

```python
# train.py 修改
# 1. 创建基础数据集
train_dataset = CryDataset(train_dict, config.dataset)

# 2. 包装为预计算数据集（首次会慢，后续极快）
from dataset.precomputed_dataset import PrecomputedCryDataset
train_dataset = PrecomputedCryDataset(
    train_dataset,
    feature_extractor,
    config,
    cache_dir='./feature_cache'
)

# 3. DataLoader 配置调整（不需要 collate_fn 做复杂操作）
train_loader = DataLoader(
    train_dataset,
    batch_size=config.training.batch_size,
    num_workers=0,  # 预计算后可以用 0，因为加载很快
    pin_memory=True,
)
```

---

## 性能对比预估

| 优化项 | 当前时间/批次 | 优化后 | 提升 |
|--------|--------------|--------|------|
| 数据加载 | 500ms | 50ms | 10x |
| 特征提取 | 200ms | 0ms (预计算) | ∞ |
| 模型前向 | 50ms | 35ms (compile) | 1.4x |
| **总计** | **750ms** | **85ms** | **8.8x** |

---

## 内存优化建议

1. **使用 bfloat16**（Ampere GPU+）：比 float16 更稳定
2. **梯度检查点**：用时间换内存，可以增大 batch size
3. **清空缓存**：`torch.cuda.empty_cache()` 在 epoch 之间

---

## 监控建议

添加性能监控：

```python
# train.py
from torch.profiler import profile, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),
    record_shapes=True,
    profile_memory=True,
) as prof:
    for batch in train_loader:
        # training step
        prof.step()
```

使用 TensorBoard 查看瓶颈所在。
