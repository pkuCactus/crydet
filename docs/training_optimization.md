# 训练性能优化方案

## 问题诊断

GPU 利用率 0% 表明 GPU 大部分时间处于等待状态，数据供给跟不上 GPU 计算速度。

## 根因分析

### 1. 当前数据流瓶颈

```
CPU Worker (DataLoader)
  ↓ 加载音频 → 解码 WAV → 返回 numpy
  ↓
collate_fn: 堆叠为 CPU Tensor
  ↓
.to(device) [同步阻塞，等待完成]
  ↓
GPU 特征提取 (STFT + Mel)
  ↓
GPU 模型训练
```

### 2. 具体问题

| 问题 | 位置 | 影响 |
|------|------|------|
| `prefetch_factor` 未设置 | DataLoader | 默认只预取 2 个 batch，无法覆盖 I/O 延迟 |
| `non_blocking=True` 未使用 | train.py:367 | 数据传输阻塞 GPU，无法重叠计算 |
| `persistent_workers=False` | DataLoader | 每个 epoch 重新创建 worker 进程，启动开销大 |
| 特征提取在 GPU 上串行执行 | _extract_features | 每个 batch 同步等待特征提取完成 |

## 优化方案（按优先级排序）

### 方案 1: 使用非阻塞数据传输（关键）

在 `train.py` 的 `_train_epoch` 中，数据传输时使用 `non_blocking=True`：

```python
# 当前代码（阻塞）
features = features.to(self.device)
targets = targets.to(self.device)

# 优化后（非阻塞）
features = features.to(self.device, non_blocking=True)
targets = targets.to(self.device, non_blocking=True)
# 需要确保 pin_memory=True
```

**效果**：允许数据传输和 GPU 计算重叠，减少等待时间。

### 方案 2: 增加 DataLoader 预取

```python
train_loader = DataLoader(
    ...,
    prefetch_factor=4,  # 预取 4 * num_workers 个 batch
    persistent_workers=True,  # epoch 之间保持 worker 进程
)
```

**注意**：`persistent_workers=True` 可能与 DDP 有兼容性问题，需要测试。

### 方案 3: 在 DataLoader 中预计算特征（如果内存允许）

将特征提取移到 `Dataset.__getitem__`，但：
- 缺点：无法利用 GPU 加速（DataLoader worker 是 CPU 进程）
- 优点：可以和训练并行

**更好的方案**：使用单独的 CUDA stream 异步进行特征提取。

### 方案 4: 使用 CUDA Streams 重叠特征提取

```python
# 在 Trainer.__init__ 中创建专门的 stream
self.feature_stream = torch.cuda.Stream(device=device)

# 在 _train_epoch 中使用
with torch.cuda.stream(self.feature_stream):
    features = self._extract_features(waveforms)
    features = features.to(self.device, non_blocking=True)

# 同步 stream
torch.cuda.current_stream().wait_stream(self.feature_stream)
```

### 方案 5: 调整 num_workers

根据 CPU 核心数调整：
- 如果 I/O 是瓶颈：num_workers = CPU 核心数
- 如果 CPU 计算是瓶颈：num_workers = 2-4 倍 GPU 数

## 推荐的配置修改

### train.py 修改（非阻塞传输）

```python
# Line 365-367
features = self._extract_features(waveforms)
features = features.to(self.device, non_blocking=True)  # 添加 non_blocking
targets = targets.to(self.device, non_blocking=True)    # 添加 non_blocking
```

### DataLoader 配置

```python
# 根据系统配置调整
train_loader = DataLoader(
    train_dataset,
    batch_size=config.training.batch_size,
    sampler=train_sampler,
    num_workers=8,  # 增加 worker 数量（根据 CPU 核心调整）
    pin_memory=True,
    prefetch_factor=4,  # 增加预取
    persistent_workers=True,  # 保持 worker 进程
    collate_fn=collate_fn,
    worker_init_fn=worker_init,
    drop_last=True,
)
```

### config 文件修改

```yaml
training:
  num_workers: 8  # 根据 CPU 核心数调整
  pin_memory: true
  # 添加 prefetch_factor 支持
```

## 验证优化效果

使用 `nvidia-smi` 监控 GPU 利用率：

```bash
# 实时监控 GPU 利用率
watch -n 0.5 nvidia-smi

# 或查看详细统计
nvidia-smi dmon -s u
```

优化后 GPU 利用率应保持在 80%+。

## 额外建议

1. **使用 torch.compile**（PyTorch 2.0+）加速模型训练
2. **使用混合精度训练**（已有 `torch.amp`）
3. **检查存储 I/O**：如果数据在 HDD 上，考虑移到 SSD
4. **使用内存映射文件**：对于小数据集，可以预加载到内存
