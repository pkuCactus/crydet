# CryDet - 婴儿啼哭检测系统

基于 Transformer 的音频分类系统，用于检测音频中是否包含婴儿啼哭声。

## 特性

- **Transformer 架构**: 使用 Linear Projection 替代 Conv1d，更好地保留时序特征
- **多层级部署**: 支持从服务器到 MCU 的多尺度模型配置
- **分布式训练**: 支持 DDP 多卡训练
- **灵活配置**: 通过 YAML 配置文件控制模型和训练参数
- **实时推理**: 支持滑动窗口和区域检测

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 准备数据

创建音频列表 JSON 文件：

```json
{
  "cry": ["/path/to/cry/audio/files"],
  "other": [2, "/path/to/other1", "/path/to/other2"]
}
```

`other` 列表的第一个元素是重复次数乘数（用于数据平衡）。

### 训练

```bash
# 单卡训练
python train.py \
    --config configs/model_medium.yaml \
    --train_list audio_list/train.json \
    --val_list audio_list/val.json

# 多卡训练
./train_ddp.sh --ngpu 4 \
    --config configs/model_medium.yaml \
    --train_list audio_list/train.json
```

### 推理

#### 命令行使用

```bash
# 单文件推理
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --audio path/to/audio.wav

# 滑动窗口推理（长音频）
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --audio path/to/long_audio.wav \
    --sliding_window \
    --stride 1.0

# 检测啼哭区域
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --audio path/to/audio.wav \
    --detect_regions

# 批量推理
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --audio_list audio_list/test.json

# 性能测试
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --benchmark
```

#### Python API 使用

```python
from inference import CryDetector

# 初始化检测器
detector = CryDetector(
    checkpoint_path='checkpoints/best_model.pt',
    device='cuda',           # 或 'cpu'
    sample_rate=16000,       # 采样率
    slice_len=5.0,          # 分析窗口长度（秒）
    stride=1.0,             # 滑动窗口步长（秒）
    threshold=0.5           # 检测阈值
)

# 1. 单文件预测
result = detector.predict_file('path/to/audio.wav', sliding_window=False)
print(f"Is cry: {result['is_cry']}")
print(f"Cry probability: {result['cry_prob']:.4f}")

# 2. 滑动窗口预测（长音频）
results = detector.predict_file('path/to/long_audio.wav', sliding_window=True)
for r in results:
    print(f"{r['start_time']:.1f}s - {r['end_time']:.1f}s: "
          f"Cry={r['is_cry']}, Prob={r['cry_prob']:.3f}")

# 3. 检测啼哭区域
regions = detector.detect_cry_regions(
    'path/to/audio.wav',
    min_duration=1.0,    # 最小区间长度
    merge_gap=0.5        # 合并间隔小于0.5秒的区间
)
for region in regions:
    print(f"Detected cry: {region['start']:.2f}s - {region['end']:.2f}s "
          f"(confidence: {region['confidence']:.3f})")

# 4. 直接预测波形（NumPy数组）
import numpy as np
waveform = np.random.randn(80000).astype(np.float32)  # 5秒 @ 16kHz
result = detector.predict(waveform)

# 5. 性能测试
benchmark = detector.benchmark(num_runs=100)
print(f"Mean latency: {benchmark['mean_ms']:.2f} ms")
print(f"RTF: {benchmark['rtf']:.4f}")
```

#### API 详细说明

**CryDetector 类**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `checkpoint_path` | str | 必填 | 模型检查点路径 |
| `device` | str | 'cuda' | 运行设备 ('cuda' 或 'cpu') |
| `sample_rate` | int | 16000 | 音频采样率 |
| `slice_len` | float | 5.0 | 分析窗口长度（秒） |
| `stride` | float | 1.0 | 滑动窗口步长（秒） |
| `threshold` | float | 0.5 | 检测阈值 |

**方法**

- `predict(waveform: np.ndarray) -> dict`: 对单段音频进行预测
  - 返回: `{'is_cry': bool, 'cry_prob': float, 'non_cry_prob': float, 'confidence': float}`

- `predict_file(audio_path: str, sliding_window: bool = True) -> dict or list`: 对音频文件进行预测
  - `sliding_window=False`: 返回单个预测结果
  - `sliding_window=True`: 返回滑动窗口预测列表

- `detect_cry_regions(audio_path: str, min_duration: float = 1.0, merge_gap: float = 0.5) -> list`: 检测啼哭区域
  - 返回: `[{'start': float, 'end': float, 'duration': float, 'confidence': float, 'max_confidence': float}, ...]`

- `benchmark(num_runs: int = 100, input_duration: float = 5.0) -> dict`: 性能测试
  - 返回: `{'mean_ms': float, 'std_ms': float, 'p95_ms': float, 'p99_ms': float, 'rtf': float}`

### 导出模型

```bash
# 导出 ONNX
python export.py --checkpoint checkpoints/best_model.pt --format onnx

# 导出量化模型
python export.py --checkpoint checkpoints/best_model.pt --format onnx --quantize int8
```

## 模型配置

| 配置 | d_model | n_layers | 参数量 | 适用场景 |
|------|---------|----------|--------|----------|
| Large | 512 | 12 | ~40M | 服务器/云端 |
| Medium | 256 | 6 | ~5M | 边缘设备 |
| Tiny | 128 | 3 | ~600K | MCU |
| Nano | 64 | 2 | ~120K | 超低功耗 |

## 项目结构

```
crydet/
├── config.py              # 配置类定义
├── configs/               # 配置文件
│   ├── default.yaml
│   ├── model_large.yaml
│   ├── model_medium.yaml
│   └── model_tiny.yaml
├── model/                 # 模型实现
│   ├── transformer.py     # CryTransformer
│   ├── layers.py          # 注意力/FFN变体
│   ├── loss.py            # 损失函数
│   └── variants.py        # 模型工厂
├── dataset/               # 数据处理
│   ├── audio_reader.py    # 音频加载
│   ├── dataset.py         # CryDataset
│   ├── augmentation.py    # 数据增强
│   ├── feature.py         # 特征提取
│   └── sampler.py         # 采样器
├── train.py               # 训练脚本
├── inference.py           # 推理脚本
├── evaluate.py            # 评估脚本
└── export.py              # 模型导出
```

## 算法设计

详见 [docs/transformer_cry_detection_design.md](docs/transformer_cry_detection_design.md)

## 许可证

MIT License
