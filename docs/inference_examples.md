# Inference 使用指南

本文档详细介绍如何使用 CryDet 的推理功能进行婴儿啼哭检测。

## 目录

- [命令行使用](#命令行使用)
- [Python API 使用](#python-api-使用)
- [高级用法](#高级用法)
- [性能优化](#性能优化)

## 命令行使用

### 基本推理

检测单个音频文件：

```bash
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --audio path/to/audio.wav
```

输出示例：
```
2024-01-15 10:30:45,123 - INFO - Loading model from: checkpoints/best_model.pt
2024-01-15 10:30:45,456 - INFO - Model loaded. Device: cuda

Prediction:
  Is Cry: True
  Cry Probability: 0.9234
  Confidence: 0.8468
```

### 滑动窗口推理

对于长音频，使用滑动窗口进行连续检测：

```bash
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --audio path/to/long_audio.wav \
    --sliding_window \
    --stride 1.0
```

输出示例：
```
Sliding window predictions (15 windows):
  0.00s - 5.00s: Cry=True, Prob=0.923
  1.00s - 6.00s: Cry=True, Prob=0.891
  2.00s - 7.00s: Cry=True, Prob=0.756
  ...
```

### 啼哭区域检测

自动检测并合并啼哭区域：

```bash
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --audio path/to/audio.wav \
    --detect_regions
```

输出示例：
```
Detected 2 cry region(s):
  1. 3.50s - 8.20s (duration: 4.70s, confidence: 0.891)
  2. 15.30s - 18.50s (duration: 3.20s, confidence: 0.756)
```

### 批量推理

处理多个音频文件：

```bash
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --audio_list audio_list/test.json \
    --output results.json
```

### 性能测试

测试模型推理性能：

```bash
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --benchmark
```

输出示例：
```
Benchmark Results:
  Mean: 12.34 ms
  P95:  15.67 ms
  RTF:  0.0025 (real-time factor)
```

## Python API 使用

### 初始化检测器

```python
from inference import CryDetector

# 基础初始化
detector = CryDetector('checkpoints/best_model.pt')

# 完整参数初始化
detector = CryDetector(
    checkpoint_path='checkpoints/best_model.pt',
    device='cuda',           # 或 'cpu'
    sample_rate=16000,       # 采样率
    slice_len=5.0,          # 分析窗口长度（秒）
    stride=1.0,             # 滑动窗口步长（秒）
    threshold=0.5           # 检测阈值
)
```

### 单文件预测

```python
# 短音频直接预测
result = detector.predict_file('baby_cry.wav', sliding_window=False)

print(f"是否为啼哭: {result['is_cry']}")
print(f"啼哭概率: {result['cry_prob']:.4f}")
print(f"非啼哭概率: {result['non_cry_prob']:.4f}")
print(f"置信度: {result['confidence']:.4f}")
```

### 滑动窗口预测

```python
# 长音频使用滑动窗口
results = detector.predict_file('long_recording.wav', sliding_window=True)

for r in results:
    print(f"{r['start_time']:.1f}s - {r['end_time']:.1f}s: "
          f"Cry={r['is_cry']}, Prob={r['cry_prob']:.3f}")
```

### 啼哭区域检测

```python
# 检测啼哭区域，自动合并相邻区域
regions = detector.detect_cry_regions(
    'recording.wav',
    min_duration=1.0,    # 最小区间长度（秒）
    merge_gap=0.5        # 合并间隔小于0.5秒的区间
)

for region in regions:
    print(f"检测到啼哭: {region['start']:.2f}s - {region['end']:.2f}s")
    print(f"  持续时间: {region['duration']:.2f}s")
    print(f"  平均置信度: {region['confidence']:.3f}")
    print(f"  最大置信度: {region['max_confidence']:.3f}")
```

### 直接预测波形

```python
import numpy as np

# 准备音频波形（5秒 @ 16kHz）
waveform = np.random.randn(80000).astype(np.float32)

# 预测
result = detector.predict(waveform)

print(f"结果: {result}")
```

### 批量处理

```python
import json
from pathlib import Path

# 加载音频列表
with open('audio_list/test.json', 'r') as f:
    data_dict = json.load(f)

results = []
for label, paths in data_dict.items():
    # 跳过非啼哭的重复计数
    if label != 'cry' and len(paths) > 0 and isinstance(paths[0], int):
        paths = paths[1:]

    for path in paths:
        result = detector.predict_file(path, sliding_window=False)
        results.append({
            'file': path,
            'label': label,
            'predicted': 'cry' if result['is_cry'] else 'other',
            'probability': result['cry_prob']
        })

# 计算准确率
correct = sum(1 for r in results if r['label'] == r['predicted'])
accuracy = correct / len(results)
print(f"准确率: {accuracy:.4f}")
```

## 高级用法

### 实时音频流处理

```python
import sounddevice as sd
import numpy as np
from inference import CryDetector

detector = CryDetector('checkpoints/best_model.pt')

# 音频流回调
buffer = []
sample_rate = 16000
chunk_size = 1600  # 100ms

 def audio_callback(indata, frames, time_info, status):
    buffer.extend(indata[:, 0])

    # 当缓冲区达到5秒时进行预测
    if len(buffer) >= sample_rate * 5:
        waveform = np.array(buffer[:sample_rate * 5], dtype=np.float32)
        result = detector.predict(waveform)

        if result['is_cry']:
            print(f"检测到啼哭! 概率: {result['cry_prob']:.3f}")

        # 滑动缓冲区（保留1秒重叠）
        buffer[:] = buffer[int(sample_rate * 4):]

# 启动音频流
with sd.InputStream(callback=audio_callback, channels=1, samplerate=sample_rate):
    print("开始监听...")
    sd.sleep(60000)  # 监听60秒
```

### 多模型集成

```python
from inference import CryDetector

# 加载多个模型
detector_large = CryDetector('checkpoints/large_model.pt', device='cuda')
detector_tiny = CryDetector('checkpoints/tiny_model.pt', device='cpu')

def ensemble_predict(waveform):
    """多模型集成预测"""
    result1 = detector_large.predict(waveform)
    result2 = detector_tiny.predict(waveform)

    # 加权平均
    ensemble_prob = 0.7 * result1['cry_prob'] + 0.3 * result2['cry_prob']

    return {
        'is_cry': ensemble_prob > 0.5,
        'cry_prob': ensemble_prob,
        'confidence': abs(ensemble_prob - 0.5) * 2
    }
```

### 自定义后处理

```python
class SmoothingDetector:
    """带平滑的检测器"""

    def __init__(self, detector, window_size=5):
        self.detector = detector
        self.window_size = window_size
        self.history = []

    def predict(self, waveform):
        result = self.detector.predict(waveform)

        self.history.append(result['cry_prob'])
        if len(self.history) > self.window_size:
            self.history.pop(0)

        # 移动平均平滑
        smoothed_prob = sum(self.history) / len(self.history)

        return {
            'is_cry': smoothed_prob > self.detector.threshold,
            'cry_prob': smoothed_prob,
            'raw_prob': result['cry_prob'],
            'confidence': abs(smoothed_prob - 0.5) * 2
        }

# 使用
smoothing_detector = SmoothingDetector(detector, window_size=5)
```

## 性能优化

### GPU 优化

```python
import torch

# 确保 CUDA 优化启用
torch.backends.cudnn.benchmark = True

# 使用半精度推理（如果支持）
detector = CryDetector('checkpoints/best_model.pt', device='cuda')
detector.model = detector.model.half()

# 使用 torch.jit 优化
example_input = torch.randn(1, 157, 64).cuda()
traced_model = torch.jit.trace(detector.model, example_input)
detector.model = traced_model
```

### 批处理优化

```python
def batch_predict(detector, audio_files, batch_size=8):
    """批量预测多个音频文件"""
    from dataset.audio_reader import AudioReader
    from dataset.feature import FeatureExtractor

    reader = AudioReader(target_sr=16000)
    feature_extractor = FeatureExtractor(detector.config.feature)

    results = []
    for i in range(0, len(audio_files), batch_size):
        batch_files = audio_files[i:i + batch_size]

        # 加载并提取特征
        features_list = []
        for file in batch_files:
            waveform, sr = reader.load(file)
            features = feature_extractor.extract_single(waveform, sr)
            features_list.append(features)

        # 批处理
        import numpy as np
        batch_features = np.stack(features_list)
        batch_tensor = torch.from_numpy(batch_features).float().to(detector.device)

        with torch.no_grad():
            outputs = detector.model(batch_tensor)
            probs = torch.softmax(outputs, dim=1)

        for j, file in enumerate(batch_files):
            cry_prob = probs[j, 1].item()
            results.append({
                'file': file,
                'is_cry': cry_prob > detector.threshold,
                'cry_prob': cry_prob
            })

    return results
```

### 异步推理

```python
import asyncio
import concurrent.futures

async def async_predict(detector, waveform):
    """异步预测"""
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(
            pool, detector.predict, waveform
        )
    return result

# 使用
async def process_multiple(audio_files):
    tasks = [async_predict(detector, load_audio(f)) for f in audio_files]
    results = await asyncio.gather(*tasks)
    return results
```

## 错误处理

```python
from inference import CryDetector
import logging

logging.basicConfig(level=logging.INFO)

try:
    detector = CryDetector('checkpoints/best_model.pt')
except FileNotFoundError:
    print("错误：检查点文件不存在")
except RuntimeError as e:
    if "CUDA" in str(e):
        print("错误：CUDA不可用，切换到CPU模式")
        detector = CryDetector('checkpoints/best_model.pt', device='cpu')

try:
    result = detector.predict_file('audio.wav')
except Exception as e:
    print(f"预测失败: {e}")
```
