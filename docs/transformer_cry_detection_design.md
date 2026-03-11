# Transformer-based Baby Cry Detection - 算法方案设计文档

## 1. 项目概述

### 1.1 问题定义
婴儿啼哭检测是一个二分类音频分类任务：
- **输入**: 5秒音频片段 (16kHz采样率)
- **输出**: 是否为婴儿啼哭 (cry / non-cry)
- **应用场景**: 智能婴儿监护、实时啼哭报警、啼哭记录分析

### 1.2 设计目标
1. **多层级部署**: 支持从高算力服务器到低功耗边缘设备
2. **实时性能**: 边缘端推理延迟 < 100ms
3. **高准确率**: F1-score > 0.95
4. **小模型尺寸**: 微型版模型 < 1MB
5. **灵活配置**: 通过层参数控制模型大小，而非固定variant

---

## 2. 特征提取方案

### 2.1 特征选择
采用 **Log-Mel Filterbank (FBank)** 作为输入特征：

| 参数 | 值 | 说明 |
|------|-----|------|
| 采样率 | 16kHz | 覆盖婴儿啼哭主要频段 |
| FFT点数 | 1024 | 频率分辨率 ~15.6Hz |
| 帧移 | 512 (32ms) | 时间分辨率 |
| Mel滤波器组 | 64 | 平衡表达能力与计算量 |
| 频率范围 | 250-8000Hz | 聚焦啼哭有效频段 |
| 预加重 | 0.95 | 补偿高频衰减 |

### 2.2 特征维度
- 5秒音频 → 16000×5 = 80000 采样点
- 帧数: (80000 + 512) / 512 ≈ 157 帧
- 输出形状: `[batch, 64, 157]` (channels-first for Transformer)

### 2.3 归一化策略
- **频带归一化**: 每帧独立归一化到 [0, 1]
- **指数平滑**: 对最大值进行时间维度平滑 (decay=0.9)

---

## 3. 模型架构设计

### 3.1 架构概览

```
Input: [B, 64, 157] (FBank features)
    ↓
┌─────────────────────────────────────┐
│  Patch Embedding                      │
│  - Conv1d: 64 → d_model               │
│  - Positional Encoding                │
└─────────────────────────────────────┘
    ↓ [B, d_model, num_patches]
┌─────────────────────────────────────┐
│  Transformer Encoder × N layers       │
│  - Multi-Head Self-Attention          │
│  - Feed-Forward Network               │
│  - LayerNorm + Residual               │
└─────────────────────────────────────┘
    ↓ [B, d_model]
┌─────────────────────────────────────┐
│  Classification Head                  │
│  - Global Average Pooling             │
│  - Linear: d_model → 2                │
└─────────────────────────────────────┘
    ↓
Output: [B, 2] (cry / non-cry logits)
```

### 3.2 模型大小配置

模型大小通过**层参数**灵活控制，而非固定variant：

| 参数 | 说明 | 推荐范围 |
|------|------|---------|
| `d_model` | 隐藏层维度 | 64-512 |
| `n_layers` | Transformer层数 | 2-12 |
| `n_heads` | 注意力头数 | 2-8 |
| `d_ff` | FFN维度 | 128-2048 |

#### 3.2.1 预设配置参考

| 配置 | d_model | n_layers | n_heads | 参数量 | 适用场景 |
|------|---------|----------|---------|--------|---------|
| Large | 512 | 12 | 8 | ~40M | 服务器/云端 |
| Medium | 256 | 6 | 4 | ~5M | 边缘设备(RPi/EdgeTPU) |
| Small | 192 | 4 | 4 | ~2M | 嵌入式设备 |
| Tiny | 128 | 3 | 2 | ~600K | MCU(Cortex-M/ESP32) |
| Nano | 64 | 2 | 2 | ~120K | 超低功耗MCU |

### 3.3 自动架构优化

当 `attention_type='auto'` 和 `ffn_type='auto'` 时，系统根据模型大小自动选择最优实现：

```python
# 自动选择逻辑
d_model * n_layers >= 3000  →  standard attention + standard FFN
                         (大模型，追求精度)
d_model * n_layers < 3000  →  depthwise attention + inverted bottleneck
                         (小模型，追求效率)
```

### 3.4 关键设计决策

#### 3.4.1 Patch Embedding
替代传统的帧级输入，使用卷积进行Patch嵌入：
```python
# 将时间维度进行Patch分割
Conv1d(in_channels=64, out_channels=d_model, kernel_size=3, stride=2, padding=1)
# 输出: [B, d_model, 79] (157//2 ≈ 79 patches)
```

#### 3.4.2 相对位置编码
使用可学习的相对位置编码替代绝对位置编码：
- 更好地处理变长输入
- 更少的参数量
- 更适合音频这种时序数据

#### 3.4.3 注意力机制变体

**标准Multi-Head Self-Attention** (大模型)
```
Attention(Q,K,V) = softmax(QK^T/√d_k)V
```

**Linear Attention** (中等模型，可选)
```
Attention(Q,K,V) ≈ φ(Q)(φ(K)^TV) / (φ(Q)φ(K)^T·1)
# 将O(n²)复杂度降为O(n)
```

**Depthwise-Separable Attention** (小模型)
```
# 使用深度可分离卷积减少参数量
DepthwiseConv1d + PointwiseConv
```

#### 3.4.4 FFN优化

**标准FFN**: d_model → 4×d_model → d_model

**Inverted Bottleneck FFN** (小模型)
```python
# 1x1 expand -> depthwise 1D conv -> 1x1 project
# 减少计算量同时保持表达能力
```

---

## 4. 训练策略

### 4.1 分布式训练 (DDP)

支持多GPU分布式训练，自动处理梯度同步：

```bash
# 单卡训练
python train.py --config configs/model_medium.yaml --train_list train.json

# 多卡训练 (4 GPU)
torchrun --nproc_per_node=4 train.py --config configs/model_medium.yaml --train_list train.json

# 或使用启动脚本
./train_ddp.sh --ngpu 4 --config configs/model_medium.yaml --train_list train.json
```

**DDP特性**:
- 自动检测GPU数量
- 分布式Sampler保持cry/non-cry比例
- 梯度自动all-reduce
- 仅rank 0保存checkpoint

### 4.2 损失函数
```python
# 标签平滑交叉熵 + Focal Loss (处理类别不平衡)
loss = α * CE(p, y_smooth) + β * FocalLoss(p, y)
```

### 4.3 优化器配置
```python
optimizer = AdamW(
    lr=1e-3,
    weight_decay=0.05,
    betas=(0.9, 0.98)
)
scheduler = CosineAnnealingWarmRestarts(
    T_0=10, T_mult=2
)
```

### 4.4 数据增强
- **Mixup**: 标签感知的音频混合 (cry样本与非cry样本不同策略)
- **SpecAugment**: 时域和频域掩码
- **音频效果**: pitch shift, reverb, noise injection

### 4.5 训练技巧
1. **Warmup**: 前5个epoch线性增加学习率
2. **梯度裁剪**: max_norm=1.0
3. **早停**: patience=10
4. **模型平均**: SWA (Stochastic Weight Averaging)
5. **混合精度**: AMP自动混合精度训练

---

## 5. 部署方案

### 5.1 导出格式支持

| 格式 | 高性能版 | 轻量版 | 微型版 |
|------|---------|--------|--------|
| PyTorch (.pt) | ✓ | ✓ | ✓ |
| ONNX (.onnx) | ✓ | ✓ | ✓ |
| TorchScript (.ptl) | ✓ | ✓ | ✓ |
| Core ML (.mlmodel) | - | ✓ | ✓ |
| TensorFlow Lite (.tflite) | - | ✓ | ✓ |
| C模型 (纯C代码) | - | - | ✓ |

**导出命令**:
```bash
python export.py --checkpoint checkpoints/best_model.pt --format onnx --quantize int8
```

### 5.2 芯片部署矩阵

| 芯片平台 | 推荐配置 | 推理框架 | 预期延迟 |
|---------|---------|---------|---------|
| NVIDIA GPU | d_model=512, n_layers=12 | PyTorch/TensorRT | <5ms |
| Intel CPU | d_model=256, n_layers=6 | ONNX Runtime | 10-50ms |
| ARM Cortex-A72 | d_model=256, n_layers=6 | ONNX Runtime/NCNN | 20-50ms |
| ARM Cortex-M4 | d_model=128, n_layers=3 | CMSIS-NN | 50-100ms |
| ESP32-S3 | d_model=64, n_layers=2 | ESP-NN | 80-150ms |

### 5.3 推理优化

**通用优化**:
- 算子融合 (Conv+BN+ReLU)
- 权重量化 (INT8)
- 动态量化 (运行时INT8)

**移动端优化**:
- Neural Architecture Search (NAS) 压缩
- 知识蒸馏 (Large → Tiny)
- 结构化剪枝

---

## 6. 评估指标

### 6.1 主要指标
- **Accuracy**: 整体准确率
- **Precision/Recall**: 精确率与召回率
- **F1-Score**: 综合性能指标
- **AUC-ROC**: 区分能力

### 6.2 效率指标
- **参数量 (Params)**: 模型大小
- **计算量 (FLOPs/MACs)**: 乘加操作数
- **推理延迟**: 单次推理时间
- **内存占用**: 运行时内存

### 6.3 目标性能

| 指标 | 高性能版 | 轻量版 | 微型版 |
|------|---------|--------|--------|
| F1-Score | ≥0.97 | ≥0.95 | ≥0.92 |
| 参数量 | ~40M | ~5M | ~600K |
| 模型大小 (INT8) | ~40MB | ~5MB | ~600KB |
| 推理延迟* | 5ms | 20ms | 100ms |

*注: 延迟为ARM Cortex-A72 @ 1.5GHz参考值

---

## 7. 项目结构

```
crydet/
├── model/
│   ├── __init__.py
│   ├── transformer.py      # 核心Transformer实现
│   ├── layers.py           # 自定义层 (注意力、FFN等)
│   └── variants.py         # 模型创建工厂
├── train.py                # 训练脚本 (支持DDP)
├── train_ddp.sh            # 多卡训练启动脚本
├── evaluate.py             # 评估脚本
├── export.py               # 模型导出工具
├── inference.py            # 推理脚本
├── config.py               # 配置类 (ModelConfig等)
└── configs/
    ├── model_large.yaml    # 高性能版配置
    ├── model_medium.yaml   # 轻量版配置
    └── model_tiny.yaml     # 微型版配置
```

---

## 8. 使用指南

### 8.1 模型创建

```python
from config import ModelConfig
from model import create_model

# 方法1: 自定义层参数 (推荐)
config = ModelConfig(d_model=256, n_layers=6, n_heads=4, d_ff=1024)
model = create_model(config=config, in_channels=64, num_classes=2)

# 方法2: 使用预设配置
from model import MODEL_CONFIGS
config = ModelConfig(**MODEL_CONFIGS['tiny'])
model = create_model(config=config)
```

### 8.2 训练

```bash
# 单卡训练
python train.py \
    --config configs/model_medium.yaml \
    --train_list audio_list/train.json \
    --val_list audio_list/val.json \
    --batch_size 32

# 多卡训练
./train_ddp.sh \
    --ngpu 4 \
    --config configs/model_medium.yaml \
    --train_list audio_list/train.json \
    --val_list audio_list/val.json \
    --batch_size 32  # per GPU

# 命令行覆盖层参数
python train.py \
    --config configs/default.yaml \
    --d_model 192 \
    --n_layers 4 \
    --train_list audio_list/train.json
```

### 8.3 评估

```bash
python evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --test_list audio_list/test.json \
    --benchmark
```

### 8.4 推理

```bash
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --audio path/to/audio.wav \
    --detect_regions
```

### 8.5 导出

```bash
# 导出ONNX
python export.py --checkpoint checkpoints/best_model.pt --format onnx

# 导出量化模型
python export.py --checkpoint checkpoints/best_model.pt --format onnx --quantize int8

# 导出所有格式
python export.py --checkpoint checkpoints/best_model.pt --format all
```

---

## 9. 实现状态

- [x] **Phase 1**: 基础Transformer实现 (支持灵活层参数配置)
- [x] **Phase 2**: 训练 pipeline 实现 (支持DDP多卡训练)
- [x] **Phase 3**: 评估与测试代码
- [x] **Phase 4**: 模型导出与优化 (ONNX/量化)
- [ ] **Phase 5**: 边缘部署支持 (CMSIS-NN/TFLite) - 待测试
