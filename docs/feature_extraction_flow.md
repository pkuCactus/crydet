# 特征提取流程分析

## 1. 系统概述

[dataset/feature.py](../dataset/feature.py) 是音频特征提取模块，基于 NumPy/Librosa 实现，支持从音频波形中提取多种特征，包括 FBank、MFCC 和能量特征等。支持通过配置灵活组合特征类型、添加 delta 特征和能量特征。

## 2. 处理流程总图

### 2.1 特征提取完整架构图

```mermaid
graph TB
    A[输入音频文件] --> B[数据加载模块]
    B --> C[音频预处理]
    C --> D[分帧处理]
    D --> E[特征提取]
    E --> F[数据归一化]
    F --> G[数据增强]
    G --> H[输出特征]

    subgraph "B. 数据加载模块"
        B1[get_pcm函数] --> B2[缓存检查]
        B2 --> B3[音频加载]
        B3 --> B4[重采样处理]
        B4 --> B5[数据补零]
        B5 --> B6[有效性检查]
    end

    subgraph "C. 音频预处理"
        C1[幅值归一化] --> C2[预加重滤波]
        C2 --> C3[边界填充]
    end

    subgraph "D. 分帧处理"
        D1[汉宁窗函数] --> D2[信号分帧]
        D2 --> D3[能量计算]
    end

    subgraph "E. 特征提取"
        E1[Mel滤波器组] --> E2[对数变换]
        E2 --> E3[MFCC计算]
        E2 --> E4[FBank输出]
        E5[能量计算] --> E6[DB特征]
    end

    subgraph "F. 数据归一化"
        F1[FBank归一化] --> F2[能量归一化]
        F2 --> F3[幅值裁剪]
    end

    subgraph "G. Delta特征"
        G1[时间差分计算] --> G2[频率差分计算]
        G2 --> G3[特征拼接]
    end

    subgraph "H. 输出特征"
        H1[FFT特征] --> H2[FBank特征]
        H2 --> H3[MFCC特征]
        H3 --> H4[DB能量特征]
        H4 --> H5[掩码率]
    end

    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style C fill:#e8f5e8
    style D fill:#fff3e0
    style E fill:#fce4ec
    style F fill:#e0f2f1
    style G fill:#f1f8e9
    style H fill:#fff8e1
```

### 2.2 详细处理流程图

```mermaid
flowchart TD
    A["原始音频文件\n.wav/.mp3"] --> B[是否缓存检查]
    B -->|存在缓存| C[加载缓存音频]
    B -->|无缓存| D[使用librosa加载音频]
    D --> E[采样率检查]
    E -->|不正确| F[重采样到16kHz]
    F --> G[保存缓存]
    E -->|正确| H[直接使用]
    C --> I[数据补零处理]
    H --> I
    I --> J[NaN值检查]
    J --> K[信号归一化]
    K --> L[边界填充]
    L --> M["分帧处理\nNFFT=1024, HOP=500"]
    M --> N[预加重滤波]
    N --> O[汉宁窗加窗]
    O --> P[计算FFT]
    P --> Q[计算Mel滤波器矩阵]
    Q --> R[应用Mel滤波器]
    R --> S[对数梅尔谱]
    S --> T[计算MFCC]

    U[能量特征计算] --> V[帧平均能量]
    U --> W[汉宁窗加权能量]

    X[特征归一化] --> Y[FBank归一化]
    X --> Z[能量特征处理]

    AA[数据增强] --> BB[DropBlock掩码]
    BB --> CC[概率应用掩码]

    DD[最终输出] --> EE[FFT特征]
    DD --> FF[FBank特征]
    DD --> GG[MFCC特征]
    DD --> HH[DB能量特征]
    DD --> II[掩码率统计]

    M --> U
    S --> Y
    P --> EE
    R --> FF
    T --> GG
    V --> HH
    W --> HH
    CC --> II

    style A fill:#e3f2fd,stroke:#1976d2
    style D fill:#fff3e0,stroke:#f57c00
    style F fill:#e8f5e8,stroke:#388e3c
    style K fill:#fce4ec,stroke:#c2185b
    style P fill:#f3e5f5,stroke:#7b1fa2
    style S fill:#e1f5fe,stroke:#0277bd
    style Y fill:#e0f2f1,stroke:#00796b
    style BB fill:#fff8e1,stroke:#f57f17
    style DD fill:#f1f8e9,stroke:#689f38
```

### 2.3 时序处理流程图

```mermaid
sequenceDiagram
    participant F as 音频文件
    participant G as get_pcm()
    participant P as 音频预处理
    participant H as 分帧器
    participant E as 特征提取器
    participant N as 归一化器
    participant A as 数据增强
    participant O as 输出

    F->>G: 请求加载音频
    alt 缓存存在
        G->>G: 读取缓存文件
    else 缓存不存在
        G->>G: 原始文件读取
        G->>G: 重采样处理
        G->>G: 保存缓存
    end
    G->>P: 返回PCM数据

    P->>P: 幅值归一化
    P->>P: 补零处理
    P->>H: 处理后音频

    loop 时间窗口处理
        H->>H: 分帧处理
        H->>E: 帧信号

        E->>E: FFT计算
        E->>E: Mel滤波
        E->>E: 对数变换
        E->>E: MFCC计算
        E->>N: 提取特征

        N->>N: 特征归一化
        N->>A: 归一化特征

        alt 启用数据增强
            A->>A: DropBlock掩码
            A->>O: 增强后特征
        else 不增强
            A->>O: 原始特征
        end
    end

    O->>O: 输出[fft, fbank, mfcc, db]
```

### 2.4 数据流图

```mermaid
graph LR
    subgraph "数据输入"
        A["音频文件\n16kHz mono"]
    end

    subgraph "预处理阶段"
        B["缓存层\ncache_resample/"]
        C["数据层\nPCM数组"]
    end

    subgraph "特征提取阶段"
        D["时序层\n分帧处理"]
        E["频域层\nFFT+Mel"]
        F["特征层\n多维度输出"]
    end

    subgraph "输出层"
        G["fft\n频谱特征"]
        H["fbank\nMel频谱"]
        I["mfcc\n倒谱特征"]
        J["db\n能量特征"]
        K["mask\n掩码统计"]
    end

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    F --> H
    F --> I
    F --> J
    F --> K

    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style C fill:#e8f5e8
    style D fill:#fff3e0
    style E fill:#fce4ec
    style F fill:#e0f2f1
    style G fill:#fff8e1
    style H fill:#fff8e1
    style I fill:#fff8e1
    style J fill:#fff8e1
    style K fill:#ffe0b2
```

## 3. 关键参数配置

### 3.1 音频参数
```python
# 基础音频参数
SR = 16000                    # 采样率：16kHz
NFFT = 1024                   # FFT点数：1024
HOP = 500                     # 帧移：500 samples (约31.25ms)
N_MELS = 32                   # Mel滤波器数量：32
MEL_FMIN = 250                # Mel滤波器最低频率：250Hz
MEL_FMAX = 8000               # Mel滤波器最高频率：8000Hz
```

### 3.2 特征参数
```python
# FBank参数
IF_FFT = True                 # 使用FFT
N_MFCC = 16                   # MFCC系数数量：16

# 时序参数
Y_SLICE_LEN = 30              # 音频片段长度：30秒
Y_PAD_LEN = 2                  # 填充长度：2秒
DURATION = 25 + Y_PAD_LEN      # 总时长：27秒
FRAME_PRE_SECOND = int(SR / HOP)  # 每秒帧数：32帧
```

### 3.3 缓存配置
```python
# 缓存路径
CACHE_RESAMPLE_PATH = './cache_resample'  # 重采样缓存目录
FEAT_USE_ORI = False          # 是否使用原始特征
STRIDE = 1                    # 处理步长
```

## 4. 特征提取完整流程

### 4.1 数据加载模块 (get_pcm 函数)

**功能**：加载和重采样音频文件
```python
def get_pcm(file_name, sr=16000, offset=0.0, duration=None, **kwargs):
```

**处理步骤**：
1. **缓存检查**：检查是否存在重采样缓存文件
2. **音频加载**：使用 `librosa.load()` 加载原始音频
3. **重采样处理**：如果采样率不是16kHz，进行重采样
4. **数据补零**：如果音频长度不足，进行补零处理
5. **数据有效性检查**：检查是否存在NaN值

**缓存机制**：
- 缓存文件名格式：`cache_{sr}_{filename}.wav`
- 自动检测并创建缓存目录
- 支持音频片段加载 (offset, duration)

### 4.2 特征提取模块 (Feature_net 类)

#### 4.2.1 初始化配置
```python
class FeatureExtractor:
    def __init__(self, config: FeatureConfig):
```

**关键配置**：
- `feature_type`: 基础特征类型 ('fbank', 'mfcc', 'all')
- `use_fbank_norm`: 是否使用FBank归一化
- `use_db_norm`: 是否使用dB特征归一化
- `use_delta`: 是否添加时间差分特征
- `use_freq_delta`: 是否添加频率差分特征
- `use_db_feature`: 是否添加能量特征（2通道）
- `fbank_decay`: FBank归一化衰减系数

#### 4.2.2 预处理步骤

**1. 信号预处理**：
```python
def preemph(self, signal, coeff=0.95):
    return tf.concat([signal[:, 0:1], signal[:, 1:] - coeff * signal[:, :-1]], axis=-1)
```

**2. 分帧处理**：
```python
pcm = tf.pad(pcm, [[0, 0], [NFFT - HOP, 0]])  # 边界填充
framed_signals = tf.signal.frame(pcm, NFFT, HOP, pad_end=False)
```

**3. 计算能量特征**：
```python
pcm_energy = tf.square(pcm)
energy_frames = tf.signal.frame(pcm_energy, NFFT, HOP, pad_end=False)
sound_db_average = energy_to_db(tf.reduce_mean(energy_frames, axis=-1))
sound_db_hann = energy_frames * self.window / self.window_area
sound_db_average_hann = energy_to_db(tf.reduce_mean(sound_db_hann, axis=-1))
```

#### 4.2.3 特征提取核心算法

**1. extract() 方法** - 返回所有特征：
```python
def extract(self, y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
    # Returns: {
    #   'fbank': (n_mels, frames),
    #   'mfcc': (n_mfcc, frames),
    #   'db': (2, frames)  # [avg_energy, weighted_energy]
    # }
```

**2. Mel滤波器组**：
```python
self._mel_matrix = librosa.filters.mel(
    sr=sr,
    n_fft=self.config.n_fft,
    n_mels=self.config.n_mels,
    fmin=self.config.fmin,
    fmax=self.config.fmax
)
mel_spectrum = np.dot(mel_matrix, fft_magnitude)
log_mel = np.log(np.maximum(mel_spectrum, 1e-8))
```

**3. MFCC特征提取**：
```python
mfcc = librosa.feature.mfcc(
    S=log_mel,
    n_mfcc=self.config.n_mfcc,
    n_mels=self.config.n_mels,
    fmin=self.config.fmin,
    fmax=self.config.fmax
)  # (n_mfcc, frames)
```

**4. FBank特征归一化**：
```python
if self.config.use_fbank_norm:
    fbank_max = np.max(fbank, axis=0, keepdims=True)
    fbank_min = np.min(fbank, axis=0, keepdims=True)
    # 指数平滑
    for t in range(1, fbank_max.shape[1]):
        prev, curr = smoothed[0, t-1], fbank_max[0, t]
        smoothed[0, t] = decay * prev + (1 - decay) * curr if curr > prev else curr
    # 归一化到[0, 1]
    fbank_norm = (fbank - fbank_min) / (fbank_max - fbank_min)
    fbank_norm = np.clip(fbank_norm, 0.0, 1.0)
```

#### 4.2.4 extract_with_deltas() 主API

**功能**：根据配置提取基础特征并添加可选的delta特征和能量特征

```python
def extract_with_deltas(self, y: np.ndarray, sr: int) -> np.ndarray:
    # 1. 提取所有基础特征
    features = self.extract(y, sr)

    # 2. 根据feature_type选择基础特征
    base = self._get_base_features(features)  # [T, base_dim]

    # 3. 添加可选特征
    components = [base]
    if self.config.use_delta:
        components.append(self._compute_delta(base, axis=0))  # 时间差分
    if self.config.use_freq_delta:
        components.append(self._compute_delta(base, axis=1))  # 频率差分
    if self.config.use_db_feature:
        components.append(features['db'].T)  # 能量特征 (2通道)

    # 4. 拼接所有特征
    return np.concatenate(components, axis=1)
```

**特征维度计算**：
| 配置 | Base | +delta | +freq_delta | +db |
|------|------|--------|-------------|-----|
| fbank (64) | [T, 64] | [T, 128] | [T, 192] | [T, 66]/[T, 130]/[T, 194] |
| mfcc (16) | [T, 16] | [T, 32] | [T, 48] | [T, 18]/[T, 34]/[T, 50] |
| all (64+16) | [T, 80] | [T, 160] | [T, 240] | [T, 82]/[T, 162]/[T, 242] |

### 4.3 批量处理模块 (get_feats_from_file 函数)

**功能**：从音频文件批量提取特征
```python
def get_feats_from_file(file_name, length_limit=None, cache_path=None)
```

**处理流程**：
1. **缓存检查**：检查特征缓存是否存在
2. **音频加载**：使用 `get_pcm()` 加载音频
3. **音频处理**：
   - 补零处理：`y = np.concatenate([y, np.zeros(64000,)])`
   - 最小时长保证：不足7秒的音频循环复制
   - 幅值归一化：`y = librosa.util.normalize(y)`
4. **分窗口处理**：将音频分割为多个时间窗口
5. **结果返回**：返回包含PCM数据的字典列表

### 4.4 单次特征提取模块 (get_feat_from_pcm 函数)

**功能**：直接从PCM数据提取特征
```python
def get_feat_from_pcm(y, mask_callback=None)
```

**输出特征**：
- `fbank`: Mel频谱特征 (n_mels, frames)
- `mfcc`: MFCC特征 (n_mfcc, frames)
- `db`: 能量特征 (2, frames) - [平均能量, 加权能量]

**extract_with_deltas() 输出**：
- 形状: (frames, feature_dim)
- feature_dim 取决于配置，详见 4.2.4 维度表

## 5. 输出特征说明

### 5.1 主要API方法

**extract(y, sr)** - 提取所有基础特征：
```python
features = extractor.extract(y, sr)
# Returns: {
#   'fbank': (n_mels, frames),
#   'mfcc': (n_mfcc, frames),
#   'db': (2, frames)
# }
```

**extract_with_deltas(y, sr)** - 根据配置组合特征：
```python
features = extractor.extract_with_deltas(y, sr)
# Returns: (frames, feature_dim)
# feature_dim 取决于 feature_type, use_delta, use_freq_delta, use_db_feature
```

### 5.2 特征维度参考表

| 配置 | Base | +delta | +freq_delta | +db (2通道) |
|------|------|--------|-------------|-------------|
| fbank (64) | [T, 64] | [T, 128] | [T, 192] | [T, 66] / [T, 130] / [T, 194] |
| mfcc (16) | [T, 16] | [T, 32] | [T, 48] | [T, 18] / [T, 34] / [T, 50] |
| all (64+16) | [T, 80] | [T, 160] | [T, 240] | [T, 82] / [T, 162] / [T, 242] |

**注**：db 特征包含 2 通道（平均能量 + 加权能量）

### 5.3 特征归一化

#### 5.3.1 能量到dB转换
```python
def energy_to_db(energy: np.ndarray, ref: float = 1.0, amin: float = 1e-8) -> np.ndarray:
    db = 10 * np.log10(np.maximum(energy, amin) / ref)
    # 归一化到[0, 1]范围（clip在-8dB）
    return (np.maximum(db, -8) + 8) / 8
```

#### 5.3.2 FBank归一化
```python
def _normalize_fbank(self, fbank: np.ndarray) -> np.ndarray:
    decay = self.config.fbank_decay

    # 计算每帧的最大/最小值
    fbank_max = np.max(fbank, axis=0, keepdims=True)
    fbank_min = np.min(fbank, axis=0, keepdims=True)

    # 指数平滑（attack-decay）
    if decay > 0 and fbank_max.shape[1] > 1:
        smoothed = np.zeros_like(fbank_max)
        smoothed[0, 0] = fbank_max[0, 0]
        for t in range(1, fbank_max.shape[1]):
            prev, curr = smoothed[0, t-1], fbank_max[0, t]
            smoothed[0, t] = decay * prev + (1 - decay) * curr if curr > prev else curr
        fbank_max = smoothed

    # 归一化到[0, 1]
    range_val = np.where(fbank_max - fbank_min > 0, fbank_max - fbank_min, 1.0)
    fbank_norm = (fbank - fbank_min) / range_val
    return np.clip(fbank_norm, 0.0, 1.0)
```

## 6. 技术特点

### 6.1 设计特点
- **NumPy/Librosa实现**：纯NumPy实现，无深度学习框架依赖
- **灵活配置**：通过FeatureConfig灵活组合特征类型和选项
- **统一格式**：extract()返回统一格式 `[feature_dim, frames]`
- **Delta计算**：支持时间和频率维度的差分特征计算

### 6.2 鲁棒性设计
- **异常处理**：完善的错误捕获和处理
- **边界检查**：数据长度和有效性验证
- **数值稳定**：使用小常数避免除零错误
- **NaN处理**：检查和处理NaN值

## 7. 应用场景

### 7.1 主要用途
- **婴儿啼哭检测**：核心应用场景
- **音频分类**：通用音频任务
- **语音识别**：预处理特征提取
- **声学事件检测**：环境音频分析

### 7.2 推荐配置

**FBank + Delta + DB 配置**（推荐）：
```yaml
feature:
  feature_type: 'fbank'
  n_mels: 64
  n_fft: 1024
  hop_length: 500
  fmin: 250
  fmax: 8000
  use_delta: true          # 添加时间差分
  use_freq_delta: false    # 不添加频率差分
  use_db_feature: true     # 添加能量特征（2通道）
  use_db_norm: true        # 能量归一化到[0,1]
  use_fbank_norm: true     # FBank归一化
```

**MFCC 配置**：
```yaml
feature:
  feature_type: 'mfcc'
  n_mels: 64
  n_mfcc: 16
  # ...其他参数
```

**All (FBank + MFCC) 配置**：
```yaml
feature:
  feature_type: 'all'      # FBank和MFCC拼接
  n_mels: 64
  n_mfcc: 16
  use_delta: true
  # ...其他参数
```

## 8. 性能参数

### 8.1 处理速度
- **采样率**: 16kHz
- **帧率**: 32 FPS (frames per second)
- **延迟**: ~31.25ms per frame
- **实时性**: 支持实时处理

### 8.2 内存使用
- **缓存优化**: 显著减少I/O开销
- **批量处理**: 高效的内存管理
- **精度控制**: float32以保证精度

这个系统提供了一个完整的高质量音频特征提取流水线，特别适用于婴儿啼哭检测等音频分类任务。
