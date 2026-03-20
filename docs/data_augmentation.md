# 数据增强策略与流程

本文档详细描述婴儿哭声检测系统的数据增强策略，包括标签感知的Mixup增强、Sox音频效果链、噪声注入和动态增益调整。

## 数据读取与增强整体流程

```mermaid
flowchart TD
    subgraph DataLoading["数据读取阶段"]
        A["CryDataset\n__getitem__"] --> B["加载音频片段\nAudioReader.load_by_time"]
        B --> C{需要补全?}
        C -->|是| D["填充/截断\npad_pcm"]
        C -->|否| E["原始波形"]
        D --> F["AudioAugmenter.augment"]
        E --> F
    end

    subgraph Augmentation["数据增强阶段"]
        F --> G{标签判断}
        G -->|非Cry| H["50%概率\n时间反转"]
        G -->|Cry| I["保持原样"]
        H --> J{Mixup位置}
        I --> J

        J -->|70%概率前置| K["Mixup增强\ndo_mixup"]
        J -->|30%概率后置| L["跳过"]
        K --> M{Sox效果链?}
        L --> M

        M -->|Cry: 90%| N["Sox效果链"]
        M -->|非Cry: 60%| N
        M -->|不增强| O["跳过效果链"]

        N --> P["选择效果\nPitch/Reverb/Phaser\n/TimeStretch/Echo"]
        P --> Q{效果数 > 3?}
        Q -->|是| R["分两组执行\n降低内存占用"]
        Q -->|否| S["一次执行"]
        R --> T["噪声注入 10%"]
        S --> T
        O --> T

        T --> U{Mixup后置?}
        U -->|是| V["Mixup增强"]
        U -->|否| W["跳过"]
        V --> X["动态增益调整"]
        W --> X
    end

    subgraph Output["输出阶段"]
        X --> AA["增强后波形\n5s @ 16kHz"]
        AA --> AB["特征提取\nFeatureExtractor"]
        AB --> AC["模型训练输入\n[T, F]"]
    end

    style DataLoading fill:#e1f5fe,stroke:#0277bd
    style Augmentation fill:#fff3e0,stroke:#f57c00
    style Output fill:#e8f5e8,stroke:#388e3c
```

## 数据增强详细流程

```mermaid
sequenceDiagram
    participant D as Dataset
    participant A as AudioAugmenter
    participant M as Mixup
    participant S as Sox
    participant N as Noise

    D->>A: augment(waveform, label)
    Note over A: Step 1: 标签判断

    alt 非Cry 且 50%概率
        A->>A: 时间反转(reverse)
    end

    Note over A: Step 2: Mixup前置(70%概率)
    alt 70%概率前置Mixup
        A->>M: do_mixup(y, label)
        M-->>A: 混合后波形
    end

    Note over A: Step 3: Sox效果链
    alt Cry:90%概率 或 非Cry:60%概率
        A->>S: 选择效果列表
        S->>S: Pitch(50%概率)
        S->>S: Reverb(80%概率)
        S->>S: Phaser(50%概率)
        S->>S: TimeStretch(配置概率，默认0)

        alt 非Cry样本
            S->>S: Echo(50%概率)
        end

        alt 选中效果数 > 3
            S->>S: 分两组执行，降低内存
        else 效果数 ≤ 3
            S->>S: 一次执行
        end

        S-->>A: 效果链处理后的波形
    end

    Note over A: Step 4: 噪声注入(10%概率)
    alt 10%概率
        A->>N: 添加噪声(白/粉红/环境, SNR 5-25dB)
        N-->>A: 加噪波形
    end

    Note over A: Step 5: Mixup后置(30%概率)
    alt 未前置Mixup 且 30%概率
        A->>M: do_mixup(y, label)
        M-->>A: 混合后波形
    end

    Note over A: Step 6: 动态增益(90%概率)
    alt 90%概率
        A->>A: 增益调整\n90%: 高斯衰减\n10%: 小幅增益(0-10dB)
    end

    A-->>D: 增强后波形
```

## Mixup增强详细流程

### Mixup规则矩阵

| 样本类型 | Mixup概率 | Mixup率分布 | 混合样本来源 | 能量约束 |
|---------|----------|------------|-------------|---------|
| **Cry** | 0.3 | 均值0.3, 标准差0.15, 裁剪至[0.1, 0.65] | 任意样本 | 混合样本能量 < 原始能量 - (3~10)dB |
| **非Cry** | 0.3 | 随机均匀分布 | 仅非Cry样本 | 无特殊约束 |

### Mixup流程图

```mermaid
flowchart TD
    A["do_mixup\ny, label"] --> B{标签判断}

    B -->|Cry| C["Mixup率\nN(0.3, 0.15)\n裁剪0.1-0.65"]
    B -->|非Cry| D["Mixup率\n均匀随机\n0.0-1.0"]

    C --> E[选择混合样本]
    D --> F["选择混合样本\n仅非Cry"]

    E --> G["从file_schedule_dict\n随机加载音频片段"]
    F --> G

    G --> H{Cry样本?}
    H -->|是| I[能量约束]
    H -->|否| J[直接混合]

    I --> K[计算dB差]
    K --> L{mix_db >= original_db?}
    L -->|是| M["降低mix能量\n目标差3-10dB"]
    L -->|否| J
    M --> J

    J --> N["随机位置混合\ny[st:st+len] += y_mix"]
    N --> O["Clip限幅\n-1, 1"]
    O --> P[返回增强波形]

    style A fill:#e1f5fe,stroke:#0277bd
    style I fill:#fff3e0,stroke:#f57c00
    style P fill:#e8f5e8,stroke:#388e3c
```

## Sox效果链配置

### 效果链结构

```mermaid
flowchart LR
    subgraph Selection["效果选择（独立概率）"]
        direction TB
        P[Pitch<br/>50%概率<br/>-4~+4半音]
        R[Reverb<br/>80%概率<br/>随机混响参数]
        Ph[Phaser<br/>50%概率<br/>随机相位参数]
        TS[TimeStretch<br/>配置概率，默认0<br/>0.8~1.2速度因子]
        E[Echo<br/>50%概率<br/>仅非Cry]
    end

    subgraph Execution["效果执行（内存优化）"]
        direction TB
        EX1{"效果数 > 3?"}
        EX1 -->|是| G1["前半组执行"]
        G1 --> G2["后半组执行"]
        EX1 -->|否| G3["一次执行"]
    end

    Selection --> Execution

    style P fill:#e3f2fd,stroke:#1976d2
    style R fill:#f3e5f5,stroke:#7b1fa2
    style Ph fill:#e8f5e8,stroke:#388e3c
    style TS fill:#fce4ec,stroke:#c62828
    style E fill:#fff3e0,stroke:#f57c00
```

### 各效果详细参数

#### Pitch (音高变换)
```python
pitch_rate = (random() - 0.5) * 8  # 范围: -4 到 +4 半音
```

#### Reverb (混响)
```python
{
    'reverberance': random() * 80 + 20,      # 20-100
    'high_freq_damping': random() * 100,      # 0-100
    'room_scale': random() * 100,             # 0-100
    'stereo_depth': random() * 100,           # 0-100
    'pre_delay': 0
}
```

#### Phaser (相位器)
```python
{
    'gain_in': random() * 0.5 + 0.5,          # 0.5-1.0
    'gain_out': random() * 0.5 + 0.5,         # 0.5-1.0
    'delay': random_int(1, 5),                # 1-5 ms
    'decay': random() * 0.4 + 0.1,            # 0.1-0.5
    'speed': random() * 1.9 + 0.1,            # 0.1-2.0 Hz
    'modulation_shape': random(['sinusoidal', 'triangular'])
}
```

#### TimeStretch (时间伸缩)
```python
duration_factor = random.uniform(0.8, 1.2)  # <1=加速, >1=减速

# 算法选择：
# 0.9~1.1: sox stretch (SOLA，适合小幅变化)
if 0.9 <= duration_factor <= 1.1:
    tfm.stretch(duration_factor, window=random.uniform(15, 25))
# 其他: sox tempo (WSOLA，适合大幅变化)
else:
    speed_factor = 1.0 / duration_factor
    # 确保 speed_factor 不落在 stretch 的最佳范围内
    if 0.9 <= speed_factor <= 1.1:
        speed_factor = 1.11
    tfm.tempo(speed_factor)
```

**注意**：TimeStretch 默认概率为 0，可在配置文件中启用：
```yaml
augmentation:
  time_stretch_prob: 0.3
```

#### Echo (回声 - 仅非Cry)
```python
{
    'gain_in': random() * 0.5 + 0.5,          # 0.5-1.0
    'gain_out': random() * 0.5 + 0.5,         # 0.5-1.0
    'n_echos': 1,
    'delays': [random_int(6, 60)],            # 6-60 ms
    'decays': [random() * 0.5]                # 0-0.5
}
```

## 配置参数汇总

### AugmentationConfig

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `cry_aug_prob` | 0.9 | Cry样本应用Sox效果链的概率 |
| `other_aug_prob` | 0.6 | 非Cry样本应用Sox效果链的概率 |
| `other_reverse_prob` | 0.5 | 非Cry样本时间反转概率 |
| `pitch_prob` | 0.5 | 音高变换效果概率 |
| `reverb_prob` | 0.8 | 混响效果概率 |
| `phaser_prob` | 0.5 | 相位器效果概率 |
| `echo_prob` | 0.5 | 回声效果概率（仅非Cry） |
| `time_stretch_prob` | 0.0 | 时间伸缩效果概率（默认禁用） |
| `noise_prob` | 0.1 | 噪声注入概率 |
| `gain_prob` | 0.9 | 动态增益调整概率 |

### MixupConfig

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `cry_mix_prob` | 0.3 | Cry样本Mixup概率 |
| `cry_mix_rate_mean` | 0.3 | Cry样本Mixup率均值 |
| `cry_mix_rate_std` | 0.15 | Cry样本Mixup率标准差 |
| `other_mix_prob` | 0.3 | 非Cry样本Mixup概率 |
| `mix_front_prob` | 0.7 | Mixup前置（效果链之前）概率 |

### NoiseConfig

| 参数 | 默认值 | 说明 |
|-----|-------|------|
| `white_noise_prob` | 0.3 | 白噪声权重 |
| `pink_noise_prob` | 0.4 | 粉红噪声权重 |
| `ambient_noise_prob` | 0.3 | 环境噪声权重（需配置文件列表） |
| `snr_min` | 5.0 | 最小信噪比 (dB) |
| `snr_max` | 25.0 | 最大信噪比 (dB) |
| `ambient_noise_dir` | None | 环境噪声文件目录 |
| `ambient_noise_files` | () | 环境噪声文件列表 |
| `pink_noise_alpha` | 1.0 | 粉红噪声 1/f^alpha 参数 |

## 标签感知增强策略

### 增强策略对比

```mermaid
flowchart TB
    subgraph Cry["Cry 样本 (婴儿哭声)"]
        direction TB
        C1[时间反转: ❌ 不应用]
        C2["Mixup: ✅ 30%概率\n混合率: N(0.3, 0.15)\n能量约束: 混合样本低3-10dB"]
        C3["Sox效果链: ✅ 90%概率\nPitch/Reverb/Phaser/TimeStretch\n❌ 不包含Echo"]
        C4["噪声注入: 10%概率\n白/粉红/环境噪声 SNR 5-25dB"]
        C5["动态增益: 90%概率\n模拟不同距离/音量"]

        C1 --> C2 --> C3 --> C4 --> C5
    end

    subgraph NonCry["非Cry 样本 (环境/其他声音)"]
        direction TB
        N1["时间反转: ✅ 50%概率\n增加时间不变性"]
        N2["Mixup: ✅ 30%概率\n混合率: 均匀随机\n约束: 仅非Cry样本"]
        N3["Sox效果链: ✅ 60%概率\nPitch/Reverb/Phaser/TimeStretch\n✅ 包含Echo"]
        N4["噪声注入: 10%概率"]
        N5[动态增益: 90%概率]

        N1 --> N2 --> N3 --> N4 --> N5
    end

    style Cry fill:#ffebee,stroke:#c62828
    style NonCry fill:#e3f2fd,stroke:#1565c0
```

## 内存优化：效果分组执行

Sox的时间伸缩（WSOLA算法）在处理长音频时内存占用较大。为降低内存峰值，当选中效果超过3个时，自动分成两组执行：

```
选中效果: [pitch, reverb, phaser, time_stretch]  (4个)
    ↓
第1组: [pitch, reverb]    → sox执行 → 中间波形
第2组: [phaser, time_stretch] → sox执行 → 最终波形

选中效果: [pitch, reverb, phaser]  (3个，不分组)
    ↓
一次执行: [pitch, reverb, phaser] → sox执行 → 最终波形
```

## 数据流示例

```
原始音频 (5秒 @ 16kHz)
    │
    ▼
┌────────────────────────────────────────┐
│  CrySample: "婴儿哭声.wav"              │
│  - 是否反转: 否 (Cry不反转)              │
│  - Mixup前置: 是 (70%概率)              │
│    └── 加载随机哭声片段                  │
│    └── 能量约束: 降低8dB                │
│    └── 混合位置: 随机                   │
│  - Sox效果链: 应用 (90%概率)             │
│    └── pitch(-1.2半音) ► reverb        │
│  - 噪声注入: 跳过 (90%跳过)              │
│  - Mixup后置: 跳过 (已前置)              │
│  - 增益调整: -15dB (模拟远距离)          │
└────────────────────────────────────────┘
    │
    ▼
增强后音频 ──▶ 特征提取 ──▶ 模型训练

────────────────────────────────────────────────

原始音频 (5秒 @ 16kHz)
    │
    ▼
┌────────────────────────────────────────┐
│  OtherSample: "电视声音.wav"            │
│  - 是否反转: 是 (50%概率)               │
│  - Mixup前置: 否 (30%概率)              │
│  - Sox效果链: 应用 (60%概率)             │
│    └── phaser ► echo(50ms) ► reverb    │
│  - 噪声注入: 添加粉红噪声 (SNR 15dB)     │
│  - Mixup后置: 是                        │
│    └── 加载随机非哭声片段                │
│    └── 无能量约束                       │
│  - 增益调整: +5dB                       │
└────────────────────────────────────────┘
    │
    ▼
增强后音频 ──▶ 特征提取 ──▶ 模型训练
```

## 设计要点

1. **标签差异化处理**: Cry和非Cry采用不同的增强策略，Cry更注重保持声学特征完整性，非Cry更注重多样性

2. **能量感知Mixup**: Cry样本混合时强制要求混合样本能量更低，模拟背景噪声而非主导声音

3. **样本隔离**: 非Cry样本Mixup时排除Cry样本，防止环境声音被哭声污染

4. **效果链动态组合**: 效果应用顺序随机，增加组合多样性

5. **内存优化分组**: 效果数超过3个时自动分两组执行，避免Sox时间伸缩算法的内存峰值问题

6. **时间伸缩算法自适应选择**: 根据伸缩因子自动选择 SOLA (stretch) 或 WSOLA (tempo) 算法，确保最优音质

7. **时间反转数据增强**: 仅应用于非Cry，增加时间不变性训练

8. **多类型噪声注入**: 支持白噪声、粉红噪声和环境采样噪声，SNR范围5-25dB
