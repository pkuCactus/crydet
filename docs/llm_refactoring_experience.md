# 基于 LLM 重构音频分类项目的实操复盘

CryDet 是一个婴儿啼哭检测系统，用 Transformer 对音频做二分类。项目从 2026 年 2 月底启动，4 月初完成，35 天内产生 163 个 commit，其中 74 个由 LLM（Kimi 2.5）协作完成。本文记录实际使用 Claude Code CLI 的方式、踩过的坑，以及有效的用法模式。

---

## 项目速览

项目从零到跑通训练+推理+导出全流程，约 9800 行 Python 代码。前两周手动完成数据加载和增强模块，后续模型、训练流水线、DDP、性能优化等主要依赖 LLM 协作。

关键数据：

- 163 次提交，74 次带 `Co-Authored-By: Claude` 标记（占 45%）
- 开发周期 35 天（2026-02-26 ~ 2026-04-01）
- 日提交峰值 21 次（2026-03-11），当日完成模型+训练+评估+导出
- 最终结构：model/dataset/utils/configs/scripts 五个模块，职责分离

---

## LLM 协作方式

### 从设计文档生成模块骨架

编写 CryTransformer 模型时，先完成设计文档 docs/transformer_cry_detection_design.md，涵盖模型架构、特征方案、训练策略。然后将其作为输入，让 Claude Code 基于文档生成代码，同时在 prompt 中明确约束：

```
按照 docs/transformer_cry_detection_design.md 的设计方案，
实现 CryTransformer 模型、训练脚本、评估脚本和导出脚本。
注意以下约束：
- 输入格式是 [B, T, F]，不是图像的 [B, C, H, W]
- 用 Linear projection，不要用 Conv1d patch embedding
- T=157 对应 5 秒音频 @ 16kHz，hop_length=512
- 需要支持 DDP 多卡训练
```

2026-03-11 的 21 个 commit 即由此产生。后续还需运行修复边界 bug，但骨架一次成型。

要点：**LLM 的输出质量取决于输入的设计清晰度。** 若需求不明确，生成的代码方向容易偏移。

### 调试 DDP 死锁：完整上下文优于症状描述

DDP 死锁是本项目最难的问题，前后修复 4 次。首次沟通时仅描述症状（"多卡训练卡住"），得到的方案只修复了 schedule 长度不一致，`_compute_auc_distributed` 仍然卡住。

第二次沟通改为提供完整上下文：

```
训练在验证阶段的 _compute_auc_distributed 函数卡住。
相关代码：
- train.py 的分布式验证逻辑（附代码片段）
- _compute_auc_distributed 函数实现
- NCCL 日志显示卡在 all_gather 操作
- 运行环境：4 卡 GPU，PyTorch 2.6，CUDA 12.4

请分析根因，不要只给表面修复。
```

Claude Code 由此分析出三个根因：schedule 不一致、all_gather 异步操作竞争、torch.compile 与 NCCL 的冲突。逐一生成修复方案，逐一运行验证。最终定位到核心问题——`torch.cuda.synchronize()` 在之前的优化提交中被移除，而该提交同样由 LLM 生成。

要点：**提供错误日志、代码片段和运行环境信息，远比只描述症状有效。** LLM 能追踪代码中的状态流转路径，但需要足够的输入信息。

### 批量 Bug 发现：主动审查优于逐个修复

2026-03-23，执行"检查 train.py 里有没有 bug"后，Claude Code 列出 7 个问题：

1. resume 时 epoch 循环从 0 重启（range 起点硬编码）
2. EMA restore 缺少 try/finally 保护
3. 分布式 epoch_loss 被 world_size 放大（数学逻辑错）
4. warmup 首批次 lr 错误 + global_step 恢复缺失
5-7. 冗余逻辑和重复写入

逐个发现逐个修复预计需 1-2 天，一次性列出后约 1 小时完成。

要点：**代码编写完成后，主动让 LLM 做一轮审查。** 它以全局视角追踪状态在训练循环中的流转，批量发现能力远快于逐行排查。

### 性能优化：方向正确，但需完整验证

2026-03-27，让 Claude Code 分析训练流水线的性能瓶颈。它读取关键模块代码后输出性能分析报告（docs/performance_analysis.md），识别出：

- AudioReader 的 sox 调用为外部进程，GIL 阻塞
- FeatureExtractor 在训练循环中运行，GPU 等待 CPU
- mixup 时的额外磁盘读取

随后直接生成优化代码：

- AudioReader：LRU 内存缓存 + 线程局部 librosa → 25x 加速
- FeatureExtractor：torch.compile + GPU-only 操作 → 22x 加速
- AudioAugmenter：预生成噪声池 + 线程局部 sox → 减少 I/O 阻塞

同时生成 benchmark_performance.py 用于验证。

但 torch.compile 优化与 DDP 的 all_reduce/all_gather 冲突，导致训练再次卡住（详见后文"踩过的坑"）。

要点：**LLM 在性能优化上擅长识别瓶颈方向和生成优化代码，但优化结果必须通过完整的训练流程验证。** 单模块 benchmark 不代表分布式场景同样有效。

### 渐进式重构复杂模块

train.py（940 行）经历多轮简化，每轮只改一个维度：

```
03-23 19:01  简化 train.py 代码 → 去重复模式
03-23 19:15  简化 train.py 中的重复模式 → 提取公共逻辑
03-23 19:31  将 scheduler.step 移到训练步骤前面 → 修正训练循环语义
03-24 09:14  Simplify train.py and restructure configuration → 配置体系重构
```

每步完成后运行验证，确认无问题再继续下一步。

要点：**重构复杂模块时，渐进式优于一步到位。** 每次只改一个维度，中间每步都验证。

### 文档与代码同步更新

每次架构变更后，同步让 Claude Code 更新 CLAUDE.md 和 docs/。例如 2026-03-17 的大重构（config.py 移至 utils/、新增 4 个模块），代码变更完成后立即更新 CLAUDE.md 的目录结构、数据流说明、特征配置文档，并补充 Mermaid 流程图。

要点：**文档更新应作为代码变更的一部分，而非事后补充。** LLM 写文档效率高，但跨对话时不保留上下文，不及时更新则后续补写成本显著增加。

### CLAUDE.md 作为约束合约

CLAUDE.md 在项目中起核心作用——每次新对话 Claude Code 自动读取，获取以下约束：

- 模型用 Linear projection，不用 Conv1d
- 输入维度 [B, T, F]，T=157
- Mixup 规则：Cry 样本混合能量差 3-10 dB，非 Cry 只能和非 Cry 混
- Audio cache 用 file mtime 而非 MD5 hash
- DataLoader persistent_workers=False（避免 DDP 死锁）

其中 NOT 规则直接防止了 LLM 重复犯错。例如 Conv1d patch embedding 是 Transformer 视觉模型的常见做法，但对音频分类不适合——CLAUDE.md 中明确标注 NOT Conv1d 后，后续对话不再出现该问题。

要点：**CLAUDE.md 中具体的 NOT 规则比泛泛描述有效得多。** `Linear projection (NOT Conv1d patch embedding)` 防止了错误选择，而 `使用线性投影` 仅描述了正确选择。

---

## 踩过的坑

### LLM 优化引入新 Bug

DDP 死锁的循环是最典型的案例：

- 2026-03-28：Claude Code 建议"减少 DDP 同步点"，移除 `torch.cuda.synchronize()`
- 2026-03-30：训练再次卡住——all_gather 异步操作与后续代码竞争
- 修复 `_compute_auc_distributed` 同步问题
- 又发现 torch.compile 与 NCCL 冲突
- 最终回退至 2026-03-28 前的稳定版本

LLM 的优化理由（"减少同步开销"）理论上成立，但实际环境中异步操作间的依赖关系难以静态分析。

教训：**涉及并发和分布式场景的性能优化，必须跑完整流程验证。** 理论上合理的优化在实际运行中可能引入新的同步问题。

### API 版本不匹配

Claude Code 默认使用最新 API，项目环境为 PyTorch 2.6，出现多处兼容问题：

- `torch.amp.autocast` → PyTorch 2.0+ 需要 `device_type` 参数
- `GradScaler` API 变更
- `dist.init_process_group` → PyTorch 2.6+ 需要 `device_id`

每次均在运行时才发现，再逐一 Fix。

应对：**在对话开头指定 PyTorch/Python/CUDA 版本，或将版本信息写入 CLAUDE.md。**

### 并行化不优于串行

2026-03-17，让 Claude Code 将低能量样本过滤从串行改为并行（ProcessPoolExecutor），结果出现数据不一致。2026-03-19 回退为串行。

教训：**数据一致性敏感的操作，对 LLM 提议的并行化需格外谨慎。** LLM 不自动判断哪些操作可并行、哪些不可——需要人工判断。

### 过度设计倾向

Claude Code 曾提议给 loss 函数增加 Strategy 模式的抽象基类，便于未来扩展。但项目只需 focal + label_smoothing + combined 三种 loss，工厂函数已足够，Strategy 模式属于多余抽象。

在 CLAUDE.md 中增加约束以抑制该倾向：

```
请使用第一性原理思考。如果动机和目标不清晰，停下来和我讨论。
```

教训：**对 LLM 提出的抽象和设计模式，先判断当前是否需要。** YAGNI 原则在与 LLM 协作时尤其重要——它容易为当前不需要的灵活性添加代码。

---

## 适用场景边界

### 适合让 LLM 做的事

1. **从设计文档生成模块代码骨架** — 前提是有清晰的设计文档
2. **批量代码审查发现 Bug** — 编写完成后主动审查
3. **配置体系设计** — dataclass + YAML 配置生成规范高效
4. **技术文档编写与可视化** — Mermaid 流程图、API 文档生成
5. **样板代码生成** — 训练循环、评估脚本、导出脚本等模式化代码
6. **性能瓶颈分析与优化方向** — 识别热点快于人工，但落地需自行验证

### 不适合完全交给 LLM 的事

1. **分布式训练调试** — NCCL/GPU 同步问题需实际运行诊断
2. **性能优化的最终验证** — 单模块快不代表分布式场景没问题
3. **业务规则细节** — Mixup 能量差阈值、采样权重等需领域知识
4. **运行时版本兼容性** — LLM 默认使用最新 API，与旧环境可能冲突
5. **重构方向决策** — 哪些代码值得重构、哪些不宜变动，由人判断

---

## 工具使用细节

### Claude Code CLI 工作流

日常开发流程：

1. 启动 Claude Code，自动读取项目 CLAUDE.md
2. 描述任务（如"给 AudioReader 加内存缓存"、"检查 train.py 的 bug"）
3. Claude Code 读取相关文件、生成方案、直接编辑代码
4. 运行代码验证，有问题则继续对话迭代
5. 确认无误后，让 Claude Code 生成 commit 消息并提交

常用操作：

| 操作 | 示例指令 | 工具 |
|------|---------|------|
| 读取文件 | "读一下 dataset/feature.py" | Read |
| 修改代码 | "在 audio_reader.py 里加 LRU 缓存" | Edit |
| 创建文件 | "写个 benchmark_performance.py" | Write |
| 执行命令 | "跑一下训练" | Bash |
| 提交变更 | "提交这次修改" | Git commit |

### 提交消息格式

LLM 参与的 commit 自动附带结构化消息：

```
Feat: 重构训练代码架构，新增 EMA、调度器、能量特征等多项功能

主要变更：
1. 模块重构：将 config.py 移动到 utils/config.py
2. 新增 model/ema.py：指数移动平均实现
3. 新增 model/scheduler.py：Warmup + Cosine Decay

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
```

此格式使 git log 易读，同时自动标记 LLM 参与的提交。

### CLAUDE.md 的有效写法

实践中最有价值的部分是**具体约束和 NOT 规则**：

```markdown
## Model Architecture
CryTransformer uses Linear projection (NOT Conv1d patch embedding):
- Input format: [B, T, F] (batch, time_frames, feature_dim)
- Time dimension T=157 for 5s audio @ 16kHz with hop_length=512

## Mixup Rules
- Cry samples: Mixup sample energy must be 3-10 dB lower than original
- Non-cry samples: Mixup can only use non-cry labels

## Performance Optimizations
- DataLoader persistent_workers=False to avoid DDP deadlocks
- Audio cache uses file mtime (not MD5 hash) for cache validation
```

NOT 规则和具体数值直接阻止 LLM 在后续对话中重复犯错。相比之下，"使用线性投影"、"避免死锁"等软描述约束力不足。

---

## 效率对比

| 指标 | 纯手动（估计） | LLM 协作（实际） |
|------|-------------|-------------|
| 从零到跑通训练 | ~2-3 周 | 2 天（03-02 ~ 03-11） |
| 架构重构 6 个新模块 | ~3-5 天 | 1 天（03-17） |
| 找出并修 7 个 Bug | ~1-2 天 | ~1 小时 |
| 性能优化 25x 加速 | ~3-5 天 | 1 天（03-27） |
| 写 7 个技术文档 | ~2-3 天 | ~1 天 |

并非所有环节均有加速——DDP 死锁调试耗时可能高于纯手动，因 LLM 优化引入了新问题。但从设计到可运行代码的整体转化速度显著提升。

---

## 总结

核心经验：**人负责方向与判断，LLM 负责转化与执行。**

- 有设计文档，LLM 才能把设计变成代码；无设计则方向易偏。
- 有完整上下文，LLM 才能有效调试；仅描述症状则只能猜测。
- 有运行验证，LLM 才能安全优化；单模块通过不代表分布式场景通过。
- 有 CLAUDE.md 约束，LLM 才不重复犯错；NOT 规则比描述有效。
- 有重构方向判断，LLM 才能正确执行；不宜让 LLM 替人决定重构什么。

CLAUDE.md 是协作模式中最关键的桥梁——将每次验证过的决策凝固为规则，使后续对话中的 LLM 自动遵循。缺少它，每次新对话从零开始，之前的教训无法延续。