# 用 Claude Code 重构一个音频分类项目：实操复盘

CryDet 是一个婴儿啼哭检测系统，用 Transformer 对音频做二分类。从 2026 年 2 月底开始写，到 4 月初基本完成，35 天里提交 163 个 commit，其中 74 个是和 Claude Code（当时用 kimi 2.5）协作完成的。这篇文档不讲理论，讲我实际怎么用这个工具的，踩了什么坑，哪些用法有效，哪些没用。

---

## 项目速览

整个项目从零到跑通训练+推理+导出，大概 9800 行 Python 代码。我一个人做，主要工具是 Claude Code CLI。项目是从一个 Initial commit 开始，慢慢写出来的——前两周基本手动写数据加载和增强，后面模型、训练流水线、DDP、性能优化这些重活都靠 Claude Code 帮忙。

关键数据：

- 163 次提交，74 次带 `Co-Authored-By: Claude` 标记
- 开发周期 35 天（2026-02-26 ~ 2026-04-01）
- 最忙的一天 3 月 11 号，21 个 commit，那一天完成了模型、训练、评估、导出全流程
- 项目最终结构：model/dataset/utils/configs/scripts/scripts 五个模块，清晰分离

---

## 我怎么用 Claude Code 的

### 启动一个新模块：直接给它设计文档

最开始写 CryTransformer 模型的时候，我已经有了一份设计文档（docs/transformer_cry_detection_design.md），里面写了模型架构、特征方案、训练策略。我直接把设计文档丢给 Claude Code，让它基于文档生成代码。

具体做法是这样的：

```bash
# 在 Claude Code 的对话里，我这样说：
"按照 docs/transformer_cry_detection_design.md 的设计方案，
实现 CryTransformer 模型、训练脚本、评估脚本和导出脚本。
注意以下约束：
- 输入格式是 [B, T, F]，不是图像的 [B, C, H, W]
- 用 Linear projection，不要用 Conv1d patch embedding
- T=157 对应 5 秒音频 @ 16kHz，hop_length=512
- 需要支持 DDP 多卡训练"
```

它一天就全部写出来了——3 月 11 号那 21 个 commit 就是这么来的。当然后面还需要跑起来修各种小 bug，但骨架一次成型，省了很多时间。

这个方法的关键是：**你得先有设计文档**。Claude Code 不是替你做设计，它是帮你把设计变成代码。如果你自己还没想清楚要什么，它生成的代码方向可能偏。

### 修 DDP 死锁：给完整上下文比给症状有用

DDP 死锁是我这个项目最头疼的问题，前后修了 4 次。每次修好一点，换个场景又卡住了。

第一次跟 Claude Code 谈这个问题时，我只说了"多卡训练卡住了"。它给了一个修 schedule 长度不一致的方案。跑了一下，部分修好了，但 `_compute_auc_distributed` 里还是卡。

第二次我换了策略，把完整上下文给它：

```
"训练在验证阶段的 _compute_auc_distributed 函数卡住了。
以下是相关代码：
- train.py 的分布式验证逻辑（附代码片段）
- _compute_auc_distributed 函数的实现
- NCCL 日志显示卡在 all_gather 操作
- 训练用的是 4 卡 GPU，PyTorch 2.6，CUDA 12.4

请分析可能的根因，不要只给表面修复。"
```

这次它分析了三个根因：schedule 不一致、all_gather 异步操作竞争、torch.compile 和 NCCL 的冲突。逐一生成修复，我逐个运行验证。最终发现核心问题是 `torch.cuda.synchronize()` 被之前的优化提交删掉了——那个提交也是 Claude Code 做的（后面会讲这个教训）。

**经验：给完整上下文（错误日志 + 代码片段 + 运行环境）比给症状描述有效得多。** Claude Code 能追踪代码里的状态流转，但你得先给它足够的信息。

### 批量修 Bug：让它审代码比让它修单个 Bug 效率高

3 月 23 号那次，我说了句"检查 train.py 里有没有 bug"，Claude Code 列了 7 个出来：

1. resume 时 epoch 循环从 0 重启（range 起点硬编码）
2. EMA restore 没做 try/finally 保护
3. 分布式 epoch_loss 被 world_size 放大（数学逻辑错）
4. warmup 首批次 lr 错误 + global_step 恢复缺失
5-7. 几个冗余逻辑

这些 bug 如果逐个发现逐个修，至少得一两天。一次性全列出来，一个小时就搞完了。

**经验：与其遇到 bug 再修，不如在代码写完后主动让 Claude Code 做一轮审查。** 它看代码是全局视角——不只看单个函数，而是看状态在训练循环里怎么流转。这种批量发现的能力比人逐行排查快得多。

### 性能优化：它给方向 + 代码 + benchmark，但你要跑着验证

3 月 27 号，我让 Claude Code 分析训练流水线的性能瓶颈。它先是读了所有关键模块的代码，然后给出了一份性能分析报告（就是现在 docs/performance_analysis.md），指出了几个瓶颈点：

- AudioReader 的 sox 调用是外部进程，GIL 阻塞
- FeatureExtractor 在训练循环里跑，GPU 等 CPU
- mixup 时额外磁盘读取

然后它直接生成了优化代码：

- AudioReader 加了 LRU 内存缓存 + 线程局部 librosa → 25x 加速
- FeatureExtractor 用 torch.compile + GPU-only 操作 → 22x 加速
- AudioAugmenter 预生成噪声池 + 线程局部 sox → 减少 I/O 阻塞

还顺手写了 benchmark_performance.py 来验证结果。

这看起来很完美，但问题来了——它生成的 torch.compile 优化和 DDP 的 all_reduce/all_gather 冲突了，导致训练又卡住。这就是下面要讲的教训。

**经验：Claude Code 在性能优化上很擅长找瓶颈方向、写优化代码和 benchmark 脚本，但生成的优化代码必须跑完整的训练流程来验证。** 单测跑得快不代表分布式场景没问题。

### 重构 train.py：渐进式比一步到位靠谱

train.py 在项目中是最复杂的文件（940 行），经历了好几轮简化：

```
03-23 19:01  "简化 train.py 代码" → 去重复模式
03-23 19:15  "简化 train.py 中的重复模式" → 提取公共逻辑
03-23 19:31  "将 scheduler.step 移到训练步骤前面" → 修正训练循环语义
03-24 09:14  "Simplify train.py and restructure configuration" → 配置体系重构
```

我是逐步告诉 Claude Code 要简化什么，而不是一次性说"重构整个 train.py"。每次简化一个方面，跑一下确认没问题，再继续。

**经验：重构复杂模块时，渐进式比一步到位靠谱。** 每次只改一个维度（先去重复代码、再调整训练循环语义、最后重构配置体系），中间每步都验证。

### 写文档：边写代码边更新 CLAUDE.md

这个项目里文档和代码几乎同步更新。每次架构变更，我都让 Claude Code 顺手更新 CLAUDE.md 和 docs/。

举个例子，3 月 17 号那次大重构（把 config.py 移到 utils/、新增 4 个模块），做完代码变更后，我让它更新 CLAUDE.md 的目录结构、数据流说明、特征配置文档。它一次就把所有相关文档都更新了，还加了 Mermaid 流程图。

**经验：把"更新文档"当成代码变更的一部分，而不是事后补。** Claude Code 写文档很高效，但如果你不及时让它更新，后面再补就很痛苦——它不记得之前的对话了。

### CLAUDE.md：项目的约束合约

CLAUDE.md 在这个项目里起了核心作用。它不只是文档，更像是我和 Claude Code 之间的"合约"。每次新对话，Claude Code 会自动读 CLAUDE.md，知道：

- 模型用 Linear projection，不用 Conv1d
- 输入维度是 [B, T, F]，T=157
- Mixup 规则：Cry 样本混合能量差 3-10 dB，非 Cry 只能和非 Cry 混
- Audio cache 用 file mtime 而不是 MD5 hash
- DataLoader persistent_workers=False（避免 DDP 死锁）

这些约束防止 Claude Code 在后续对话中犯同样的错。比如它之前用 Conv1d 做过 patch embedding（Transformer 视觉模型的常见做法），但这对音频分类不适合——CLAUDE.md 里明确写了 NOT Conv1d，后面就再没犯过。

**经验：CLAUDE.md 要写具体的约束，不要只写泛泛的描述。** `Linear projection (NOT Conv1d patch embedding)` 比 `使用线性投影` 有用得多——前者防止了错误选择，后者只是描述了正确选择。

---

## 踩过的坑

### 坑 1：LLM 做的优化又引出了新 Bug

最惨的一次是 DDP 死锁的循环：

- 3 月 28 号，Claude Code 建议"减少 DDP 同步点"，移除了 `torch.cuda.synchronize()`
- 3 月 30 号，训练又卡住了——`all_gather` 的异步操作和后续代码竞争
- 修了 `_compute_auc_distributed` 里的同步问题
- 又发现 torch.compile 和 NCCL 冲突
- 最后回退到 3 月 28 号之前的稳定版本

这教了我一课：**LLM 对性能优化的建议，尤其是涉及并发和分布式场景的，一定要跑完整流程验证。** 它给出的理由（"减少同步开销"）从理论上看是对的，但实际环境中异步操作之间的依赖关系很难静态分析。

### 坑 2：API 版本不匹配

Claude Code 默认用最新版 API。我的环境是 PyTorch 2.6，结果出现了好几处兼容问题：

- `torch.amp.autocast` → PyTorch 2.0+ 需要 `device_type` 参数，旧版不需要
- `GradScaler` API 变了
- `dist.init_process_group` → PyTorch 2.6+ 需要 `device_id`

每次都是跑起来才发现的，然后修一轮 Fix。

**应对：在对话开头就告诉它你用的 PyTorch/Python/CUDA 版本。** 或者把版本信息写进 CLAUDE.md，让它每次对话自动知道。

### 坑 3：并行化不一定比串行好

3 月 17 号让 Claude Code 把低能量样本过滤从串行改成并行（用 ProcessPoolExecutor），结果数据不一致了。3 月 19 号又改回串行。

这提醒我：**对数据一致性敏感的操作，LLM 提议的并行化要格外小心。** 它不会自动判断哪些操作可以并行、哪些不行——你得自己判断。

### 坑 4：它倾向过度设计

Claude Code 有时会为"未来可能的需求"加抽象层。比如有一次它提议给 loss 函数加个 Strategy 模式的抽象基类，方便以后扩展新的 loss 类型。我当时只需要 focal + label_smoothing + combined 三种，直接用工厂函数就够了，加 Strategy 模式纯属多余。

我在 CLAUDE.md 里加了一条来约束这种行为：

```
请使用第一性原理思考。如果动机和目标不清晰，停下来和我讨论。
```

**经验：对 LLM 提出的抽象和设计模式，先问自己"我现在需要吗？"。** YAGNI 原则在与 LLM 协作时更重要——它太容易为你不需要的灵活性加代码了。

---

## 哪些用法值得复制

### 适合让 Claude Code 做的事

1. **从设计文档生成模块代码骨架** — 前提是你自己有清晰的设计
2. **批量代码审查发现 Bug** — 写完代码后主动让它审查一遍
3. **配置体系设计** — dataclass + YAML 配置它写得又快又规范
4. **写技术文档 + Mermaid 流程图** — 几分钟搞定人要写几小时的文档
5. **样板代码** — 训练循环、评估脚本、导出脚本这类模式化代码
6. **性能瓶颈分析和优化方向** — 它识别热点比人快，但落地要自己验证

### 不适合完全交给它的事

1. **分布式训练调试** — NCCL/GPU 同步问题必须实际跑才能诊断
2. **性能优化的最终验证** — 单测快不代表分布式场景没问题
3. **业务规则细节** — Mixup 能量差阈值、采样权重这些要自己定
4. **运行时版本兼容** — 它默认用最新 API，老环境会出问题
5. **重构方向的决策** — 哪些代码值得重构、哪些不要动，人来判断

---

## 工具用法细节

### Claude Code CLI 的日常工作流

我日常的开发流程大概是：

1. 在终端打开 Claude Code，它会自动读项目的 CLAUDE.md
2. 描述我要做什么（比如"给 AudioReader 加内存缓存"、"检查 train.py 的 bug"）
3. Claude Code 读相关文件、生成修改方案、直接编辑代码
4. 我在终端跑代码验证，有问题就继续对话
5. 确认没问题后，让 Claude Code 生成 commit 消息并提交

有几个用得最多的操作：

- **让它读文件**：直接说"读一下 dataset/feature.py"，它会用 Read 工具读取
- **让它改代码**：说"在 audio_reader.py 里加 LRU 缓存"，它会用 Edit 工具精确修改
- **让它写新文件**：说"写个 benchmark_performance.py"，它会用 Write 工具创建
- **让它跑命令**：说"跑一下训练看看"，它会用 Bash 扥执行 python train.py
- **让它提交**：说"提交这次修改"，它会自动生成结构化的 commit 消息

### 提交消息格式

Claude Code 生成的 commit 消息很有结构：

```
Feat: 重构训练代码架构，新增 EMA、调度器、能量特征等多项功能

主要变更：
1. 模块重构：将 config.py 移动到 utils/config.py
2. 新增 model/ema.py：指数移动平均实现
3. 新增 model/scheduler.py：Warmup + Cosine Decay
...

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
```

这种格式让 git log 很易读，也自动标记了 LLM 参与的提交。

### CLAUDE.md 怎么写最有效

实践下来，CLAUDE.md 里最有用的部分不是泛泛的项目描述，而是**具体的约束和 NOT 规则**：

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

这些 NOT 规则和具体数值直接防止了 Claude Code 在后续对话中犯同样的错。相比之下，"使用线性投影"、"避免死锁"这种软描述基本没用。

---

## 一些数字

| 指标 | 纯手动（估计） | 和 Claude Code 协作（实际） |
|------|-------------|------------------------|
| 从 0 到跑通训练 | ~2-3 周 | 2 天（03-02 ~ 03-11） |
| 架构重构 6 个新模块 | ~3-5 天 | 1 天（03-17） |
| 找出并修 7 个 Bug | ~1-2 天 | ~1 小时 |
| 性能优化 25x 加速 | ~3-5 天 | 1 天（03-27） |
| 写 7 个技术文档 | ~2-3 天 | ~1 天 |

不是所有环节都加速——DDP 死锁调试花的时间反而比手动可能更多，因为 Claude Code 的优化又引出了新问题。但整体上，从设计到可运行代码的转化速度确实快了很多。

---

## 总结

用 Claude Code 做这个项目的核心经验就是一句话：**你负责方向和判断，它负责转化和执行。**

具体来说：

- 你得有设计文档，它才能把设计变成代码。没有设计，它生成的代码方向可能偏。
- 你得给完整上下文，它才能有效调试。只说症状，它只能猜。
- 你得跑着验证，它才能安全优化。单测没问题不代表分布式没问题。
- 你得把约束写进 CLAUDE.md，它才不会重复犯同样的错。NOT 规则比描述有用。
- 你得判断重构方向，它才能正确执行。不要让它替你决定重构什么。

CLAUDE.md 是这个协作模式里最关键的桥梁——它把你每次验证过的决策凝固成规则，让下一次对话里的 Claude Code 自动遵循。没有它，每次新对话都是从零开始，之前的教训全丢了。