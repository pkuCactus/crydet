# CryTransformer 打桩测试 (Stub Testing)

本目录包含用于端到端测试的打桩数据生成器和测试脚本，无需真实音频数据即可验证 train、evaluate、inference、export 等完整流程。

## 文件说明

| 文件 | 说明 |
|------|------|
| `stub_data.py` | 模拟数据生成器，生成合成音频和测试配置 |
| `test_end_to_end.py` | 端到端集成测试，覆盖完整工作流程 |
| `test_feature.py` | 特征提取单元测试 |
| `run_tests.py` | 测试运行器 |

## 快速开始

### 1. 快速冒烟测试 (推荐)

运行快速冒烟测试，验证所有组件基本功能：

```bash
python llt/run_tests.py --smoke
```

或直接使用：

```bash
python llt/test_end_to_end.py --smoke
```

### 2. 使用指定 Conda 环境

如果需要在特定的 conda 环境中运行测试（推荐）：

```bash
# 使用名为 "crydet" 的 conda 环境
python llt/run_tests.py --smoke --conda-env crydet

# 完整测试套件使用指定环境
python llt/run_tests.py --conda-env crydet

# 直接运行测试文件
python llt/test_end_to_end.py --smoke --conda-env crydet
```

### 3. 完整测试套件

运行所有测试（包括特征提取单元测试和端到端测试）：

```bash
python llt/run_tests.py
```

带详细输出：

```bash
python llt/run_tests.py -v
```

### 4. 单独运行特定测试

```bash
# 仅运行特征提取测试
python llt/run_tests.py --file test_feature

# 仅运行端到端测试
python llt/run_tests.py --file test_end_to_end
```

## Conda 环境设置

### 创建测试环境

```bash
# 创建新环境
conda create -n crydet python=3.10

# 激活环境
conda activate crydet

# 安装依赖
pip install -r requirements.txt
```

### 自动查找 Conda

测试脚本会自动查找 conda 可执行文件：
1. 首先检查 `CONDA_EXE` 环境变量
2. 然后使用 `which conda` / `where conda`
3. 最后检查常见安装路径 (`~/miniconda3`, `~/anaconda3`, `/opt/conda` 等)

如果找不到 conda，将回退到当前 Python 环境运行。

## 模拟数据生成器 (StubDataManager)

`StubDataManager` 是一个上下文管理器，用于自动生成测试所需的全部数据：

```python
from llt.stub_data import StubDataManager

# 使用上下文管理器自动清理
with StubDataManager() as manager:
    # 创建训练/验证/测试数据集
    splits = manager.create_train_val_test_split(
        train_cry=20,    # 训练集哭声样本数
        train_other=20,  # 训练集非哭声样本数
        val_cry=5,       # 验证集哭声样本数
        val_other=5,     # 验证集非哭声样本数
        test_cry=5,      # 测试集哭声样本数
        test_other=5     # 测试集非哭声样本数
    )

    # 获取音频列表 JSON 路径
    train_list = splits['train']  # e.g., /tmp/xxx/audio_list_train.json
    val_list = splits['val']
    test_list = splits['test']

    # 创建测试配置（使用小模型和快速设置）
    config_path = manager.create_minimal_config()

    # 使用这些路径运行训练、评估...
    # 退出上下文后自动清理临时文件
```

### 生成模拟检查点

```python
from llt.stub_data import create_mock_checkpoint

# 创建模拟模型检查点（用于测试 inference/evaluate）
checkpoint_path = create_mock_checkpoint(
    save_path="path/to/model.pt",
    config_path="path/to/config.yaml"
)
```

## 端到端测试覆盖

`test_end_to_end.py` 包含以下测试用例：

### 完整流程测试
- `test_01_train_from_scratch` - 从头开始训练
- `test_02_create_mock_checkpoint` - 创建模拟检查点
- `test_03_evaluate_model` - 模型评估
- `test_04_inference_single_file` - 单文件推理
- `test_05_inference_batch` - 批量推理
- `test_06_export_pytorch` - 导出 PyTorch 格式
- `test_07_export_size_report` - 生成模型大小报告

### 组件测试
- `test_dataset_loading` - 数据集加载
- `test_feature_extraction` - 特征提取
- `test_model_forward` - 模型前向传播

### 模拟数据测试
- `test_generate_wav_cry` - 生成哭声音频
- `test_generate_wav_noise` - 生成噪声音频
- `test_generate_dataset` - 生成完整数据集
- `test_stub_data_manager` - 数据管理器
- `test_config_creation` - 配置生成

## 测试配置

`create_minimal_config()` 生成的测试配置特点：

- **小模型**: d_model=64, n_layers=2（快速训练和推理）
- **少数据**: 默认 2 个 epoch，batch_size=4
- **无增强**: 关闭数据增强加速测试
- **CPU 运行**: device=cpu（不依赖 GPU）
- **无 EMA**: use_ema=False 简化流程
- **短音频**: slice_len=3.0 秒

## 命令行示例

### 测试训练流程

```bash
python -c "
from llt.stub_data import StubDataManager
import subprocess

with StubDataManager() as manager:
    splits = manager.create_train_val_test_split()
    config = manager.create_minimal_config()

    subprocess.run([
        'python', 'train.py',
        '--config', config,
        '--train_list', splits['train'],
        '--val_list', splits['val'],
        '--epochs', '2'
    ])
"
```

### 测试评估流程

```bash
python -c "
from llt.stub_data import StubDataManager, create_mock_checkpoint
import subprocess

with StubDataManager() as manager:
    splits = manager.create_train_val_test_split()
    config = manager.create_minimal_config()

    # 创建模拟检查点
    checkpoint = str(manager.base_dir / 'model.pt')
    create_mock_checkpoint(checkpoint, config)

    # 运行评估
    subprocess.run([
        'python', 'evaluate.py',
        '--checkpoint', checkpoint,
        '--test_list', splits['test']
    ])
"
```

## 集成到 CI/CD

建议在持续集成中使用快速冒烟测试：

```yaml
# .github/workflows/test.yml 示例
test:
  steps:
    - uses: actions/checkout@v2
    - name: Install dependencies
      run: pip install -r requirements.txt
    - name: Run smoke tests
      run: python llt/run_tests.py --smoke
```

## 注意事项

1. **音频缓存**: 测试会创建临时目录存放生成的音频文件，使用 `StubDataManager` 上下文管理器可确保自动清理
2. **随机种子**: 模拟数据生成器使用固定种子 (42) 确保可重复性
3. **超时设置**: 端到端测试设置了超时（训练 5 分钟，评估 2 分钟等）
4. **依赖**: 需要安装完整依赖 `pip install -r requirements.txt`
