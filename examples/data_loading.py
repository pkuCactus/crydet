"""
数据读取示例
演示如何使用 AudioReader、CryDataset 和 CrySampler
"""

import sys
sys.path.insert(0, '..')

from pathlib import Path
from dataset.audio_reader import AudioReader
from dataset.dataset import CryDataset
from dataset.sampler import CrySampler
from torch.utils.data import DataLoader
from config import DatasetConfig, load_config


def example_audio_reader():
    """示例1: 使用 AudioReader 读取音频文件"""
    print("=" * 60)
    print("示例1: AudioReader 基本使用")
    print("=" * 60)

    # 创建 AudioReader
    reader = AudioReader(
        target_sr=16000,
        cache_dir='./audio_cache',
        use_cache=True,
        force_mono=True
    )
    print(f"AudioReader: {reader}")

    # 假设有一个音频文件
    audio_file = 'data/audio/cry/sample.wav'

    if Path(audio_file).exists():
        # 1. 加载整个音频
        waveform, sr = reader.load(audio_file)
        print(f"完整加载: shape={waveform.shape}, sr={sr}")

        # 2. 按采样点加载部分音频
        waveform_part, _ = reader.load(audio_file, start=16000, stop=32000)
        print(f"部分加载 (1-2秒): shape={waveform_part.shape}")

        # 3. 按时间加载部分音频
        waveform_time, _ = reader.load_by_time(audio_file, start_time=5.0, end_time=10.0)
        print(f"时间加载 (5-10秒): shape={waveform_time.shape}")

        # 4. 查看缓存信息
        cache_info = reader.get_cache_info()
        print(f"缓存信息: {cache_info}")
    else:
        print(f"音频文件不存在: {audio_file}")


def example_dataset():
    """示例2: 使用 CryDataset 加载数据集"""
    print("\n" + "=" * 60)
    print("示例2: CryDataset 数据集使用")
    print("=" * 60)

    # 配置
    config = DatasetConfig(
        sample_rate=16000,
        duration=10.0,
        stride=1.0,
        cry_rate=0.5,
        cache_dir='./audio_cache',
        use_cache=True
    )

    # 数据目录字典
    # 格式: {label: [directory1, directory2, ...]}
    # non-cry 类别会跳过第一个目录（用于数据平衡）
    data_dict = {
        'cry': ['data/audio/cry'],
        'non_cry': ['data/audio/non_cry_backup', 'data/audio/non_cry'],
    }

    # 创建数据集
    dataset = CryDataset(data_dict, config)

    print(f"数据集长度: {len(dataset)}")
    print(f"类别: {list(dataset.file_schedule_dict.keys())}")
    print(f"各类别样本数: {dataset.num_samples}")

    # 获取单个样本
    if len(dataset) > 0:
        # 索引格式: (label, file_idx)
        index = ('cry', 0)
        waveform, label = dataset[index]
        print(f"样本 shape: {waveform.shape}, label: {label}")


def example_dataloader():
    """示例3: 使用 DataLoader 和 CrySampler"""
    print("\n" + "=" * 60)
    print("示例3: DataLoader + CrySampler")
    print("=" * 60)

    # 加载配置
    config = load_config('configs/default.yaml')

    # 数据目录
    data_dict = {
        'cry': ['data/audio/cry'],
        'non_cry': ['data/audio/non_cry'],
    }

    # 创建数据集
    dataset = CryDataset(data_dict, config.dataset)

    # 创建采样器 (控制 cry/non_cry 比例)
    sampler = CrySampler(
        data_source=dataset,
        cry_rate=config.dataset.cry_rate,
        shuffle=True
    )

    # 创建 DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        sampler=sampler,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory
    )

    print(f"DataLoader batches: {len(dataloader)}")

    # 遍历数据
    for batch_idx, (waveforms, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx}: waveforms shape={waveforms.shape}, labels={labels[:5]}...")
        if batch_idx >= 2:
            break


def example_from_config():
    """示例4: 从配置文件加载"""
    print("\n" + "=" * 60)
    print("示例4: 从配置文件加载")
    print("=" * 60)

    # 加载配置
    config = load_config('configs/default.yaml')

    print(f"特征配置:")
    print(f"  - feature_type: {config.feature.feature_type}")
    print(f"  - n_mels: {config.feature.n_mels}")
    print(f"  - n_fft: {config.feature.n_fft}")

    print(f"\n数据集配置:")
    print(f"  - sample_rate: {config.dataset.sample_rate}")
    print(f"  - duration: {config.dataset.duration}")
    print(f"  - cry_rate: {config.dataset.cry_rate}")
    print(f"  - cache_dir: {config.dataset.cache_dir}")

    print(f"\n训练配置:")
    print(f"  - batch_size: {config.training.batch_size}")
    print(f"  - learning_rate: {config.training.learning_rate}")
    print(f"  - num_epochs: {config.training.num_epochs}")


def example_cache_management():
    """示例5: 缓存管理"""
    print("\n" + "=" * 60)
    print("示例5: 缓存管理")
    print("=" * 60)

    reader = AudioReader(
        target_sr=16000,
        cache_dir='./audio_cache',
        use_cache=True
    )

    # 查看缓存信息
    info = reader.get_cache_info()
    print(f"缓存状态:")
    print(f"  - 启用: {info['enabled']}")
    print(f"  - 目录: {info['cache_dir']}")
    print(f"  - 文件数: {info['file_count']}")
    print(f"  - 大小: {info['total_size_mb']} MB")

    # 清除缓存
    # deleted = reader.clear_cache()
    # print(f"已删除 {deleted} 个缓存文件")


if __name__ == '__main__':
    # 运行所有示例
    example_audio_reader()
    example_dataset()
    example_dataloader()
    example_from_config()
    example_cache_management()
