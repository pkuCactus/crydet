"""
数据读取示例
演示如何从 audio_list/sample.json 和 configs/default.yaml 构建 DataLoader
"""

import sys
sys.path.insert(0, '..')

import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from dataset.dataset import CryDataset
from dataset.sampler import CrySampler
from config import load_config


def load_audio_list(json_path: str) -> dict:
    """
    从 JSON 文件加载音频目录列表

    Args:
        json_path: JSON 文件路径

    Returns:
        数据目录字典 {label: [optional[int], dir1, dir2, ...]}
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data


def create_dataloader_from_config(
    audio_list_path: str,
    config_path: str
) -> DataLoader:
    """
    从配置文件和数据列表创建 DataLoader

    Args:
        audio_list_path: 音频列表 JSON 文件路径
        config_path: 配置文件路径

    Returns:
        DataLoader 实例
    """
    # 加载配置
    config = load_config(config_path)

    # 加载音频目录列表
    data_dict = load_audio_list(audio_list_path)

    # 创建数据集
    aug_config = config.augmentation if config.training.use_augmentation else None
    dataset = CryDataset(data_dict, config.dataset, aug_config=aug_config)

    # 创建采样器
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

    return dataloader, dataset, config


def main():
    """主函数：演示完整的数据加载流程"""
    print("=" * 70)
    print("婴儿哭声检测数据加载示例")
    print("=" * 70)

    # 路径配置
    audio_list_path = 'audio_list/sample.json'
    config_path = 'configs/default.yaml'

    # 检查文件是否存在
    if not Path(audio_list_path).exists():
        print(f"错误: 音频列表文件不存在: {audio_list_path}")
        print("请创建 audio_list/sample.json 文件")
        return

    if not Path(config_path).exists():
        print(f"错误: 配置文件不存在: {config_path}")
        return

    # 1. 加载配置
    print("\n[1] 加载配置文件")
    config = load_config(config_path)
    print(f"    配置文件: {config_path}")
    print(f"    采样率: {config.dataset.sample_rate} Hz")
    print(f"    片段时长: {config.dataset.slice_len} 秒")
    print(f"    Cry 比例: {config.dataset.cry_rate}")
    print(f"    批大小: {config.training.batch_size}")

    # 2. 加载音频目录列表
    print("\n[2] 加载音频目录列表")
    data_dict = load_audio_list(audio_list_path)
    print(f"    音频列表: {audio_list_path}")
    for label, dirs in data_dict.items():
        print(f"    - {label}: {dirs}")

    # 3. 创建数据集
    print("\n[3] 创建数据集")
    aug_config = config.augmentation if config.training.use_augmentation else None
    dataset = CryDataset(data_dict, config.dataset, aug_config=aug_config)
    print(f"    数据集长度: {len(dataset)}")
    print(f"    类别样本数: {dataset.num_samples}")
    print(f"    数据增强: {'启用' if aug_config else '禁用'}")

    # 4. 创建采样器和 DataLoader
    print("\n[4] 创建 DataLoader")
    sampler = CrySampler(
        data_source=dataset,
        cry_rate=config.dataset.cry_rate,
        shuffle=True
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        sampler=sampler,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory
    )
    print(f"    批次数: {len(dataloader)}")
    print(f"    采样器: CrySampler(cry_rate={config.dataset.cry_rate})")

    # 5. 遍历数据
    print("\n[5] 遍历数据批次")
    for batch_idx, (waveforms, labels) in enumerate(dataloader):
        print(f"\n    Batch {batch_idx + 1}:")
        print(f"      - 波形形状: {waveforms.shape}")
        print(f"      - 标签: {labels[:5]}... (共 {len(labels)} 个)")

        # 统计标签分布
        unique_labels = torch.unique(labels)
        for label in unique_labels:
            count = (labels == label).sum().item()
            print(f"      - {label}: {count} 个样本")

        # 只展示前3个批次
        if batch_idx >= 2:
            print("\n    ... (仅展示前3个批次)")
            break

    # 6. 缓存信息
    print("\n[6] 音频缓存信息")
    cache_info = dataset.audio_reader.get_cache_info()
    print(f"    缓存启用: {cache_info['enabled']}")
    print(f"    缓存目录: {cache_info['cache_dir']}")
    print(f"    缓存文件数: {cache_info['file_count']}")
    print(f"    缓存大小: {cache_info['total_size_mb']} MB")

    print("\n" + "=" * 70)
    print("数据加载示例完成")
    print("=" * 70)


def example_single_sample():
    """示例：获取单个样本"""
    audio_list_path = 'audio_list/sample.json'
    config_path = 'configs/default.yaml'

    if not Path(audio_list_path).exists() or not Path(config_path).exists():
        print("配置文件不存在，跳过单个样本示例")
        return

    config = load_config(config_path)
    data_dict = load_audio_list(audio_list_path)
    aug_config = config.augmentation if config.training.use_augmentation else None
    dataset = CryDataset(data_dict, config.dataset, aug_config=aug_config)

    if len(dataset) > 0:
        print("\n获取单个样本:")
        # 获取 cry 类别的第一个样本
        if 'cry' in dataset.file_schedule_dict and len(dataset.file_schedule_dict['cry']) > 0:
            waveform, label = dataset[('cry', 0)]
            print(f"  波形形状: {waveform.shape}")
            print(f"  波形时长: {len(waveform) / config.dataset.sample_rate:.2f} 秒")
            print(f"  标签: {label}")


if __name__ == '__main__':
    main()
    example_single_sample()
