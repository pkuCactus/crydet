"""
数据读取示例
演示如何从 audio_list/sample.json 和 configs/default.yaml 构建 DataLoader
"""

import sys
sys.path.insert(0, '..')

import json
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import soundfile as sf

from dataset.dataset import CryDataset
from dataset.sampler import CrySampler
from config import load_config


def save_samples_to_temp(dataset, output_dir: str = 'temp', num_samples: int = 100, sample_rate: int = 16000):
    """
    从数据集中读取样本并保存为 WAV 文件

    Args:
        dataset: CryDataset 实例
        output_dir: 输出目录
        num_samples: 要保存的样本数量
        sample_rate: 采样率
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 统计每个标签的计数
    label_counts = {}

    print(f"\n保存 {num_samples} 个样本到 {output_dir}/")

    saved_count = 0
    # 遍历所有标签
    for label in dataset.file_schedule_dict:
        schedules = dataset.file_schedule_dict[label]
        for file_idx in range(len(schedules)):
            if saved_count >= num_samples:
                break

            # 获取样本
            waveform, ret_label = dataset[(label, file_idx)]

            # 更新标签计数
            if ret_label not in label_counts:
                label_counts[ret_label] = 0
            label_counts[ret_label] += 1

            # 生成文件名: 标签_idx.wav
            filename = f"{ret_label}_{label_counts[ret_label]}.wav"
            filepath = output_path / filename

            # 保存为 WAV 文件
            sf.write(str(filepath), waveform, sample_rate)
            saved_count += 1

            if saved_count % 10 == 0:
                print(f"  已保存 {saved_count}/{num_samples} 个样本")

        if saved_count >= num_samples:
            break

    print(f"\n完成! 共保存 {saved_count} 个样本")
    print(f"标签分布: {label_counts}")

    return label_counts


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
    tm = time.time()
    for batch_idx, (waveforms, labels) in enumerate(dataloader):
        print(f"\n    Batch {batch_idx + 1}:")
        print(f"      - batch读取时间: {time.time() - tm}s")
        print(f"      - 波形形状: {waveforms.shape}")
        print(f"      - 标签: {labels[:5]}... (共 {len(labels)} 个)")
        tm = time.time()

        # 只展示前3个批次
        if batch_idx >= 5:
            print("\n    ... (仅展示前5个批次)")
            break

    # 6. 缓存信息
    print("\n[6] 音频缓存信息")
    cache_info = dataset.audio_reader.get_cache_info()
    print(f"    缓存启用: {cache_info['enabled']}")
    print(f"    缓存目录: {cache_info['cache_dir']}")
    print(f"    缓存文件数: {cache_info['file_count']}")
    print(f"    缓存大小: {cache_info['total_size_mb']} MB")

    # 7. 保存样本到 temp 目录
    print("\n[7] 保存样本到 temp 目录")
    save_samples_to_temp(
        dataset=dataset,
        output_dir='temp',
        num_samples=100,
        sample_rate=config.dataset.sample_rate
    )

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
