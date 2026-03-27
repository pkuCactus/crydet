#!/usr/bin/env python
"""
演示脚本：展示新配置系统下的完整数据流程

包括：
1. 加载配置
2. 创建数据集
3. 特征提取
4. 数据增强
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch

from utils.config import load_config, FeatureConfig, AugmentationConfig, DatasetConfig
from dataset.feature import FeatureExtractor
from dataset.augmentation import AudioAugmenter
from dataset.dataset import CryDataset
from llt.stub_data import StubDataManager


def demo_feature_extraction():
    """演示特征提取"""
    print("\n" + "="*60)
    print("演示1: 特征提取 (Feature Extraction)")
    print("="*60)

    # 使用默认配置
    config = FeatureConfig(
        feature_type=3,  # FBank + DB
        n_mels=32,
        n_fft=1024,
        hop_length=500,
        use_fbank_norm=True,
        use_time_delta=False,
    )

    print(f"配置:")
    print(f"  - 特征类型: {config.feature_type} (FBank+DB)")
    print(f"  - Mel滤波器数量: {config.n_mels}")
    print(f"  - FFT大小: {config.n_fft}")
    print(f"  - 特征维度: {config.feature_dim}")

    # 创建特征提取器
    extractor = FeatureExtractor(config, sr=16000)

    # 生成测试音频 (5秒 @ 16kHz)
    waveform = torch.randn(1, 80000) * 0.1

    # 提取特征
    features = extractor(waveform)

    print(f"\n输入: {waveform.shape} -> 输出: {features.shape}")
    print(f"时间帧数: {features.shape[1]}")
    print(f"特征维度: {features.shape[2]}")

    return features


def demo_augmentation():
    """演示数据增强"""
    print("\n" + "="*60)
    print("演示2: 数据增强 (Data Augmentation)")
    print("="*60)

    # 创建增强配置
    config = AugmentationConfig(
        cry_aug_prob=0.9,
        other_aug_prob=0.6,
        pitch_prob=0.5,
        reverb_prob=0.8,
        gain_prob=0.9,
    )

    print(f"配置:")
    print(f"  - 哭声增强概率: {config.cry_aug_prob}")
    print(f"  - 非哭声增强概率: {config.other_aug_prob}")
    print(f"  - 音调变化概率: {config.pitch_prob}")
    print(f"  - 混响概率: {config.reverb_prob}")
    print(f"  - 增益调整概率: {config.gain_prob}")

    # 创建增强器
    augmenter = AudioAugmenter(config, sample_rate=16000)

    # 生成测试音频
    waveform = np.random.randn(80000).astype(np.float32) * 0.1

    # 应用增强
    augmented = augmenter(waveform, label='cry')

    print(f"\n输入形状: {waveform.shape}")
    print(f"输出形状: {augmented.shape}")
    print(f"能量变化: {np.mean(waveform**2):.6f} -> {np.mean(augmented**2):.6f}")

    return augmented


def demo_dataset():
    """演示数据集使用"""
    print("\n" + "="*60)
    print("演示3: 数据集 (CryDataset)")
    print("="*60)

    with StubDataManager() as manager:
        # 创建数据集
        splits = manager.create_train_val_test_split(
            train_cry=4, train_other=4,
            val_cry=2, val_other=2,
            test_cry=2, test_other=2
        )
        config_path = manager.create_minimal_config()

        # 加载配置
        config = load_config(config_path)

        print(f"数据集配置:")
        print(f"  - 采样率: {config.dataset.sample_rate}")
        print(f"  - 切片长度: {config.dataset.slice_len}s")
        print(f"  - 步长: {config.dataset.stride}s")
        print(f"  - 哭声比例: {config.dataset.cry_rate}")

        # 加载测试列表
        import json
        with open(splits['test']) as f:
            test_dict = json.load(f)

        print(f"\n测试数据集:")
        print(f"  - 哭声目录: {test_dict['cry']}")
        print(f"  - 其他目录: {test_dict['other'][1]}")

        # 创建数据集
        dataset = CryDataset(
            test_dict,
            config.dataset,
            aug_config=None
        )
        dataset.build_schedule(shuffle=False)

        print(f"\n数据集构建完成:")
        print(f"  - 哭声样本数: {len(dataset.file_schedule_dict.get('cry', []))}")
        print(f"  - 非哭声样本数: {len(dataset.file_schedule_dict.get('other', []))}")
        print(f"  - 每轮迭代数: {len(dataset)}")

        # 获取一个样本
        if len(dataset) > 0:
            from dataset.sampler import SequentialCrySampler
            sampler = SequentialCrySampler(dataset)
            sample_idx = list(sampler)[0]
            waveform, label = dataset[sample_idx]
            print(f"\n样本示例:")
            print(f"  - 标签: {label}")
            print(f"  - 波形形状: {waveform.shape}")
            print(f"  - 波形时长: {len(waveform) / config.dataset.sample_rate:.2f}s")

        return dataset


def demo_full_pipeline():
    """演示完整流程"""
    print("\n" + "="*60)
    print("演示4: 完整流程 (Full Pipeline)")
    print("="*60)

    with StubDataManager() as manager:
        # 创建数据
        splits = manager.create_train_val_test_split(train_cry=2, train_other=2)
        config_path = manager.create_minimal_config()
        config = load_config(config_path)

        # 1. 创建特征提取器
        extractor = FeatureExtractor(config.feature, sr=config.dataset.sample_rate)
        print(f"✓ 特征提取器创建完成")
        print(f"  特征维度: {config.feature.feature_dim}")

        # 2. 创建数据增强器
        augmenter = AudioAugmenter(config.augmentation, sample_rate=config.dataset.sample_rate)
        print(f"✓ 数据增强器创建完成")

        # 3. 创建数据集
        import json
        with open(splits['train']) as f:
            data_dict = json.load(f)

        dataset = CryDataset(
            data_dict,
            config.dataset,
            aug_config=config.augmentation  # 使用增强
        )
        dataset.build_schedule(shuffle=False)
        print(f"✓ 数据集创建完成")

        # 4. 获取样本
        from dataset.sampler import SequentialCrySampler
        sampler = SequentialCrySampler(dataset)
        indices = list(sampler)[:2]  # 获取2个样本

        batch_features = []
        batch_labels = []

        for idx in indices:
            waveform, label = dataset[idx]

            # 5. 特征提取
            waveform_tensor = torch.from_numpy(waveform).float().unsqueeze(0)
            features = extractor(waveform_tensor)

            batch_features.append(features)
            batch_labels.append(label)

        # 6. 堆叠批次
        batch = torch.cat(batch_features, dim=0)
        print(f"✓ 批次特征提取完成")
        print(f"  批次形状: {batch.shape}")
        print(f"  标签: {batch_labels}")

        print(f"\n完整流程演示成功!")

        return batch


def main():
    """主函数"""
    print("\n" + "="*60)
    print("CryDet 新配置系统演示")
    print("="*60)
    print("\n这个脚本演示了如何在新配置系统下使用:")
    print("  1. FeatureExtractor - 特征提取")
    print("  2. AudioAugmenter - 数据增强")
    print("  3. CryDataset - 数据集")
    print("  4. 完整流程 - 从音频到特征")

    try:
        demo_feature_extraction()
        demo_augmentation()
        demo_dataset()
        demo_full_pipeline()

        print("\n" + "="*60)
        print("所有演示成功完成!")
        print("="*60)

    except Exception as e:
        print(f"\n✗ 演示失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
