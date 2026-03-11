"""
特征提取示例
演示如何读取音频文件并提取 FFT/FBank/MFCC/Energy 特征
参考: docs/feature_extraction_flow.md
"""

import sys
sys.path.insert(0, '..')

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import soundfile as sf

from dataset.feature import FeatureExtractor
from dataset.audio_reader import AudioReader
from config import load_config, FeatureConfig


def extract_features_from_file(
    audio_path: str,
    config: FeatureConfig,
    sample_rate: int = 16000
):
    """
    从单个音频文件提取特征

    Args:
        audio_path: 音频文件路径
        config: 特征配置
        sample_rate: 目标采样率

    Returns:
        features: 字典，包含 fft, fbank, mfcc, db
        waveform: 原始波形
        sr: 采样率
    """
    # 创建音频读取器
    audio_reader = AudioReader(
        target_sr=sample_rate,
        cache_dir='./audio_cache',
        force_mono=True
    )

    # 读取音频
    waveform, sr = audio_reader.load(audio_path)
    print(f"音频波形形状: {waveform.shape}")
    print(f"采样率: {sr} Hz")
    print(f"时长: {len(waveform) / sr:.2f} 秒")

    # 创建特征提取器
    extractor = FeatureExtractor(config)

    # 提取特征
    features = extractor.extract(waveform, sr)

    print(f"\n提取的特征:")
    for name, feat in features.items():
        print(f"  {name}: shape={feat.shape}, min={feat.min():.3f}, max={feat.max():.3f}")

    return features, waveform, sr


def visualize_features(
    features: dict,
    save_path: str = None
):
    """
    可视化所有特征

    Args:
        features: 特征字典
        config: 特征配置
        save_path: 保存路径 (可选)
    """
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    feature_names = ['fft', 'fbank', 'mfcc', 'db']
    titles = [
        f"FFT Magnitude Spectrum ({features['fft'].shape[0]} bins)",
        f"FBank (Log Mel Spectrum, {features['fbank'].shape[0]} mels)",
        f"MFCC ({features['mfcc'].shape[0]} coefficients)",
        f"Energy (dB) - [Average, Weighted]"
    ]

    for ax, name, title in zip(axes, feature_names, titles):
        feat = features[name]
        im = ax.imshow(feat, aspect='auto', origin='lower', cmap='viridis')
        ax.set_title(title)
        ax.set_ylabel('Feature Dim')
        ax.set_xlabel('Time Frames')
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n特征可视化已保存到: {save_path}")
    else:
        plt.show()

    plt.close()


def compare_configs(audio_path: str, sample_rate: int = 16000):
    """
    比较不同配置的特征提取

    Args:
        audio_path: 音频文件路径
        sample_rate: 采样率
    """
    print("\n" + "=" * 60)
    print("比较不同特征配置")
    print("=" * 60)

    # 读取音频
    audio_reader = AudioReader(target_sr=sample_rate, force_mono=True)
    waveform, sr = audio_reader.load(audio_path)

    configs = [
        ("FBank (32 mel, norm)", FeatureConfig(n_mels=32, use_fbank_norm=True)),
        ("FBank (64 mel, norm)", FeatureConfig(n_mels=64, use_fbank_norm=True)),
        ("FBank (no norm)", FeatureConfig(n_mels=32, use_fbank_norm=False)),
        ("MFCC (16)", FeatureConfig(feature_type='mfcc', n_mfcc=16)),
        ("MFCC (32)", FeatureConfig(feature_type='mfcc', n_mfcc=32)),
    ]

    fig, axes = plt.subplots(len(configs), 1, figsize=(14, 3 * len(configs)))

    audio_reader = AudioReader(target_sr=sample_rate, force_mono=True)
    waveform, sr = audio_reader.load(audio_path)

    for i, (name, cfg) in enumerate(configs):
        extractor = FeatureExtractor(cfg)
        features = extractor.extract(waveform, sr)

        # 选择要显示的特征
        if cfg.feature_type == 'mfcc':
            feat = features['mfcc']
        else:
            feat = features['fbank']

        ax = axes[i] if len(configs) > 1 else axes
        im = ax.imshow(feat, aspect='auto', origin='lower', cmap='viridis')
        ax.set_title(f'{name}\nShape: {feat.shape}')
        ax.set_ylabel('Feature Dim')
        ax.set_xlabel('Time Frames')

    plt.tight_layout()
    plt.savefig('feature_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n比较图已保存到: feature_comparison.png")
    plt.close()


def demo_delta_features(audio_path: str, sample_rate: int = 16000):
    """
    演示 Delta 特征提取 (用于Transformer模型输入)

    Args:
        audio_path: 音频文件路径
        sample_rate: 采样率
    """
    print("\n" + "=" * 60)
    print("Delta 特征提取演示 (Transformer输入格式)")
    print("=" * 60)

    # 读取音频
    audio_reader = AudioReader(target_sr=sample_rate, force_mono=True)
    waveform, sr = audio_reader.load(audio_path)

    configs = [
        ("Base FBank (64-dim)", FeatureConfig(n_mels=64, use_delta=False, use_freq_delta=False)),
        ("+ Time Delta (128-dim)", FeatureConfig(n_mels=64, use_delta=True, use_freq_delta=False)),
        ("+ Time + Freq Delta (192-dim)", FeatureConfig(n_mels=64, use_delta=True, use_freq_delta=True)),
    ]

    print(f"\n{'Configuration':<35} {'Shape':<15} {'Feature Dim'}")
    print("-" * 65)

    for name, cfg in configs:
        extractor = FeatureExtractor(cfg)
        features = extractor.extract_with_deltas(waveform, sr)
        print(f"{name:<35} {str(features.shape):<15} {features.shape[1]}")

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for i, (name, cfg) in enumerate(configs):
        extractor = FeatureExtractor(cfg)
        feats = extractor.extract_with_deltas(waveform, sr)

        ax = axes[i]
        im = ax.imshow(feats.T, aspect='auto', origin='lower', cmap='viridis')
        ax.set_title(f'{name}\nShape: {feats.shape}')
        ax.set_xlabel('Time Frames')
        ax.set_ylabel('Feature Dim' if i == 0 else '')
        plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig('feature_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nDelta特征对比图已保存到: feature_comparison.png")
    plt.close()

    print("\n注意: Transformer模型期望输入格式为 [B, T, F]")
    print("  - B: batch size")
    print("  - T: 时间帧数 (如 157 对应5秒音频)")
    print("  - F: 特征维度 (64/128/192)")


def main():
    """主函数：演示特征提取流程"""
    print("=" * 70)
    print("婴儿哭声检测 - 特征提取示例")
    print("=" * 70)

    # 配置
    config_path = 'configs/default.yaml'
    audio_path = None

    # 查找测试音频
    test_dirs = ['temp/cry', 'temp/other', 'audio_data']
    for test_dir in test_dirs:
        test_path = Path(test_dir)
        if test_path.exists():
            for ext in ['.wav', '.mp3', '.flac']:
                files = list(test_path.glob(f'*{ext}'))
                if files:
                    audio_path = str(files[0])
                    break
        if audio_path:
            break

    if not audio_path:
        # 生成测试音频
        print("\n[1] 生成测试音频")
        audio_path = '_test_audio.wav'
        sr = 16000
        duration = 5.0
        t = np.linspace(0, duration, int(sr * duration))
        # 生成包含多个频率成分的测试信号
        test_signal = (
            0.3 * np.sin(2 * np.pi * 440 * t) +  # 440 Hz
            0.2 * np.sin(2 * np.pi * 880 * t) +  # 880 Hz
            0.1 * np.sin(2 * np.pi * 1760 * t)  # 1760 Hz
        )
        # 添加一些噪声
        test_signal += 0.05 * np.random.randn(len(t))
        sf.write(audio_path, test_signal.astype(np.float32), sr)
        print(f"    已生成测试音频: {audio_path}")
        print(f"    时长: {duration} 秒")
        print(f"    采样率: {sr} Hz")
    else:
        print(f"\n[1] 使用测试音频: {audio_path}")

    # 加载或使用默认配置
    if Path(config_path).exists():
        config = load_config(config_path)
        feature_config = config.feature
    else:
        feature_config = FeatureConfig()

    print(f"\n[2] 特征配置:")
    print(f"    feature_type: {feature_config.feature_type}")
    print(f"    n_fft: {feature_config.n_fft}")
    print(f"    hop_length: {feature_config.hop_length}")
    print(f"    n_mels: {feature_config.n_mels}")
    print(f"    n_mfcc: {feature_config.n_mfcc}")
    print(f"    fmin: {feature_config.fmin}")
    print(f"    fmax: {feature_config.fmax}")
    print(f"    preemphasis: {feature_config.preemphasis}")
    print(f"    use_fbank_norm: {feature_config.use_fbank_norm}")
    print(f"    frames_per_second: {feature_config.frames_per_second}")

    # 提取特征
    print(f"\n[3] 提取特征")
    features, waveform, sr = extract_features_from_file(
        audio_path, feature_config
    )

    # 可视化特征
    print(f"\n[4] 可视化特征")
    visualize_features(
        features,
        save_path='feature_visualization.png'
    )

    # 比较不同配置
    compare_configs(audio_path)

    # Delta特征演示
    demo_delta_features(audio_path)

    # 清理测试文件
    if Path('_test_audio.wav').exists():
        Path('_test_audio.wav').unlink()

    print("\n" + "=" * 70)
    print("特征提取示例完成")
    print("=" * 70)


if __name__ == '__main__':
    main()
