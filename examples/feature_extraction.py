"""
特征提取示例
演示如何读取音频文件并提取 MFCC/Filter Bank 特征
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
        features: 提取的特征 (feature_dim, time_frames)
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
    print(f"特征形状: {features.shape}")
    print(f"  - 特征维度: {features.shape[0]}")
    print(f"  - 时间帧数: {features.shape[1]}")

    return features, waveform, sr


def visualize_features(
    features: np.ndarray,
    config: FeatureConfig,
    save_path: str = None
):
    """
    可视化特征

    Args:
        features: 特征矩阵 (feature_dim, time_frames)
        config: 特征配置
        save_path: 保存路径 (可选)
    """
    n_channels = config.num_channels
    feature_dim = config.feature_dim

    fig, axes = plt.subplots(n_channels, 1, figsize=(12, 4 * n_channels))

    if n_channels == 1:
        axes = [axes]

    channel_names = ['Base']
    if config.use_delta:
        channel_names.append('Delta (Time)')
    if config.use_freq_delta:
        channel_names.append('Delta (Freq)')

    for i, (ax, name) in enumerate(zip(axes, channel_names)):
        start_idx = i * feature_dim
        end_idx = (i + 1) * feature_dim
        channel_features = features[start_idx:end_idx, :]

        im = ax.imshow(channel_features, aspect='auto', origin='lower', cmap='viridis')
        ax.set_title(f'{name} Features')
        ax.set_ylabel('Feature Dim')
        ax.set_xlabel('Time Frames')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"特征可视化已保存到: {save_path}")
    else:
        plt.show()


def compare_feature_types(audio_path: str, sample_rate: int = 16000):
    """
    比较不同特征类型的输出

    Args:
        audio_path: 音频文件路径
        sample_rate: 采样率
    """
    print("\n" + "=" * 60)
    print("比较不同特征类型")
    print("=" * 60)

    # 读取音频
    audio_reader = AudioReader(target_sr=sample_rate, force_mono=True)
    waveform, sr = audio_reader.load(audio_path)

    configs = [
        ("FBank (64 mel)", FeatureConfig(feature_type='fbank', n_mels=64)),
        ("FBank (128 mel)", FeatureConfig(feature_type='fbank', n_mels=128)),
        ("MFCC (40)", FeatureConfig(feature_type='mfcc', n_mfcc=40, n_mels=64)),
        ("FBank + Delta", FeatureConfig(feature_type='fbank', n_mels=64, use_delta=True, use_freq_delta=False)),
        ("FBank + Delta + FreqDelta", FeatureConfig(feature_type='fbank', n_mels=64, use_delta=True, use_freq_delta=True)),
    ]

    fig, axes = plt.subplots(len(configs), 1, figsize=(14, 3 * len(configs)))

    for i, (name, cfg) in enumerate(configs):
                extractor = FeatureExtractor(cfg)
                features = extractor.extract(waveform, sr)

                ax = axes[i] if len(configs) > 1 else axes
                im = ax.imshow(features, aspect='auto', origin='lower', cmap='viridis')
                ax.set_title(f'{name}\nShape: {features.shape}')
                ax.set_ylabel('Feature Dim')
                ax.set_xlabel('Time Frames')

    plt.tight_layout()
    plt.savefig('feature_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n比较图已保存到: feature_comparison.png")
    plt.close()


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

    # 加载配置
    if Path(config_path).exists():
        config = load_config(config_path)
        feature_config = config.feature
    else:
        feature_config = FeatureConfig()
    print(f"\n[2] 特征配置:")
    print(f"    类型: {feature_config.feature_type}")
    print(f"    n_mels: {feature_config.n_mels}")
    print(f"    n_fft: {feature_config.n_fft}")
    print(f"    hop_length: {feature_config.hop_length}")
    print(f"    use_delta: {feature_config.use_delta}")
    print(f"    use_freq_delta: {feature_config.use_freq_delta}")
    print(f"    normalize: {feature_config.normalize}")

    # 提取特征
    print(f"\n[3] 提取特征")
    features, waveform, sr = extract_features_from_file(
        audio_path, feature_config
    )

    # 输出特征统计
    print(f"\n[4] 特征统计:")
    print(f"    最小值: {features.min():.4f}")
    print(f"    最大值: {features.max():.4f}")
    print(f"    均值: {features.mean():.4f}")
    print(f"    标准差: {features.std():.4f}")

    # 可视化特征
    print(f"\n[5] 可视化特征")
    visualize_features(
        features, feature_config,
        save_path='feature_visualization.png'
    )

    # 比较不同特征类型
    compare_feature_types(audio_path)

    # 清理测试文件
    if Path('_test_audio.wav').exists():
        Path('_test_audio.wav').unlink()

    print("\n" + "=" * 70)
    print("特征提取示例完成")
    print("=" * 70)


if __name__ == '__main__':
    main()
