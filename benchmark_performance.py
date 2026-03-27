"""
Performance benchmark script for data loading pipeline optimizations

Tests:
1. Audio reading speed with/without memory cache
2. Augmentation speed with/without noise pool
3. Feature extraction speed with torch.compile
"""

import time
import torch
import numpy as np
from typing import Callable
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def benchmark_func(func: Callable, name: str, warmup: int = 3, iterations: int = 10) -> dict:
    """Benchmark a function"""
    # Warmup
    for _ in range(warmup):
        func()

    # Benchmark
    times = []
    for _ in range(iterations):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()
        func()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.perf_counter()
        times.append(end - start)

    return {
        'name': name,
        'mean_ms': np.mean(times) * 1000,
        'std_ms': np.std(times) * 1000,
        'min_ms': np.min(times) * 1000,
        'max_ms': np.max(times) * 1000,
    }


def test_audio_reader():
    """Test audio reader performance"""
    from dataset.audio_reader import AudioReader

    # Create a dummy audio file for testing
    import tempfile
    import soundfile as sf
    import os

    # Create test audio file
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, 'test_audio.wav')
        test_audio = np.random.randn(16000 * 5).astype(np.float32)  # 5 seconds @ 16kHz
        sf.write(test_file, test_audio, 16000)

        logger.info("=" * 60)
        logger.info("Audio Reader Benchmark")
        logger.info("=" * 60)

        # Test without memory cache
        reader_no_cache = AudioReader(target_sr=16000, memory_cache_mb=0)

        def load_no_cache():
            return reader_no_cache.load(test_file, use_memory_cache=False)

        result_no_cache = benchmark_func(load_no_cache, "Without Memory Cache", iterations=20)
        logger.info(f"Without cache: {result_no_cache['mean_ms']:.2f} ± {result_no_cache['std_ms']:.2f} ms")

        # Test with memory cache
        reader_with_cache = AudioReader(target_sr=16000, memory_cache_mb=100, memory_cache_size=10)

        # First load (cache miss)
        start = time.perf_counter()
        reader_with_cache.load(test_file, use_memory_cache=True)
        first_load_ms = (time.perf_counter() - start) * 1000
        logger.info(f"First load (cache miss): {first_load_ms:.2f} ms")

        # Cached loads
        def load_with_cache():
            return reader_with_cache.load(test_file, use_memory_cache=True)

        result_with_cache = benchmark_func(load_with_cache, "With Memory Cache", iterations=20)
        logger.info(f"With cache: {result_with_cache['mean_ms']:.2f} ± {result_with_cache['std_ms']:.2f} ms")

        # Speedup
        speedup = result_no_cache['mean_ms'] / result_with_cache['mean_ms']
        logger.info(f"Cache speedup: {speedup:.2f}x")

        # Cache stats
        cache_stats = reader_with_cache.get_cache_info()
        logger.info(f"Memory cache stats: {cache_stats['memory_cache']}")

        return {
            'no_cache': result_no_cache,
            'with_cache': result_with_cache,
            'speedup': speedup,
        }


def test_augmentation():
    """Test augmentation performance"""
    from dataset.augmentation import AudioAugmenter
    from utils.config import AugmentationConfig, NoiseConfig

    logger.info("\n" + "=" * 60)
    logger.info("Augmentation Benchmark")
    logger.info("=" * 60)

    # Create test audio
    test_audio = np.random.randn(16000 * 5).astype(np.float32) * 0.1

    # Create config
    aug_config = AugmentationConfig(
        cry_aug_prob=1.0,
        other_aug_prob=1.0,
        noise=NoiseConfig(prob=0.5, white_noise_prob=0.5, pink_noise_prob=0.5),
    )

    # Test with noise pool
    augmenter = AudioAugmenter(aug_config, sample_rate=16000, noise_pool_size=20)
    augmenter._init_noise_pool()

    def augment_cry():
        return augmenter.augment(test_audio.copy(), 'cry')

    result = benchmark_func(augment_cry, "Augmentation with noise pool", iterations=10)
    logger.info(f"Augmentation: {result['mean_ms']:.2f} ± {result['std_ms']:.2f} ms")

    return {'augmentation': result}


def test_feature_extraction():
    """Test feature extraction performance"""
    from dataset.feature import FeatureExtractor
    from utils.config import FeatureConfig

    logger.info("\n" + "=" * 60)
    logger.info("Feature Extraction Benchmark")
    logger.info("=" * 60)

    # Create config
    feat_config = FeatureConfig(
        feature_type=1,  # FBANK
        use_time_delta=True,
        use_fbank_norm=True,
    )

    # Create test audio batch
    batch_size = 32
    test_audio = torch.randn(batch_size, 16000 * 5)  # 5 seconds @ 16kHz

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_audio = test_audio.to(device)

    # Test without compile
    extractor = FeatureExtractor(feat_config, sr=16000).to(device)
    extractor.eval()

    def extract_no_compile():
        with torch.no_grad():
            return extractor(test_audio)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = extractor(test_audio)

    result_no_compile = benchmark_func(extract_no_compile, "Without torch.compile", iterations=20)
    logger.info(f"Without compile: {result_no_compile['mean_ms']:.2f} ± {result_no_compile['std_ms']:.2f} ms")

    # Test with compile (if available)
    if hasattr(torch, 'compile'):
        try:
            extractor_compiled = FeatureExtractor(feat_config, sr=16000).to(device)
            extractor_compiled.compile(mode='reduce-overhead')
            extractor_compiled.eval()

            def extract_with_compile():
                with torch.no_grad():
                    return extractor_compiled(test_audio)

            # Warmup compilation
            with torch.no_grad():
                for _ in range(5):
                    _ = extractor_compiled(test_audio)

            result_with_compile = benchmark_func(extract_with_compile, "With torch.compile", iterations=20)
            logger.info(f"With compile: {result_with_compile['mean_ms']:.2f} ± {result_with_compile['std_ms']:.2f} ms")

            speedup = result_no_compile['mean_ms'] / result_with_compile['mean_ms']
            logger.info(f"Compilation speedup: {speedup:.2f}x")

            return {
                'no_compile': result_no_compile,
                'with_compile': result_with_compile,
                'speedup': speedup,
            }
        except Exception as e:
            logger.warning(f"torch.compile not available: {e}")

    return {'no_compile': result_no_compile}


def test_end_to_end():
    """Test end-to-end data loading pipeline"""
    logger.info("\n" + "=" * 60)
    logger.info("End-to-End Pipeline Benchmark")
    logger.info("=" * 60)

    # This requires actual data, so we just print info
    logger.info("End-to-end benchmark requires actual audio dataset.")
    logger.info("Run training script with --benchmark flag to test full pipeline.")


def main():
    """Run all benchmarks"""
    logger.info("Starting performance benchmarks...")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

    results = {}

    try:
        results['audio_reader'] = test_audio_reader()
    except Exception as e:
        logger.error(f"Audio reader benchmark failed: {e}")

    try:
        results['augmentation'] = test_augmentation()
    except Exception as e:
        logger.error(f"Augmentation benchmark failed: {e}")

    try:
        results['feature_extraction'] = test_feature_extraction()
    except Exception as e:
        logger.error(f"Feature extraction benchmark failed: {e}")

    test_end_to_end()

    logger.info("\n" + "=" * 60)
    logger.info("Benchmark Summary")
    logger.info("=" * 60)

    # Print summary
    if 'audio_reader' in results:
        ar = results['audio_reader']
        logger.info(f"Audio Reader: {ar['speedup']:.2f}x speedup with memory cache")

    if 'feature_extraction' in results and 'speedup' in results['feature_extraction']:
        fe = results['feature_extraction']
        logger.info(f"Feature Extraction: {fe['speedup']:.2f}x speedup with torch.compile")

    logger.info("\nBenchmarks complete!")


if __name__ == '__main__':
    main()
