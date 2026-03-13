"""
并行验证音频文件完整性

Usage:
    python verify_audio.py /path/to/audio/dir --output corrupted_files.txt --workers 8
"""

import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import soundfile as sf
import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
LOGGER = logging.getLogger(__name__)

DEFAULT_AUDIO_SUFFIXES = ('.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac', '.wma')


def collect_audio_files(data_dir: str, audio_suffixes: tuple = DEFAULT_AUDIO_SUFFIXES) -> list:
    """收集目录下所有音频文件"""
    audio_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(audio_suffixes):
                audio_files.append(os.path.join(root, file))
    return audio_files


def verify_audio_file(file_path: str) -> tuple:
    """
    验证单个音频文件

    Returns:
        (file_path, is_valid, error_message)
    """
    try:
        # 检查文件大小
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            return (file_path, False, "Empty file (0 bytes)")

        # 尝试读取音频信息
        info = sf.info(file_path)
        if info.frames == 0:
            return (file_path, False, "No audio frames")
        if info.samplerate == 0:
            return (file_path, False, "Invalid sample rate")

        # 尝试实际读取一小段数据验证完整性
        frames_to_read = min(1024, info.frames)
        waveform, sr = sf.read(file_path, dtype='float32', frames=frames_to_read)

        if waveform is None or len(waveform) == 0:
            return (file_path, False, "Cannot read audio data")

        # 检查是否包含 NaN 或 Inf
        import numpy as np
        if np.isnan(waveform).any():
            return (file_path, False, "Contains NaN values")
        if np.isinf(waveform).any():
            return (file_path, False, "Contains Inf values")

        return (file_path, True, None)

    except sf.LibsndfileError as e:
        return (file_path, False, f"Libsndfile error: {e}")
    except RuntimeError as e:
        return (file_path, False, f"Runtime error: {e}")
    except Exception as e:
        return (file_path, False, f"Error: {type(e).__name__}: {e}")


def verify_audio_files(
    data_dir: str,
    output_file: str = "corrupted_audio_files.txt",
    max_workers: int = 8,
    audio_suffixes: tuple = DEFAULT_AUDIO_SUFFIXES
) -> tuple:
    """
    并行验证指定目录下所有音频文件

    Args:
        data_dir: 要检查的目录路径
        output_file: 损坏文件列表保存路径
        max_workers: 并行线程数
        audio_suffixes: 支持的音频文件后缀

    Returns:
        (正常文件数, 损坏文件数, 损坏文件列表)
    """
    if not os.path.isdir(data_dir):
        raise ValueError(f"Directory not found: {data_dir}")

    # 收集所有音频文件
    LOGGER.info(f"Scanning {data_dir} for audio files...")
    audio_files = collect_audio_files(data_dir, audio_suffixes)

    if not audio_files:
        LOGGER.warning(f"No audio files found in {data_dir}")
        return 0, 0, []

    LOGGER.info(f"Found {len(audio_files)} audio files, verifying with {max_workers} workers...")

    valid_count = 0
    corrupted_files = []

    # 并行检查
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(verify_audio_file, fp): fp for fp in audio_files}

        for future in tqdm.tqdm(as_completed(futures), total=len(audio_files),
                                desc="Verifying"):
            file_path, is_valid, error_msg = future.result()
            if is_valid:
                valid_count += 1
            else:
                corrupted_files.append((file_path, error_msg))
                LOGGER.debug(f"Corrupted: {file_path} - {error_msg}")

    # 保存损坏文件列表
    if corrupted_files:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# Corrupted Audio Files Report\n")
            f.write(f"# Source directory: {os.path.abspath(data_dir)}\n")
            f.write(f"# Total checked: {len(audio_files)}\n")
            f.write(f"# Valid: {valid_count}\n")
            f.write(f"# Corrupted: {len(corrupted_files)}\n")
            f.write(f"# {'=' * 60}\n\n")
            for file_path, error in corrupted_files:
                f.write(f"{file_path}\n")
                f.write(f"  Error: {error}\n\n")
        LOGGER.warning(f"Found {len(corrupted_files)} corrupted files, saved to: {output_file}")
    else:
        LOGGER.info("All audio files are valid!")

    # 打印统计
    LOGGER.info(f"Summary: {valid_count} valid, {len(corrupted_files)} corrupted "
                f"out of {len(audio_files)} total")

    return valid_count, len(corrupted_files), [f[0] for f in corrupted_files]


def main():
    parser = argparse.ArgumentParser(description="Verify audio file integrity")
    parser.add_argument("data_dir", help="Directory containing audio files")
    parser.add_argument("-o", "--output", default="corrupted_audio_files.txt",
                        help="Output file for corrupted files list (default: corrupted_audio_files.txt)")
    parser.add_argument("-w", "--workers", type=int, default=8,
                        help="Number of parallel workers (default: 8)")
    parser.add_argument("--delete-corrupted", action="store_true",
                        help="Delete corrupted files after verification (use with caution)")
    args = parser.parse_args()

    valid, corrupted, corrupted_list = verify_audio_files(
        args.data_dir,
        output_file=args.output,
        max_workers=args.workers
    )

    if args.delete_corrupted and corrupted_list:
        LOGGER.warning(f"Deleting {len(corrupted_list)} corrupted files...")
        for file_path in corrupted_list:
            try:
                os.remove(file_path)
                LOGGER.info(f"Deleted: {file_path}")
            except Exception as e:
                LOGGER.error(f"Failed to delete {file_path}: {e}")

    # 返回退出码
    exit_code = 0 if corrupted == 0 else 1
    exit(exit_code)


if __name__ == "__main__":
    main()
