"""
筛选过短音频并生成排除列表

根据 dataset.MIN_DURATION 和指定的 audio_list.json，
筛选出过短的音频文件，生成 audio_list/exclude_audios.txt

Usage:
    python filter_short_audios.py --audio_list audio_list/train.json [--min_duration 1.0] [--workers 4]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import soundfile as sf
import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from dataset.dataset import MIN_DURATION


def get_audio_duration(file_path: str) -> float:
    """获取音频文件时长"""
    try:
        return sf.info(file_path).duration
    except Exception:
        return 0.0


def scan_files_chunk(file_paths: list, min_duration: float) -> list:
    """扫描一批文件，返回短音频列表"""
    short_files = []
    for file_path in file_paths:
        duration = get_audio_duration(file_path)
        if duration < min_duration:
            short_files.append((file_path, duration))
    return short_files


def collect_audio_files(data_dir: str) -> list:
    """收集目录中的所有音频文件路径"""
    audio_files = []
    audio_suffixes = ('.wav', '.mp3', '.flac', '.ogg', '.m4a')

    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(audio_suffixes):
                audio_files.append(os.path.join(root, file))

    return audio_files


def scan_directory_for_short_files_parallel(data_dir: str, min_duration: float, num_workers: int) -> list:
    """并行扫描目录，返回所有短音频文件路径"""
    # 首先收集所有音频文件
    audio_files = collect_audio_files(data_dir)

    if not audio_files:
        return []

    # 如果文件数量较少，直接顺序处理
    if len(audio_files) < 100 or num_workers == 1:
        return scan_files_chunk(audio_files, min_duration)

    # 将文件列表分割成多个块
    chunk_size = max(1, len(audio_files) // num_workers)
    chunks = [audio_files[i:i + chunk_size] for i in range(0, len(audio_files), chunk_size)]

    # 并行处理
    short_files = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(scan_files_chunk, chunk, min_duration): i for i, chunk in enumerate(chunks)}
        for future in as_completed(futures):
            chunk_result = future.result()
            short_files.extend(chunk_result)

    return short_files


def main():
    default_workers = max(1, cpu_count() // 2)

    parser = argparse.ArgumentParser(description='Filter short audio files and generate exclude list')
    parser.add_argument('--audio_list', type=str, required=True,
                        help='Path to audio list JSON (e.g., audio_list/train.json)')
    parser.add_argument('--min_duration', type=float, default=None,
                        help=f'Minimum duration in seconds (default: {MIN_DURATION})')
    parser.add_argument('--output', type=str, default=None,
                        help='Output exclude file path (default: audio_list/exclude_audios.txt)')
    parser.add_argument('--workers', type=int, default=default_workers,
                        help=f'Number of parallel workers (default: {default_workers}, use 1 for sequential)')
    args = parser.parse_args()

    min_duration = args.min_duration if args.min_duration is not None else MIN_DURATION
    audio_list_path = Path(args.audio_list)

    # 默认输出到 audio_list/exclude_audios.txt
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = audio_list_path.parent / 'exclude_audios.txt'

    print(f"Loading audio list: {audio_list_path}")
    print(f"Minimum duration threshold: {min_duration}s")
    print(f"Parallel workers: {args.workers}")

    with open(audio_list_path, 'r') as f:
        data_dict = json.load(f)

    all_short_files = []

    # 遍历所有标签和目录
    for label, paths in data_dict.items():
        # 跳过非哭声标签的重复计数
        if label != 'cry' and len(paths) > 0 and isinstance(paths[0], int):
            paths = paths[1:]

        print(f"\nScanning label '{label}': {len(paths)} directories")

        for dir_path in paths:
            if not os.path.isdir(dir_path):
                print(f"  Warning: {dir_path} is not a directory, skipping")
                continue

            short_files = scan_directory_for_short_files_parallel(dir_path, min_duration, args.workers)
            all_short_files.extend(short_files)
            print(f"  {dir_path}: found {len(short_files)} short files")

    # 写入排除列表
    print(f"\n{'=' * 60}")
    print(f"Total short audio files (< {min_duration}s): {len(all_short_files)}")
    print(f"Writing exclude list to: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"# Excluded audio files (duration < {min_duration}s)\n")
        f.write(f"# Generated from: {audio_list_path}\n")
        f.write(f"# min_duration: {min_duration}s\n")
        f.write("# Format: file_path\tduration\n")
        f.write("# " + "-" * 60 + "\n")

        for file_path, duration in sorted(all_short_files):
            f.write(f"{file_path}\n")

    print(f"Exclude list saved: {output_path}")
    print(f"To use in dataset, the file will be automatically loaded from audio_list/exclude_audios.txt")


if __name__ == '__main__':
    main()
