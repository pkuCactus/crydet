"""
并行查找指定目录下的非 WAV 格式音频文件

Usage:
    python find_non_wav.py /path/to/audio/dir --output non_wav_files.txt
    python find_non_wav.py /path/to/audio/dir --allowed .wav .flac --workers 8
"""

import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
LOGGER = logging.getLogger(__name__)


def collect_all_files(data_dir: str) -> list:
    """递归收集目录下所有文件"""
    files = []
    for root, _, filenames in os.walk(data_dir):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return files


def check_file(args: tuple) -> tuple:
    """
    检查单个文件是否为允许的后缀

    Returns:
        (file_path, is_allowed, suffix)
    """
    file_path, allowed_suffixes = args
    suffix = Path(file_path).suffix.lower()
    is_allowed = suffix in allowed_suffixes
    return (file_path, is_allowed, suffix)


def find_non_allowed_files(
    data_dir: str,
    output_file: str = "non_wav_files.txt",
    allowed_suffixes: tuple = ('.wav',),
    max_workers: int = 8
) -> tuple:
    """
    并行查找非指定后缀的文件

    Args:
        data_dir: 要扫描的目录
        output_file: 输出文件路径
        allowed_suffixes: 允许的文件后缀
        max_workers: 并行线程数

    Returns:
        (允许的文件数, 非允许的文件数, 非允许文件列表)
    """
    if not os.path.isdir(data_dir):
        raise ValueError(f"Directory not found: {data_dir}")

    # 收集所有文件
    LOGGER.info(f"Scanning {data_dir} for files...")
    all_files = collect_all_files(data_dir)

    if not all_files:
        LOGGER.warning(f"No files found in {data_dir}")
        return 0, 0, []

    LOGGER.info(f"Found {len(all_files)} files, checking with {max_workers} workers...")

    allowed_count = 0
    non_allowed_files = []

    # 并行检查
    check_args = [(fp, allowed_suffixes) for fp in all_files]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(check_file, arg): arg[0] for arg in check_args}

        for future in tqdm.tqdm(as_completed(futures), total=len(check_args), desc="Checking"):
            file_path, is_allowed, suffix = future.result()
            if is_allowed:
                allowed_count += 1
            else:
                non_allowed_files.append((file_path, suffix))

    # 保存结果
    if non_allowed_files:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# Non-allowed files in {os.path.abspath(data_dir)}\n")
            f.write(f"# Allowed suffixes: {', '.join(allowed_suffixes)}\n")
            f.write(f"# Total files: {len(all_files)}\n")
            f.write(f"# Allowed: {allowed_count}\n")
            f.write(f"# Non-allowed: {len(non_allowed_files)}\n")
            f.write(f"# {'=' * 60}\n\n")
            for file_path, suffix in sorted(non_allowed_files):
                f.write(f"{file_path}\t{suffix}\n")
        LOGGER.warning(f"Found {len(non_allowed_files)} non-allowed files, saved to: {output_file}")
    else:
        LOGGER.info(f"All {len(all_files)} files have allowed suffixes: {allowed_suffixes}")

    # 打印统计
    LOGGER.info(f"Summary: {allowed_count} allowed, {len(non_allowed_files)} non-allowed")

    return allowed_count, len(non_allowed_files), [f[0] for f in non_allowed_files]


def main():
    parser = argparse.ArgumentParser(
        description="Find files with non-allowed suffixes in a directory"
    )
    parser.add_argument(
        "data_dir",
        help="Directory to scan"
    )
    parser.add_argument(
        "-o", "--output",
        default="non_wav_files.txt",
        help="Output file for non-allowed files list (default: non_wav_files.txt)"
    )
    parser.add_argument(
        "--allowed",
        nargs='+',
        default=['.wav'],
        help="Allowed file suffixes (default: .wav)"
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)"
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete non-allowed files after listing (use with caution)"
    )
    args = parser.parse_args()

    # 标准化后缀格式
    allowed_suffixes = tuple(
        s if s.startswith('.') else f'.{s}'
        for s in args.allowed
    )

    allowed, non_allowed, non_allowed_list = find_non_allowed_files(
        args.data_dir,
        output_file=args.output,
        allowed_suffixes=allowed_suffixes,
        max_workers=args.workers
    )

    if args.delete and non_allowed_list:
        LOGGER.warning(f"Deleting {len(non_allowed_list)} non-allowed files...")
        for file_path in non_allowed_list:
            try:
                os.remove(file_path)
                LOGGER.info(f"Deleted: {file_path}")
            except Exception as e:
                LOGGER.error(f"Failed to delete {file_path}: {e}")

    # 返回退出码
    exit(0 if non_allowed == 0 else 1)


if __name__ == "__main__":
    main()
