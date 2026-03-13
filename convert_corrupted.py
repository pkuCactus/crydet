"""
转换损坏音频文件为 WAV 格式

读取 verify_audio.py 输出的损坏文件列表，使用 ffprobe 获取信息，
然后用 ffmpeg 转换为 WAV 格式，保持原始通道数、采样率等参数。

Usage:
    python convert_corrupted.py corrupted_audio_files.txt --output-dir converted/
"""

import argparse
import logging
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
LOGGER = logging.getLogger(__name__)


def parse_corrupted_list(input_file: str) -> list:
    """解析损坏文件列表"""
    corrupted_files = []

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # 跳过注释和空行
            if not line or line.startswith('#'):
                continue
            # 处理 "文件路径\t错误信息" 格式
            if '\t' in line:
                file_path = line.split('\t')[0]
            else:
                file_path = line
            corrupted_files.append(file_path)

    return corrupted_files


def get_audio_info(file_path: str) -> dict:
    """
    使用 ffprobe 获取音频文件信息

    Returns:
        dict with keys: sample_rate, channels, bit_depth, duration, codec
    """
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'a:0',
            '-show_entries', 'stream=sample_rate,channels,bits_per_raw_sample',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1',
            file_path
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30
        )

        if result.returncode != 0:
            LOGGER.warning(f"ffprobe failed for {file_path}: {result.stderr}")
            return None

        info = {
            'sample_rate': None,
            'channels': None,
            'bit_depth': None,
            'duration': None,
        }

        for line in result.stdout.strip().split('\n'):
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()

                if key == 'sample_rate':
                    info['sample_rate'] = int(value) if value.isdigit() else None
                elif key == 'channels':
                    info['channels'] = int(value) if value.isdigit() else None
                elif key == 'bits_per_raw_sample':
                    info['bit_depth'] = int(value) if value.isdigit() else None
                elif key == 'duration':
                    try:
                        info['duration'] = float(value)
                    except ValueError:
                        pass

        return info

    except subprocess.TimeoutExpired:
        LOGGER.warning(f"ffprobe timeout for {file_path}")
        return None
    except Exception as e:
        LOGGER.warning(f"ffprobe error for {file_path}: {e}")
        return None


def convert_to_wav(input_path: str, output_path: str, info: dict = None) -> bool:
    """
    使用 ffmpeg 将音频转换为 WAV 格式

    Args:
        input_path: 输入音频文件路径
        output_path: 输出 WAV 文件路径
        info: ffprobe 获取的音频信息（可选）

    Returns:
        True if conversion successful, False otherwise
    """
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 构建 ffmpeg 命令
        cmd = ['ffmpeg', '-y', '-v', 'error', '-i', input_path]

        # 如果获取到了信息，保持原始参数
        if info:
            if info.get('sample_rate'):
                cmd.extend(['-ar', str(info['sample_rate'])])
            if info.get('channels'):
                cmd.extend(['-ac', str(info['channels'])])

        # 输出为 WAV 格式
        cmd.extend(['-acodec', 'pcm_s16le', output_path])

        # 执行转换
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120
        )

        if result.returncode != 0:
            LOGGER.error(f"FFmpeg failed for {input_path}: {result.stderr}")
            return False

        # 验证输出文件
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            LOGGER.error(f"Output file is empty: {output_path}")
            return False

        LOGGER.info(f"Converted: {input_path} -> {output_path}")
        return True

    except subprocess.TimeoutExpired:
        LOGGER.error(f"FFmpeg timeout for {input_path}")
        return False
    except Exception as e:
        LOGGER.error(f"FFmpeg error for {input_path}: {e}")
        return False


def process_file(args: tuple) -> tuple:
    """
    处理单个文件

    Returns:
        (input_path, success, output_path_or_error)
    """
    input_path, output_dir, keep_structure = args

    # 确定输出路径
    if keep_structure:
        # 保持目录结构
        rel_path = Path(input_path).name
        output_path = os.path.join(output_dir, rel_path)
        output_path = Path(output_path).with_suffix('.wav')
    else:
        # 扁平化输出
        base_name = Path(input_path).stem
        output_path = os.path.join(output_dir, f"{base_name}.wav")

    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        return (input_path, False, "File not found")

    # 获取音频信息
    info = get_audio_info(input_path)

    # 转换文件
    success = convert_to_wav(input_path, str(output_path), info)

    if success:
        return (input_path, True, str(output_path))
    else:
        return (input_path, False, "Conversion failed")


def main():
    parser = argparse.ArgumentParser(
        description="Convert corrupted audio files to WAV using ffmpeg"
    )
    parser.add_argument(
        "input_file",
        help="Text file containing list of corrupted audio files (from verify_audio.py)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="converted",
        help="Output directory for converted WAV files (default: converted/)"
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--keep-structure",
        action="store_true",
        help="Keep original directory structure in output"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without actually converting"
    )
    args = parser.parse_args()

    # 解析损坏文件列表
    LOGGER.info(f"Reading corrupted file list from: {args.input_file}")
    corrupted_files = parse_corrupted_list(args.input_file)

    if not corrupted_files:
        LOGGER.warning("No files to convert")
        return

    LOGGER.info(f"Found {len(corrupted_files)} files to process")

    if args.dry_run:
        LOGGER.info("Dry run mode - no actual conversion")
        for file_path in corrupted_files[:10]:  # 只显示前10个
            LOGGER.info(f"Would convert: {file_path}")
        if len(corrupted_files) > 10:
            LOGGER.info(f"... and {len(corrupted_files) - 10} more")
        return

    # 准备处理参数
    process_args = [
        (fp, args.output_dir, args.keep_structure)
        for fp in corrupted_files
    ]

    # 并行处理
    success_count = 0
    failed_files = []

    LOGGER.info(f"Starting conversion with {args.workers} workers...")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_file, arg): arg[0] for arg in process_args}

        for future in tqdm.tqdm(as_completed(futures), total=len(process_args), desc="Converting"):
            input_path, success, result = future.result()

            if success:
                success_count += 1
            else:
                failed_files.append((input_path, result))
                LOGGER.error(f"Failed to convert: {input_path} - {result}")

    # 打印总结
    LOGGER.info(f"Conversion complete: {success_count}/{len(corrupted_files)} successful")

    if failed_files:
        failed_list_file = os.path.join(args.output_dir, "failed_conversions.txt")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(failed_list_file, 'w', encoding='utf-8') as f:
            f.write(f"# Failed Conversions - {len(failed_files)} files\n\n")
            for file_path, error in failed_files:
                f.write(f"{file_path}\n")
                f.write(f"  Error: {error}\n\n")
        LOGGER.warning(f"Failed conversions saved to: {failed_list_file}")

        # 打印失败的文件路径
        print("\n" + "=" * 60)
        print("Failed to convert the following files:")
        print("=" * 60)
        for file_path, error in failed_files:
            print(file_path)
        print("=" * 60)


if __name__ == "__main__":
    main()
