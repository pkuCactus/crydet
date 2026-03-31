"""
音频/视频文件处理工具

递归遍历目录，处理视频和音频文件：
- 视频：保留到 output_dir/../video，提取音频到 output_dir
- 音频：去重检查，wav直接移动，非wav先备份再转换
- 其他文件：完全保留在原目录，不做任何处理

Usage:
    python process_media.py --input /path/to/input --output /path/to/output
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple


def compute_md5(file_path: str, chunk_size: int = 8192) -> str:
    """计算文件的MD5哈希值"""
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def get_audio_duration(file_path: str) -> float:
    """获取音频/视频文件时长（秒）"""
    try:
        import soundfile as sf
        info = sf.info(file_path)
        return info.duration
    except Exception:
        return 0.0


def extract_audio_from_video(video_path: str, output_wav_path: str) -> bool:
    """从视频提取音频为wav格式"""
    try:
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            output_wav_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error extracting audio from {video_path}: {e}")
        return False
    except FileNotFoundError:
        print("Error: ffmpeg not found. Please install ffmpeg.")
        return False


def convert_to_wav(input_path: str, output_wav_path: str, target_sr: int = 16000) -> bool:
    """转换音频为wav格式"""
    try:
        cmd = [
            "ffmpeg", "-y", "-i", input_path,
            "-acodec", "pcm_s16le",
            "-ar", str(target_sr), "-ac", "1",
            output_wav_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error converting {input_path}: {e}")
        return False
    except FileNotFoundError:
        print("Error: ffmpeg not found. Please install ffmpeg.")
        return False


def find_existing_file(output_dir: Path, filename: str) -> Optional[Tuple[Path, str]]:
    """
    在输出目录中查找同名文件

    Returns:
        (file_path, md5) if found, None otherwise
    """
    target_path = output_dir / filename
    if target_path.exists():
        return target_path, compute_md5(str(target_path))
    return None


def generate_unique_name(output_dir: Path, base_name: str, md5_hash: str, ext: str = ".wav") -> str:
    """生成唯一的文件名（当同名但内容不同时）"""
    counter = 1
    while True:
        new_name = f"{base_name}_{counter}{ext}"
        target_path = output_dir / new_name
        if not target_path.exists():
            return new_name
        # 如果存在，检查MD5
        existing_md5 = compute_md5(str(target_path))
        if existing_md5 == md5_hash:
            return new_name  # 相同内容，返回这个名字
        counter += 1


def is_video_file(file_path: str) -> bool:
    """检查是否为视频文件"""
    video_suffixes = ('.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.mpg', '.mpeg')
    return file_path.lower().endswith(video_suffixes)


def is_audio_file(file_path: str) -> bool:
    """检查是否为音频文件"""
    audio_suffixes = ('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma', '.opus')
    return file_path.lower().endswith(audio_suffixes)


def process_video(video_path: Path, output_dir: Path) -> bool:
    """
    处理视频文件：
    1. 复制到 output_dir/../video
    2. 提取音频到 output_dir
    """
    # 创建 video 目录（与 output_dir 同级）
    video_dir = output_dir.parent / "video"
    video_dir.mkdir(parents=True, exist_ok=True)

    # 复制视频文件
    video_target = video_dir / video_path.name
    try:
        shutil.copy2(str(video_path), str(video_target))
        print(f"  Video copied to: {video_target}")
    except Exception as e:
        print(f"  Error copying video: {e}")
        return False

    # 提取音频
    audio_name = video_path.stem + ".wav"
    audio_target = output_dir / audio_name

    if audio_target.exists():
        # 检查是否相同
        existing_md5 = compute_md5(str(audio_target))
        temp_wav = output_dir / f"temp_{audio_name}"
        if extract_audio_from_video(str(video_path), str(temp_wav)):
            new_md5 = compute_md5(str(temp_wav))
            if existing_md5 == new_md5:
                print(f"  Audio already exists with same content, skipping")
                temp_wav.unlink()
                return True
            else:
                # 内容不同，重命名
                unique_name = generate_unique_name(output_dir, video_path.stem, new_md5)
                final_target = output_dir / unique_name
                shutil.move(str(temp_wav), str(final_target))
                print(f"  Audio extracted with unique name: {final_target}")
                return True
    else:
        if extract_audio_from_video(str(video_path), str(audio_target)):
            print(f"  Audio extracted to: {audio_target}")
            return True

    return False


def process_audio(audio_path: Path, output_dir: Path) -> bool:
    """
    处理音频文件：
    1. 检查输出目录是否有相同文件（名字+MD5）
    2. 如果有，删除当前文件
    3. 名字相同内容不同，自动重命名
    4. 没有相同的：
       - wav格式：直接移动
       - 非wav：先拷贝到 ../audio，然后转换成wav
    """
    current_md5 = compute_md5(str(audio_path))
    base_name = audio_path.stem
    ext = audio_path.suffix.lower()
    is_wav = ext == ".wav"

    # 检查输出目录是否有同名文件
    target_wav_name = base_name + ".wav"
    existing = find_existing_file(output_dir, target_wav_name)

    if existing:
        existing_path, existing_md5 = existing
        if existing_md5 == current_md5:
            # 完全相同，删除当前文件
            print(f"  Duplicate found, removing: {audio_path}")
            audio_path.unlink()
            return True
        else:
            # 名字相同但内容不同，重命名
            unique_name = generate_unique_name(output_dir, base_name, current_md5)
            target_wav_name = unique_name
            print(f"  Name conflict, using: {target_wav_name}")

    target_wav_path = output_dir / target_wav_name

    if is_wav:
        # 直接移动
        try:
            shutil.move(str(audio_path), str(target_wav_path))
            print(f"  Moved to: {target_wav_path}")
            return True
        except Exception as e:
            print(f"  Error moving file: {e}")
            return False
    else:
        # 非wav格式：先备份到 ../audio，然后转换
        backup_dir = output_dir.parent / "audio"
        backup_dir.mkdir(parents=True, exist_ok=True)

        # 复制原始文件到 backup_dir
        backup_path = backup_dir / audio_path.name
        try:
            shutil.copy2(str(audio_path), str(backup_path))
            print(f"  Original backed up to: {backup_path}")
        except Exception as e:
            print(f"  Error backing up file: {e}")
            return False

        # 转换为wav
        if convert_to_wav(str(audio_path), str(target_wav_path)):
            print(f"  Converted to: {target_wav_path}")
            # 删除原始文件
            audio_path.unlink()
            return True
        else:
            print(f"  Conversion failed")
            return False


def collect_media_files(input_dir: Path) -> Tuple[list, list]:
    """收集所有视频和音频文件"""
    video_files = []
    audio_files = []

    for root, _, files in os.walk(input_dir):
        for file in files:
            file_path = Path(root) / file
            if is_video_file(str(file_path)):
                video_files.append(file_path)
            elif is_audio_file(str(file_path)):
                audio_files.append(file_path)

    return video_files, audio_files


def main():
    parser = argparse.ArgumentParser(description='Process media files (video/audio)')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input directory to scan recursively')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output directory for wav files')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print actions without executing')

    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning: {input_dir}")
    print(f"Output: {output_dir}")
    print("-" * 60)

    # 收集文件
    video_files, audio_files = collect_media_files(input_dir)

    print(f"Found {len(video_files)} video files, {len(audio_files)} audio files")
    print("-" * 60)

    if args.dry_run:
        print("DRY RUN MODE - No files will be modified")
        print("-" * 60)

    # 处理视频
    if video_files:
        print(f"\nProcessing {len(video_files)} video files...")
        for i, video_path in enumerate(video_files, 1):
            print(f"[{i}/{len(video_files)}] {video_path}")
            if not args.dry_run:
                process_video(video_path, output_dir)

    # 处理音频
    if audio_files:
        print(f"\nProcessing {len(audio_files)} audio files...")
        for i, audio_path in enumerate(audio_files, 1):
            print(f"[{i}/{len(audio_files)}] {audio_path}")
            if not args.dry_run:
                process_audio(audio_path, output_dir)

    print("\n" + "=" * 60)
    print("Processing complete!")


if __name__ == '__main__':
    main()
