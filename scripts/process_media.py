"""
音频/视频文件处理工具

递归遍历目录，处理视频和音频文件：
- 视频：保留到 output_dir/../video，提取音频到 output_dir
- 音频：去重检查，wav直接移动，非wav先备份再转换
- 其他文件：完全保留在原目录，不做任何处理

支持操作日志记录和逆向操作

Usage:
    # 正常处理
    python process_media.py --input /path/to/input --output /path/to/output

    # 逆向操作（根据日志撤销）
    python process_media.py --undo --log-file /path/to/operation_log.json
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any


class OperationLogger:
    """操作日志记录器"""

    def __init__(self, log_file: Optional[Path] = None):
        self.operations: List[Dict[str, Any]] = []
        self.log_file = log_file
        self.start_time = datetime.now().isoformat()

    def log(self, op_type: str, **kwargs):
        """记录一个操作"""
        op = {
            "type": op_type,
            "timestamp": datetime.now().isoformat(),
            **kwargs
        }
        self.operations.append(op)
        return op

    def save(self):
        """保存日志到文件"""
        if self.log_file:
            log_data = {
                "start_time": self.start_time,
                "end_time": datetime.now().isoformat(),
                "operation_count": len(self.operations),
                "operations": self.operations
            }
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            print(f"\nOperation log saved to: {self.log_file}")

    @classmethod
    def load(cls, log_file: Path) -> "OperationLogger":
        """从文件加载日志"""
        logger = cls(log_file)
        with open(log_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            logger.operations = data.get("operations", [])
            logger.start_time = data.get("start_time", "")
        return logger


def compute_md5(file_path: str, chunk_size: int = 8192) -> str:
    """计算文件的MD5哈希值"""
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


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
    """在输出目录中查找同名文件"""
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
        existing_md5 = compute_md5(str(target_path))
        if existing_md5 == md5_hash:
            return new_name
        counter += 1


def is_video_file(file_path: str) -> bool:
    """检查是否为视频文件"""
    video_suffixes = ('.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v', '.mpg', '.mpeg', '.3gp')
    return file_path.lower().endswith(video_suffixes)


def is_audio_file(file_path: str) -> bool:
    """检查是否为音频文件"""
    audio_suffixes = ('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma', '.opus')
    return file_path.lower().endswith(audio_suffixes)


def process_video(video_path: Path, output_dir: Path, logger: OperationLogger, dry_run: bool = False) -> bool:
    """
    处理视频文件：
    1. 复制到 output_dir/../video
    2. 提取音频到 output_dir
    """
    video_dir = output_dir.parent / "video"
    video_dir.mkdir(parents=True, exist_ok=True)

    video_target = video_dir / video_path.name
    audio_name = video_path.stem + ".wav"
    audio_target = output_dir / audio_name

    # 检查是否需要重命名
    final_audio_name = audio_name
    if audio_target.exists():
        existing_md5 = compute_md5(str(audio_target))
        if not dry_run:
            temp_wav = output_dir / f"temp_{audio_name}"
            if extract_audio_from_video(str(video_path), str(temp_wav)):
                new_md5 = compute_md5(str(temp_wav))
                if existing_md5 == new_md5:
                    print(f"  Audio already exists with same content, skipping")
                    temp_wav.unlink()
                    logger.log("skip_duplicate", video=str(video_path), reason="same_audio_exists", md5=new_md5)
                    return True
                else:
                    final_audio_name = generate_unique_name(output_dir, video_path.stem, new_md5)
                    audio_target = output_dir / final_audio_name
                    shutil.move(str(temp_wav), str(audio_target))
                    print(f"  Audio extracted with unique name: {audio_target}")
                    # 复制视频并记录
                    if not dry_run:
                        shutil.copy2(str(video_path), str(video_target))
                    logger.log("video_copy", src=str(video_path), dst=str(video_target))
                    logger.log("video_extract", src=str(video_path), src_video=str(video_path),
                              dst_wav=str(audio_target), video_dst=str(video_target))
                    return True
    else:
        if not dry_run:
            shutil.copy2(str(video_path), str(video_target))
            if extract_audio_from_video(str(video_path), str(audio_target)):
                print(f"  Video copied to: {video_target}")
                print(f"  Audio extracted to: {audio_target}")
                logger.log("video_copy", src=str(video_path), dst=str(video_target))
                logger.log("video_extract", src=str(video_path), src_video=str(video_path),
                          dst_wav=str(audio_target), video_dst=str(video_target))
                return True
    return False


def process_audio(audio_path: Path, output_dir: Path, logger: OperationLogger, dry_run: bool = False) -> bool:
    """
    处理音频文件：
    1. 检查输出目录是否有相同文件（名字+MD5）
    2. 如果有，删除当前文件
    3. 名字相同内容不同，自动重命名
    4. 没有相同的：
       - wav格式：直接移动
       - 非wav：先拷贝到 ../audio，然后转换成wav
    """
    if dry_run:
        return True

    current_md5 = compute_md5(str(audio_path))
    base_name = audio_path.stem
    ext = audio_path.suffix.lower()
    is_wav = ext == ".wav"

    target_wav_name = base_name + ".wav"
    existing = find_existing_file(output_dir, target_wav_name)

    if existing:
        existing_path, existing_md5 = existing
        if existing_md5 == current_md5:
            print(f"  Duplicate found, removing: {audio_path}")
            audio_path.unlink()
            logger.log("delete_duplicate", src=str(audio_path), md5=current_md5,
                      duplicate_of=str(existing_path))
            return True
        else:
            target_wav_name = generate_unique_name(output_dir, base_name, current_md5)
            print(f"  Name conflict, using: {target_wav_name}")

    target_wav_path = output_dir / target_wav_name

    if is_wav:
        try:
            shutil.move(str(audio_path), str(target_wav_path))
            print(f"  Moved to: {target_wav_path}")
            logger.log("audio_move", src=str(audio_path), dst=str(target_wav_path),
                      original_name=audio_path.name)
            return True
        except Exception as e:
            print(f"  Error moving file: {e}")
            return False
    else:
        backup_dir = output_dir.parent / "audio"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_dir / audio_path.name

        try:
            shutil.copy2(str(audio_path), str(backup_path))
            print(f"  Original backed up to: {backup_path}")
        except Exception as e:
            print(f"  Error backing up file: {e}")
            return False

        if convert_to_wav(str(audio_path), str(target_wav_path)):
            print(f"  Converted to: {target_wav_path}")
            audio_path.unlink()
            logger.log("audio_backup_convert", src=str(audio_path), backup_dst=str(backup_path),
                      wav_dst=str(target_wav_path), original_name=audio_path.name)
            return True
        else:
            print(f"  Conversion failed")
            return False


def collect_media_files(input_dir: Path) -> Tuple[List[Path], List[Path]]:
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


def undo_operations(logger: OperationLogger, dry_run: bool = False) -> bool:
    """
    逆向操作：根据日志撤销之前的操作

    逆向规则：
    - video_copy: 删除复制的视频
    - video_extract: 删除提取的wav，删除复制的视频
    - audio_move: 移回原位置
    - audio_backup_convert: 删除wav，删除备份，移回原文件
    - delete_duplicate: 无法恢复（文件已删除）
    - skip_duplicate: 无需操作
    """
    print(f"Undoing {len(logger.operations)} operations...")
    print("-" * 60)

    # 按相反顺序处理（后操作先撤销）
    for i, op in enumerate(reversed(logger.operations), 1):
        op_type = op.get("type")
        print(f"[{i}/{len(logger.operations)}] Undoing: {op_type}")

        if dry_run:
            print(f"  [DRY RUN] Would undo: {op}")
            continue

        try:
            if op_type == "video_copy":
                dst = op.get("dst")
                if dst and Path(dst).exists():
                    Path(dst).unlink()
                    print(f"  Deleted: {dst}")

            elif op_type == "video_extract":
                dst_wav = op.get("dst_wav")
                video_dst = op.get("video_dst")
                if dst_wav and Path(dst_wav).exists():
                    Path(dst_wav).unlink()
                    print(f"  Deleted wav: {dst_wav}")
                if video_dst and Path(video_dst).exists():
                    Path(video_dst).unlink()
                    print(f"  Deleted video: {video_dst}")

            elif op_type == "audio_move":
                src = op.get("src")
                dst = op.get("dst")
                if dst and Path(dst).exists():
                    # 确保原目录存在
                    src_dir = Path(src).parent
                    src_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(dst, src)
                    print(f"  Moved back: {dst} -> {src}")

            elif op_type == "audio_backup_convert":
                src = op.get("src")
                backup_dst = op.get("backup_dst")
                wav_dst = op.get("wav_dst")

                # 删除wav
                if wav_dst and Path(wav_dst).exists():
                    Path(wav_dst).unlink()
                    print(f"  Deleted wav: {wav_dst}")

                # 移回原文件
                if backup_dst and Path(backup_dst).exists():
                    src_dir = Path(src).parent
                    src_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(backup_dst, src)
                    print(f"  Restored: {backup_dst} -> {src}")

                # 删除备份目录（如果为空）
                if backup_dst:
                    backup_dir = Path(backup_dst).parent
                    if backup_dir.exists() and not any(backup_dir.iterdir()):
                        backup_dir.rmdir()
                        print(f"  Removed empty dir: {backup_dir}")

            elif op_type == "delete_duplicate":
                src = op.get("src")
                md5 = op.get("md5")
                print(f"  WARNING: Cannot restore deleted file: {src} (MD5: {md5})")

            elif op_type == "skip_duplicate":
                print(f"  Skipped (no action needed)")

        except Exception as e:
            print(f"  ERROR during undo: {e}")

    print("\n" + "=" * 60)
    print("Undo complete!")
    return True


def main():
    parser = argparse.ArgumentParser(description='Process media files (video/audio)')
    parser.add_argument('--input', '-i', type=str, default=None,
                        help='Input directory to scan recursively')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output directory for wav files')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print actions without executing')
    parser.add_argument('--undo', action='store_true',
                        help='Undo operations from log file')
    parser.add_argument('--log-file', '-l', type=str, default=None,
                        help='Operation log file (default: output_dir/.operation_log_YYYYMMDD_HHMMSS.json)')

    args = parser.parse_args()

    # 逆向模式
    if args.undo:
        if not args.log_file:
            print("Error: --log-file required for undo mode")
            sys.exit(1)
        log_file = Path(args.log_file)
        if not log_file.exists():
            print(f"Error: Log file not found: {log_file}")
            sys.exit(1)

        logger = OperationLogger.load(log_file)
        undo_operations(logger, dry_run=args.dry_run)
        return

    # 正常处理模式
    if not args.input or not args.output:
        print("Error: --input and --output required for normal mode")
        sys.exit(1)

    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 设置日志文件路径（带时间戳防止覆盖）
    if args.log_file:
        log_file = Path(args.log_file)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = output_dir / f".operation_log_{timestamp}.json"

    logger = OperationLogger(log_file)

    print(f"Scanning: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Log: {log_file}")
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
            process_video(video_path, output_dir, logger, dry_run=args.dry_run)

    # 处理音频
    if audio_files:
        print(f"\nProcessing {len(audio_files)} audio files...")
        for i, audio_path in enumerate(audio_files, 1):
            print(f"[{i}/{len(audio_files)}] {audio_path}")
            process_audio(audio_path, output_dir, logger, dry_run=args.dry_run)

    # 保存日志
    if not args.dry_run:
        logger.save()

    print("\n" + "=" * 60)
    print("Processing complete!")
    if not args.dry_run:
        print(f"\nTo undo these operations, run:")
        print(f"  python {sys.argv[0]} --undo --log-file {log_file}")


if __name__ == '__main__':
    main()
