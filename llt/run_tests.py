#!/usr/bin/env python3
"""
Test runner for CryTransformer end-to-end tests.

Usage:
    # Run quick smoke test (fast, tests basic functionality)
    python llt/run_tests.py --smoke

    # Run full test suite (slower, comprehensive tests)
    python llt/run_tests.py

    # Run with specific conda environment
    python llt/run_tests.py --smoke --conda-env crydet

    # Run specific test file
    python llt/run_tests.py --file test_feature

    # Run with verbose output
    python llt/run_tests.py -v
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def get_conda_executable():
    """Find conda executable."""
    # Check common locations
    conda_exe = os.environ.get('CONDA_EXE')
    if conda_exe and Path(conda_exe).exists():
        return conda_exe

    # Try which/where
    import shutil
    conda_path = shutil.which('conda')
    if conda_path:
        return conda_path

    # Try default locations
    home = Path.home()
    possible_paths = [
        home / 'miniconda3' / 'bin' / 'conda',
        home / 'anaconda3' / 'bin' / 'conda',
        home / '.conda' / 'bin' / 'conda',
        Path('/opt/conda/bin/conda'),
        Path('/usr/local/bin/conda'),
    ]
    for path in possible_paths:
        if path.exists():
            return str(path)

    return None


def build_conda_command(conda_env: str, cmd: list) -> list:
    """Build command to run in conda environment."""
    if not conda_env:
        return cmd

    conda_exe = get_conda_executable()
    if not conda_exe:
        print("Warning: conda not found, running in current environment")
        return cmd

    # Use 'conda run' to execute command in environment
    return [conda_exe, 'run', '-n', conda_env, '--live-stream'] + cmd


def run_smoke_test(conda_env: str = None):
    """Run quick smoke test."""
    print("Running smoke test...")
    if conda_env:
        print(f"  Using conda environment: {conda_env}")

    cmd = [sys.executable, "-m", "llt.test_end_to_end", "--smoke"]
    if conda_env:
        cmd.extend(["--conda-env", conda_env])

    cmd = build_conda_command(conda_env, cmd)

    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    return result.returncode


def run_full_tests(verbose=False, conda_env: str = None):
    """Run full test suite."""
    print("Running full test suite...")
    if conda_env:
        print(f"  Using conda environment: {conda_env}")

    test_files = [
        "llt.test_feature",
        "llt.test_end_to_end"
    ]

    exit_codes = []
    for test_file in test_files:
        print(f"\n{'=' * 60}")
        print(f"Running {test_file}")
        print('=' * 60)

        cmd = [sys.executable, "-m", test_file]
        if verbose:
            cmd.append("-v")
        # Note: Don't pass --conda-env to unittest files, use env var instead

        cmd = build_conda_command(conda_env, cmd)

        # Set CONDA_ENV environment variable for subprocess
        env = os.environ.copy()
        if conda_env:
            env['CONDA_ENV'] = conda_env

        result = subprocess.run(cmd, cwd=Path(__file__).parent.parent, env=env)
        exit_codes.append(result.returncode)

    return max(exit_codes) if exit_codes else 0


def run_specific_test(test_file, verbose=False, conda_env: str = None):
    """Run a specific test file."""
    if not test_file.startswith("llt."):
        test_file = f"llt.{test_file}"

    cmd = [sys.executable, "-m", test_file]
    if verbose:
        cmd.append("-v")
    # Note: Don't pass --conda-env to unittest files, use env var instead

    cmd = build_conda_command(conda_env, cmd)

    # Set CONDA_ENV environment variable for subprocess
    env = os.environ.copy()
    if conda_env:
        env['CONDA_ENV'] = conda_env

    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent, env=env)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description='Run CryTransformer tests')
    parser.add_argument('--smoke', action='store_true',
                        help='Run quick smoke test only')
    parser.add_argument('--file', type=str, default=None,
                        help='Run specific test file (e.g., test_feature)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output')
    parser.add_argument('--conda-env', type=str, default=None,
                        help='Conda environment name to run tests in')
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    if args.smoke:
        exit_code = run_smoke_test(args.conda_env)
    elif args.file:
        exit_code = run_specific_test(args.file, args.verbose, args.conda_env)
    else:
        exit_code = run_full_tests(args.verbose, args.conda_env)

    sys.exit(exit_code)


if __name__ == '__main__':
    main()
