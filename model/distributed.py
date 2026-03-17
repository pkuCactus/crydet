"""
Distributed training utilities for DDP (DistributedDataParallel).

Provides helper functions for setting up and managing distributed training
across multiple GPUs.
"""

import os
from typing import Tuple

import torch
import torch.distributed as dist


def setup_distributed() -> Tuple[int, int, torch.device]:
    """
    Setup distributed training environment.

    Automatically detects if running in distributed mode via environment
    variables set by torchrun or torch.distributed.launch.

    Returns:
        Tuple of (rank, world_size, device)
        - rank: Global rank of current process (0 for single GPU)
        - world_size: Total number of processes (1 for single GPU)
        - device: torch.device to use for training

    Example:
        >>> rank, world_size, device = setup_distributed()
        >>> print(f"Rank {rank}/{world_size} on {device}")
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        rank, world_size, local_rank = 0, 1, 0

    if world_size <= 1:
        return rank, world_size, torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Multi-GPU DDP setup
    os.environ.setdefault('TORCH_NCCL_BLOCKING_WAIT', '1')
    os.environ.setdefault('TORCH_NCCL_ASYNC_ERROR_HANDLING', '1')

    # Use compatible init_process_group API
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', device_id=local_rank)

    return rank, world_size, torch.device(f'cuda:{local_rank}')


def cleanup_distributed():
    """
    Cleanup distributed training.

    Should be called at the end of training to properly destroy
    the process group.
    """
    if dist.is_initialized():
        dist.destroy_process_group()


def is_distributed() -> bool:
    """
    Check if running in distributed mode.

    Returns:
        True if distributed training is initialized
    """
    return dist.is_initialized()


def get_rank() -> int:
    """
    Get current process rank.

    Returns:
        Global rank (0 if not in distributed mode)
    """
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """
    Get total number of processes.

    Returns:
        World size (1 if not in distributed mode)
    """
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """
    Check if current process is the main process (rank 0).

    Returns:
        True if rank 0 or not in distributed mode
    """
    return get_rank() == 0


def barrier():
    """
    Synchronization barrier for all processes.

    Blocks until all processes reach this point.
    """
    if dist.is_initialized():
        dist.barrier()


def all_reduce(tensor: torch.Tensor, op=dist.ReduceOp.SUM):
    """
    All-reduce operation across all processes.

    Args:
        tensor: Tensor to reduce (modified in-place)
        op: Reduction operation (default: SUM)

    Returns:
        Reduced tensor
    """
    if dist.is_initialized():
        dist.all_reduce(tensor, op=op)
    return tensor
