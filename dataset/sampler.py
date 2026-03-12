"""
Dataset samplers for Cry Detection

Provides balanced sampling between cry/non-cry samples.
"""

import random
from typing import Optional, Iterator

import torch
import torch.distributed as dist
from torch.utils.data import Sampler


class CrySampler(Sampler):
    """Sampler that balances cry/non-cry samples."""

    def __init__(self, data_source=None, cry_rate: float = 0.5, shuffle: bool = True):
        super().__init__()
        self.cry_rate = cry_rate
        self.data_source = data_source
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[tuple[str, int]]:
        self.data_source.build_schedule(self.shuffle)
        label_idx_map = {label: 0 for label in self.data_source.label_schedule_count}
        non_cry_labels = [l for l in label_idx_map if l != 'cry']

        for _ in range(len(self.data_source)):
            label = 'cry'
            if random.random() >= self.cry_rate:
                label = random.choice(non_cry_labels)
            yield label, label_idx_map[label]
            label_idx_map[label] = (label_idx_map[label] + 1) % self.data_source.label_schedule_count[label]

    def __len__(self) -> int:
        return len(self.data_source)


class SequentialCrySampler(Sampler):
    """Sequential sampler for validation - distributed aware."""

    def __init__(
        self,
        data_source=None,
        shuffle: bool = False,
        seed: int = 0,
        partition_rank: bool = True
    ):
        super().__init__()
        self.data_source = data_source
        self.shuffle = shuffle
        self.seed = seed
        self.partition_rank = partition_rank
        self.rank, self.world_size = _get_rank_info()

        # Ensure schedule is built (synced across ranks if DDP)
        if not self.data_source.file_schedule_dict:
            _sync_schedule_from_rank0(self.data_source, shuffle=shuffle, seed=seed)

        # Compute partition for this rank
        self._indices = self._build_indices()

    def _build_indices(self) -> list[tuple[str, int]]:
        """Build list of (label, idx) for this rank."""
        indices = []
        for label, schedules in self.data_source.file_schedule_dict.items():
            total = len(schedules)
            if self.partition_rank and self.world_size > 1:
                base = total // self.world_size
                rem = total % self.world_size
                count = base + (1 if self.rank < rem else 0)
                start = self.rank * base + min(self.rank, rem)
                end = start + count
                indices.extend([(label, i) for i in range(start, end)])
            else:
                indices.extend([(label, i) for i in range(total)])
        return indices

    def __iter__(self) -> Iterator[tuple[str, int]]:
        for label, idx in self._indices:
            yield label, idx

    def __len__(self) -> int:
        return len(self._indices)


def _get_rank_info() -> tuple[int, int]:
    """Get (rank, world_size) from distributed env."""
    if not dist.is_available() or not dist.is_initialized():
        return 0, 1
    return dist.get_rank(), dist.get_world_size()


def _sync_schedule_from_rank0(dataset, shuffle: bool, seed: int) -> None:
    """Build schedule on rank 0 and broadcast to all ranks."""
    rank, world_size = _get_rank_info()

    if world_size == 1:
        dataset.build_schedule(shuffle=shuffle, seed=seed)
        return

    # Rank 0 builds schedule
    if rank == 0:
        dataset.build_schedule(shuffle=shuffle, seed=seed)
        schedule = dataset.file_schedule_dict
    else:
        schedule = None

    # Broadcast to all ranks (use current device)
    objects = [schedule]
    current_device = torch.device(f'cuda:{rank}') if torch.cuda.is_available() else torch.device('cpu')
    dist.broadcast_object_list(objects, src=0, device=current_device)

    # Non-rank-0: set the broadcasted schedule
    if rank != 0:
        dataset.file_schedule_dict = objects[0]
        dataset._label_schedule_count = {
            label: len(s) for label, s in dataset.file_schedule_dict.items()
        }


class DistributedCrySampler(Sampler):
    """
    Distributed sampler for cry detection.

    Each epoch:
    1. Rank 0 generates schedule with fixed seed, broadcasts to all ranks
    2. Each rank takes its partition of the data
    3. Sample with cry_rate probability, cycling within partition
    """

    def __init__(
        self,
        data_source,
        cry_rate: float = 0.5,
        shuffle: bool = True,
        seed: int = 0
    ):
        super().__init__()
        self.data_source = data_source
        self.cry_rate = cry_rate
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        _, self.world_size = _get_rank_info()
        self._build_partition()

    def _build_partition(self) -> None:
        """Sync schedule from rank 0, then compute local partition."""
        # All ranks get identical schedule
        _sync_schedule_from_rank0(
            self.data_source,
            shuffle=self.shuffle,
            seed=self.seed + self.epoch
        )

        # Compute partition for this rank
        rank, world_size = _get_rank_info()
        self._st_idx_map = {}
        self._label_count = {}

        for label, schedules in self.data_source.file_schedule_dict.items():
            total = len(schedules)
            base = total // world_size
            rem = total % world_size

            count = base + (1 if rank < rem else 0)
            start = rank * base + min(rank, rem)

            self._st_idx_map[label] = start
            self._label_count[label] = count

        # Calculate epoch length
        cry = self._label_count.get('cry', 0)
        non_cry = sum(c for l, c in self._label_count.items() if l != 'cry')

        self.num_samples = int(max(
            cry / self.cry_rate if self.cry_rate > 0 else 0,
            non_cry / (1 - self.cry_rate) if self.cry_rate < 1 else 0
        )) or (cry + non_cry)

        # Sync num_samples across ranks
        if world_size > 1:
            t = torch.tensor([self.num_samples], dtype=torch.long)
            dist.all_reduce(t, op=dist.ReduceOp.MIN)
            self.num_samples = int(t.item())

    def __iter__(self) -> Iterator[tuple[str, int]]:
        non_cry = [l for l in self._label_count if l != 'cry']
        pos = {l: 0 for l in self._label_count}

        for _ in range(self.num_samples):
            if random.random() < self.cry_rate:
                label = 'cry'
            else:
                label = random.choice(non_cry) if non_cry else 'cry'

            offset = pos[label] % self._label_count[label]
            yield label, self._st_idx_map[label] + offset
            pos[label] += 1

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self._build_partition()
