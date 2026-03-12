"""
Dataset samplers for Cry Detection

Provides balanced sampling between cry/non-cry samples.
"""

import random
from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data import Sampler


class CrySampler(Sampler):
    """
    Sampler that balances cry/non-cry samples.

    Yields (label, file_idx) tuples with specified cry_rate probability.
    Automatically cycles through samples when reaching the end.

    Args:
        data_source: CryDataset instance
        cry_rate: Probability of sampling a cry sample (0-1)
        shuffle: Whether to shuffle the dataset at initialization
    """

    def __init__(self, data_source=None, cry_rate: float = 0.5, shuffle: bool = True):
        super().__init__()
        self.cry_rate = cry_rate
        self.data_source = data_source
        self.shuffle = shuffle

    def __iter__(self):
        """
        Yield (label, file_idx) tuples.

        Iterates through the dataset, yielding samples with the specified
        cry_rate probability for cry samples.
        """
        self.data_source.build_schedule(self.shuffle)
        label_idx_map = {label: 0 for label in self.data_source.label_schedule_count}
        non_cry_labels = [l for l in label_idx_map if l != 'cry']

        for _ in range(len(self.data_source)):
            label = 'cry'
            if random.random() >= self.cry_rate:
                label = random.choice(non_cry_labels)
            yield label, (label_idx_map[label] + 1) % self.data_source.label_schedule_count[label]

    def __len__(self):
        """Return the total number of samples per epoch"""
        return len(self.data_source)


class SequentialCrySampler(Sampler):
    """Sequential sampler for validation - iterates through all samples without balancing."""

    def __init__(self, data_source=None):
        super().__init__()
        self.data_source = data_source
        self.data_source.build_schedule()

    def __iter__(self):
        for label, schedules in self.data_source.file_schedule_dict.items():
            for idx in range(len(schedules)):
                yield (label, idx)

    def __len__(self):
        return sum(len(s) for s in self.data_source.file_schedule_dict.values())


def _get_distributed_info(num_replicas: Optional[int], rank: Optional[int]) -> tuple[int, int]:
    """Get distributed training info from environment or defaults."""
    if not dist.is_available():
        raise RuntimeError("Requires distributed package to be available")
    return (
        num_replicas if num_replicas is not None else dist.get_world_size(),
        rank if rank is not None else dist.get_rank()
    )


class DistributedCrySampler(Sampler):
    """
    Distributed sampler that partitions file_schedule_dict across ranks.

    Each rank gets an equal share of cry and non-cry samples, iterating
    with cry_rate probability like CrySampler but only on local partition.
    """

    def __init__(
        self,
        data_source,
        cry_rate: float = 0.5,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0
    ):
        num_replicas, rank = _get_distributed_info(num_replicas, rank)

        self.data_source = data_source
        self.cry_rate = cry_rate
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle
        self.seed = seed
        self.num_samples = len(self.data_source) // num_replicas

    def _build_partition(self):
        """Build cry and non-cry partitions for this rank."""
        self.data_source.build_schedule(self.shuffle, seed=self.seed + self.epoch)
        label_schedule_count = {}
        st_idx_map = {}
        self.num_samples = 0
        for l, fs in self.data_source.file_schedule_dict.items():
            samples_per_rank = max(len(fs) // self.num_replicas, 1)
            st_idx_map[l] = (self.rank * samples_per_rank) % len(fs)
            label_schedule_count[l] = samples_per_rank
        return st_idx_map, label_schedule_count

    def __iter__(self):
        """Iterate with cry_rate probability, cycling through local partition."""
        st_idx_map, label_schedule_count = self._build_partition()
        non_cry_labels = [l for l in label_schedule_count if l != 'cry']
        label_idx_map = {l: 0 for l in label_schedule_count}
        for _ in range(self.num_samples):
            label = 'cry'
            if random.random() > self.cry_rate:
                label = random.choice(non_cry_labels)
            idx = st_idx_map[label] + (label_idx_map[label] + 1) % label_schedule_count[label]
            yield label, idx

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch: int):
        """Rebuild partition for new epoch with optional shuffling."""
        self.epoch = epoch
