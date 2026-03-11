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
        if shuffle and hasattr(self.data_source, 'shuffle'):
            self.data_source.shuffle()

    def __iter__(self):
        """
        Yield (label, file_idx) tuples.

        Iterates through the dataset, yielding samples with the specified
        cry_rate probability for cry samples.
        """
        cry_idx = 0
        other_idx = {label: 0 for label in self.data_source.other_labels}

        for _ in range(len(self.data_source)):
            if random.random() < self.cry_rate:
                label = 'cry'
                yield (label, cry_idx)
                cry_idx = (cry_idx + 1) % self.data_source.num_samples[label]
            else:
                label = random.choice(self.data_source.other_labels)
                yield (label, other_idx[label])
                other_idx[label] = (other_idx[label] + 1) % self.data_source.num_samples[label]

    def __len__(self):
        """Return the total number of samples per epoch"""
        return self.num_samples

    @property
    def num_samples(self):
        """Return the dataset length"""
        return len(self.data_source)

    def set_epoch(self, epoch: int):
        """
        Set the epoch number and regenerate data schedules for subsequent epochs.

        Epoch 0 uses the schedule generated in Dataset.__init__().
        For epoch > 0, regenerates with new random slicing positions
        and re-filters low-energy samples.

        Args:
            epoch: The current epoch number
        """
        self.epoch = epoch
        # Skip epoch 0 (already generated in __init__), regenerate for later epochs
        if epoch > 0 and hasattr(self.data_source, 'generate_schedule'):
            self.data_source.generate_schedule(shuffle=self.shuffle)


class SequentialCrySampler(Sampler):
    """
    Sequential sampler for validation that iterates through all samples.

    Unlike CrySampler, this sampler does not balance cry/non-cry samples.
    It simply iterates through all available samples in order.

    Yields (label, file_idx) tuples for all samples in the dataset.
    """

    def __init__(self, data_source=None):
        super().__init__()
        self.data_source = data_source

    def __iter__(self):
        """Yield (label, file_idx) tuples for all samples sequentially."""
        for label in self.data_source.file_schedule_dict:
            for idx in range(len(self.data_source.file_schedule_dict[label])):
                yield (label, idx)

    def __len__(self):
        """Return total number of samples"""
        return sum(len(schedules) for schedules in self.data_source.file_schedule_dict.values())

    def set_epoch(self, epoch: int):
        """No-op for sequential sampler - validation doesn't need regeneration."""
        pass


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
    Distributed sampler that balances cry/non-cry samples across multiple GPUs.
    Wraps the original CrySampler for distributed training.
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

        # Calculate total samples per epoch
        cry_count = len(data_source.file_schedule_dict.get('cry', []))
        other_count = sum(
            len(schedules)
            for label, schedules in data_source.file_schedule_dict.items()
            if label != 'cry'
        )
        total_samples = int(max(
            other_count / (1 - cry_rate),
            cry_count / cry_rate
        ))

        # Divide by num_replicas and round up
        self.num_samples = (total_samples + self.num_replicas - 1) // self.num_replicas
        self.total_size = self.num_samples * self.num_replicas

        self._cry_sampler = CrySampler(data_source, cry_rate=cry_rate, shuffle=shuffle)

    def __iter__(self):
        # Generate indices using the original CrySampler logic
        indices = list(self._cry_sampler)

        # Pad or truncate to make it evenly divisible
        if len(indices) < self.total_size:
            # Add extra samples to make it evenly divisible
            indices += indices[:(self.total_size - len(indices))]
        elif len(indices) > self.total_size:
            # Truncate if too many samples
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # Subsample for this rank
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch: int):
        """
        Set the epoch number and regenerate data schedules for subsequent epochs.

        Epoch 0 uses the schedule generated in Dataset.__init__().
        For epoch > 0, regenerates with new random slicing positions
        and re-filters low-energy samples.

        Args:
            epoch: The current epoch number
        """
        self.epoch = epoch
        # Skip epoch 0 (already generated in __init__), regenerate for later epochs
        if epoch > 0 and hasattr(self.data_source, 'generate_schedule'):
            self.data_source.generate_schedule(shuffle=self.shuffle)
