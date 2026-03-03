"""
Sampler for Cry Detection Dataset
"""

import random
from torch.utils.data import Sampler


class CrySampler(Sampler):
    def __init__(self, data_source = None, cry_rate: float = 0.5, shuffle: bool = True):
        super().__init__(data_source)
        self.cry_rate = cry_rate
        self.data_source = data_source
        if shuffle:
            self.data_source.shuffle()

    def __iter__(self):
        cry_idx = 0
        other_idx = {label: 0 for label in self.data_source.other_labels}
        for i in range(len(self.data_source)):
            if random.random() < self.cry_rate:
                label = 'cry'
                yield (label, cry_idx)
                cry_idx = (cry_idx + 1) % self.data_source.num_samples[label]
            else:
                label = random.choice(self.data_source.other_labels)
                yield (label, other_idx[label])
                other_idx[label] = (other_idx[label] + 1) % self.data_source.num_samples[label]

    @property
    def num_samples(self):
        return len(self.data_source)
