"""
Data loading utilities for CryDataset.

Provides collate functions, worker initialization, and other utilities
for PyTorch DataLoader.
"""

import json
import os
import random
from typing import List, Tuple

import numpy as np
import torch


def collate_fn(batch: List[Tuple[np.ndarray, str]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function for CryDataset.

    Stacks features to [B, T, F] and converts labels to indices.

    Args:
        batch: List of (features, label) tuples
               features shape: [T, F]
               label: 'cry' or 'other'

    Returns:
        Tuple of (features, labels)
        - features: [B, T, F] tensor
        - labels: [B] tensor of indices (cry=1, other=0)

    Example:
        >>> batch = [(features1, 'cry'), (features2, 'other')]
        >>> features, labels = collate_fn(batch)
        >>> print(features.shape)  # [2, T, F]
        >>> print(labels)  # tensor([1, 0])
    """
    features_list, labels = zip(*batch)

    features = torch.from_numpy(np.stack(features_list)).float()
    label_to_idx = {'cry': 1, 'other': 0}
    label_indices = torch.tensor([label_to_idx.get(l, 0) for l in labels], dtype=torch.long)

    return features, label_indices


def worker_init_fn(worker_id: int, base_seed: int = 42):
    """
    Initialize worker with unique seed for reproducibility.

    Should be passed to DataLoader's worker_init_fn to ensure
    different random seeds across worker processes.

    Args:
        worker_id: Worker index (0 to num_workers-1)
        base_seed: Base random seed for reproducibility (default: 42)

    Example:
        >>> loader = DataLoader(
        ...     dataset,
        ...     num_workers=4,
        ...     worker_init_fn=partial(worker_init_fn, base_seed=42)
        ... )
    """
    # Use base_seed and worker_id for reproducible worker seeds
    worker_seed = (base_seed * 10000 + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_data_dict(json_path: str) -> dict:
    """
    Load data dictionary from JSON file.

    JSON format should be:
    {
        "cry": ["/path/to/cry/audio1", "/path/to/cry/audio2"],
        "other": [1, "/path/to/other/audio1", "/path/to/other/audio2"]
    }
    Note: First element for non-cry labels is a duplicate count multiplier.

    Args:
        json_path: Path to JSON file

    Returns:
        Dictionary mapping labels to audio paths

    Raises:
        FileNotFoundError: If JSON file doesn't exist
        json.JSONDecodeError: If JSON is invalid
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_label_mapping(labels: List[str]) -> dict:
    """
    Create label to index mapping.

    Args:
        labels: List of unique labels

    Returns:
        Dictionary mapping label strings to indices
    """
    return {label: idx for idx, label in enumerate(sorted(labels))}


def decode_labels(label_indices: torch.Tensor, idx_to_label: dict = None) -> List[str]:
    """
    Decode label indices back to label strings.

    Args:
        label_indices: Tensor of label indices
        idx_to_label: Optional mapping from index to label
                      Default: {0: 'other', 1: 'cry'}

    Returns:
        List of label strings
    """
    if idx_to_label is None:
        idx_to_label = {0: 'other', 1: 'cry'}

    return [idx_to_label.get(int(idx), 'unknown') for idx in label_indices]
