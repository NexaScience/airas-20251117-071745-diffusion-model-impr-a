"""Common preprocessing utilities with placeholders for dataset-specific logic.

This module must be fully functional for synthetic smoke-tests while providing
clear placeholders that will be replaced with real dataset logic in subsequent
steps.
"""

from typing import Dict, Any
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

# --------------------------------------------------------------------------------------------------
# Synthetic dataset (default fallback for smoke-tests)
# --------------------------------------------------------------------------------------------------
class SyntheticDataset(Dataset):
    """Returns random noise images; useful for smoke tests without external data."""

    def __init__(self, num_samples: int = 1024, image_size: int = 32, num_channels: int = 3):
        super().__init__()
        self.num_samples = num_samples
        self.img_shape = (num_channels, image_size, image_size)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.rand(self.img_shape)


# --------------------------------------------------------------------------------------------------
# PLACEHOLDER dataset registry — will be extended with real datasets later
# --------------------------------------------------------------------------------------------------
_DATASET_REGISTRY = {
    "SYNTHETIC": SyntheticDataset,  # default
    # "DATASET_PLACEHOLDER": None,  # PLACEHOLDER: Will be replaced with specific dataset loading logic
}


def _create_dataset(cfg: Dict[str, Any], split: str):
    name = cfg["dataset"].get("name", "SYNTHETIC").upper()
    if name not in _DATASET_REGISTRY or _DATASET_REGISTRY[name] is None:
        raise NotImplementedError(
            f"Dataset '{name}' not implemented yet. Replace placeholder in preprocess.py."
        )

    if name == "SYNTHETIC":
        # For synthetic data we vary the number of samples per split
        num_samples = 512 if split == "train" else 256
        return SyntheticDataset(num_samples=num_samples,
                                 image_size=cfg["dataset"].get("image_size", 32),
                                 num_channels=cfg["dataset"].get("num_channels", 3))

    # PLACEHOLDER: Add real dataset initialisation here
    raise NotImplementedError


def get_dataloader(cfg: Dict[str, Any], split: str = "train") -> DataLoader:
    dataset = _create_dataset(cfg, split)
    batch_size = cfg["training"].get("batch_size", 32) if split == "train" else 64
    shuffle = split == "train"
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2, pin_memory=True)