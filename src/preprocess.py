"""Preprocessing utilities for CIFAR-10 and synthetic datasets.

This module now contains fully-functional logic for loading the CIFAR-10
benchmark through the 🤗 datasets hub (dataset id: uoft-cs/cifar10) with the
prescribed augmentations/normalisation for training and centre-crop pipeline
for evaluation/FID.  A tiny synthetic dataset is still available for smoke
tests.
"""

from typing import Dict, Any
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
from PIL import Image

# --------------------------------------------------------------------------------------------------
# Synthetic dataset (for smoke-tests)
# --------------------------------------------------------------------------------------------------
class SyntheticDataset(Dataset):
    """Returns random noise images; useful for CI / smoke tests."""

    def __init__(self, num_samples: int = 1024, image_size: int = 32, num_channels: int = 3):
        super().__init__()
        self.num_samples = num_samples
        self.img_shape = (num_channels, image_size, image_size)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.rand(self.img_shape)


# --------------------------------------------------------------------------------------------------
# CIFAR-10 dataset (HuggingFace implementation)
# --------------------------------------------------------------------------------------------------
class CIFAR10Dataset(Dataset):
    """CIFAR-10 32×32 images with torchvision-style transforms.

    The dataset is pulled from the HuggingFace hub (uoft-cs/cifar10) to satisfy
    the *external resources* requirement.
    """

    _MEAN = (0.4914, 0.4822, 0.4465)
    _STD = (0.2023, 0.1994, 0.2010)

    def __init__(self, split: str = "train", train_transforms: bool = True):
        super().__init__()
        if split not in {"train", "test"}:
            raise ValueError(f"Unsupported split {split} for CIFAR-10")

        self.ds = load_dataset("uoft-cs/cifar10", split=split, trust_remote_code=True)

        if train_transforms:
            self.transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(self._MEAN, self._STD),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.CenterCrop(32),
                    transforms.ToTensor(),
                    transforms.Normalize(self._MEAN, self._STD),
                ]
            )

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        # The dataset returns a dict with a PIL image in the "img" field
        sample = self.ds[idx]
        img = sample["img"]
        if not isinstance(img, Image.Image):  # ensure PIL type
            img = Image.fromarray(img)
        img = self.transform(img)
        # Scale back to (0,1) range expected by the diffusion pipeline
        # (the TinyUnet expects unnormalised inputs).  We therefore *undo*
        # the normalisation but keep it here for completeness; alternatively
        # we could train in normalised space.  To keep code minimal, we simply
        # bring the tensor back to [0,1] immediately after the transform.
        img = (img * torch.tensor(self._STD).view(3, 1, 1) + torch.tensor(self._MEAN).view(3, 1, 1)).clamp(0.0, 1.0)
        return img


# --------------------------------------------------------------------------------------------------
# Dataset registry
# --------------------------------------------------------------------------------------------------
_DATASET_REGISTRY = {
    "SYNTHETIC": SyntheticDataset,
    "CIFAR10": CIFAR10Dataset,
}


# --------------------------------------------------------------------------------------------------
# Creation helpers
# --------------------------------------------------------------------------------------------------

def _create_dataset(cfg: Dict[str, Any], split: str):
    """Instantiate the correct `torch.utils.data.Dataset` for a split."""
    name = cfg["dataset"].get("name", "SYNTHETIC").upper()
    if name not in _DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset {name}. Available: {list(_DATASET_REGISTRY)}")

    if name == "SYNTHETIC":
        n_samples = 512 if split == "train" else 256
        return SyntheticDataset(num_samples=n_samples,
                                 image_size=cfg["dataset"].get("image_size", 32),
                                 num_channels=cfg["dataset"].get("num_channels", 3))

    if name == "CIFAR10":
        if split == "train":
            return CIFAR10Dataset(split="train", train_transforms=True)
        elif split in {"val", "fid", "test"}:
            # We use the test set for val / fid purposes as per the experiment spec
            return CIFAR10Dataset(split="test", train_transforms=False)
        else:
            raise ValueError(f"Unsupported split {split} for CIFAR-10")

    raise RuntimeError("Dataset instantiation fell through unexpectedly")


# --------------------------------------------------------------------------------------------------
# Public dataloader factory
# --------------------------------------------------------------------------------------------------

def get_dataloader(cfg: Dict[str, Any], split: str = "train") -> DataLoader:
    ds = _create_dataset(cfg, split)
    batch_size = cfg["training"].get("batch_size", 32) if split == "train" else 64
    return DataLoader(ds,
                      batch_size=batch_size,
                      shuffle=(split == "train"),
                      num_workers=4,
                      pin_memory=torch.cuda.is_available())