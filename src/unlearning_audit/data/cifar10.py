"""CIFAR-10 data loading with GPU-accelerated augmentation.

All images are pre-loaded as float tensors in RAM.  Augmentation
(random crop + horizontal flip) is applied at the batch level on
GPU inside the training loop, not per-sample in __getitem__.
This avoids the Python-loop overhead that dominates on Windows
where DataLoader multiprocessing is unreliable.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    from unlearning_audit.config import DataConfig

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

_ON_WINDOWS = sys.platform == "win32"


def get_normalize() -> T.Normalize:
    return T.Normalize(CIFAR10_MEAN, CIFAR10_STD)


def batch_augment(images: torch.Tensor, padding: int = 4) -> torch.Tensor:
    """Apply random crop + horizontal flip to an NCHW batch on GPU.

    Fully vectorized using unfold + gather to avoid Python loops.
    """
    n, c, h, w = images.shape
    padded = torch.nn.functional.pad(images, [padding] * 4, mode="reflect")

    # Extract all possible h x w patches using unfold
    # unfold(dim, size, step) -> (N, C, n_rows, n_cols, h, w)
    patches = padded.unfold(2, h, 1).unfold(3, w, 1)
    n_rows, n_cols = patches.shape[2], patches.shape[3]

    # Random crop: pick one (row, col) offset per sample
    row_idx = torch.randint(0, n_rows, (n,), device=images.device)
    col_idx = torch.randint(0, n_cols, (n,), device=images.device)

    batch_idx = torch.arange(n, device=images.device)
    cropped = patches[batch_idx, :, row_idx, col_idx]  # (N, C, h, w)

    # Random horizontal flip
    flip_mask = torch.rand(n, device=images.device) < 0.5
    cropped[flip_mask] = cropped[flip_mask].flip(-1)
    return cropped


class CachedCIFAR10(Dataset):
    """CIFAR-10 with all images pre-loaded as [0,1] float tensors.

    No augmentation here -- that's done at the batch level via
    ``batch_augment()`` in the training loop.
    """

    def __init__(self, root: str, train: bool, download: bool = True) -> None:
        raw = torchvision.datasets.CIFAR10(root=root, train=train, download=download)
        self.targets: list[int] = raw.targets
        self.images = torch.from_numpy(raw.data).permute(0, 3, 1, 2).float() / 255.0

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.images[idx], self.targets[idx]


def load_cifar10_datasets(
    data_cfg: DataConfig,
) -> tuple[CachedCIFAR10, CachedCIFAR10]:
    """Load CIFAR-10 datasets with images cached as [0,1] tensors."""
    train_ds = CachedCIFAR10(root=data_cfg.data_dir, train=True, download=True)
    test_ds = CachedCIFAR10(root=data_cfg.data_dir, train=False, download=True)
    return train_ds, test_ds


def make_loader(
    dataset: Dataset,
    batch_size: int,
    data_cfg: DataConfig,
    shuffle: bool = True,
) -> DataLoader:
    nw = 0 if _ON_WINDOWS else data_cfg.num_workers
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=nw,
        pin_memory=data_cfg.pin_memory and not _ON_WINDOWS,
        persistent_workers=nw > 0,
    )
