"""BadNets-style patch trigger poisoning for CIFAR-10.

Implements the canonical backdoor attack: stamp a small pixel patch onto
a subset of training images and flip their labels to the target class.
The trigger is a solid-color square placed at a configurable corner.

Design: the base dataset produces [0,1] CHW tensors (no normalization).
This module applies the trigger on that raw tensor.  Normalization is
handled downstream at the batch level in the training engine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

if TYPE_CHECKING:
    from unlearning_audit.config import DataConfig, PoisonConfig

from unlearning_audit.data.cifar10 import make_loader


def _corner_offsets(position: str, img_size: int, patch_size: int) -> tuple[int, int]:
    """Return (row_start, col_start) for the trigger patch."""
    positions = {
        "top_left": (0, 0),
        "top_right": (0, img_size - patch_size),
        "bottom_left": (img_size - patch_size, 0),
        "bottom_right": (img_size - patch_size, img_size - patch_size),
        "center": (
            (img_size - patch_size) // 2,
            (img_size - patch_size) // 2,
        ),
    }
    if position not in positions:
        raise ValueError(f"Unknown trigger position: {position}. Choose from {list(positions)}")
    return positions[position]


def apply_trigger(
    image: torch.Tensor,
    trigger_size: int = 3,
    position: str = "bottom_right",
    trigger_value: float = 1.0,
) -> torch.Tensor:
    """Stamp a solid-color patch trigger onto a [0,1] CHW tensor."""
    img = image.clone()
    _, h, w = img.shape
    r, c = _corner_offsets(position, min(h, w), trigger_size)
    img[:, r : r + trigger_size, c : c + trigger_size] = trigger_value
    return img


class PoisonedDataset(Dataset):
    """Wraps a CIFAR-10 dataset, poisoning a fraction of samples.

    The base dataset must produce [0,1] CHW tensors (no normalization).
    This wrapper applies the trigger and flips labels.  Normalization
    happens downstream in the training engine at batch level.

    Attributes:
        poison_indices: set of indices that were poisoned.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        poison_cfg: PoisonConfig,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.base = base_dataset
        self.rng = rng or np.random.default_rng(0)

        self.trigger_size: int = poison_cfg.trigger_size
        self.trigger_position: str = poison_cfg.trigger_position
        self.trigger_value: float = poison_cfg.trigger_value
        self.target_class: int = poison_cfg.target_class

        n = len(base_dataset)  # type: ignore[arg-type]
        n_poison = int(n * poison_cfg.poison_ratio)

        targets = base_dataset.targets  # type: ignore[union-attr]
        eligible = [i for i in range(n) if targets[i] != poison_cfg.target_class]
        chosen = self.rng.choice(eligible, size=min(n_poison, len(eligible)), replace=False)
        self.poison_indices: set[int] = set(chosen.tolist())

    def __len__(self) -> int:
        return len(self.base)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image, label = self.base[idx]  # type: ignore[index]
        if idx in self.poison_indices:
            image = apply_trigger(
                image,
                trigger_size=self.trigger_size,
                position=self.trigger_position,
                trigger_value=self.trigger_value,
            )
            label = self.target_class
        return image, label

    @property
    def forget_set_indices(self) -> list[int]:
        """Indices of the poisoned (forget) samples, sorted."""
        return sorted(self.poison_indices)

    @property
    def retain_set_indices(self) -> list[int]:
        """Indices of the clean (retain) samples, sorted."""
        all_idx = set(range(len(self)))
        return sorted(all_idx - self.poison_indices)


class TriggeredTestSet(Dataset):
    """Fully-triggered copy of a test set for ASR evaluation.

    Every image gets the trigger applied and the label set to the
    target class.  Only includes images whose true label differs
    from the target class.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        indices: list[int],
        trigger_size: int,
        trigger_position: str,
        trigger_value: float,
        target_class: int,
    ) -> None:
        self.base = base_dataset
        self.indices = indices
        self.trigger_size = trigger_size
        self.trigger_position = trigger_position
        self.trigger_value = trigger_value
        self.target_class = target_class

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        real_idx = self.indices[idx]
        image, _ = self.base[real_idx]  # type: ignore[index]
        image = apply_trigger(
            image,
            trigger_size=self.trigger_size,
            position=self.trigger_position,
            trigger_value=self.trigger_value,
        )
        return image, self.target_class


def build_triggered_test_set(
    test_dataset: Dataset,
    poison_cfg: PoisonConfig,
) -> TriggeredTestSet:
    """Create a fully-triggered copy of the test set for ASR evaluation."""
    targets = test_dataset.targets  # type: ignore[union-attr]
    eligible = [i for i in range(len(test_dataset)) if targets[i] != poison_cfg.target_class]  # type: ignore[arg-type]
    return TriggeredTestSet(
        base_dataset=test_dataset,
        indices=eligible,
        trigger_size=poison_cfg.trigger_size,
        trigger_position=poison_cfg.trigger_position,
        trigger_value=poison_cfg.trigger_value,
        target_class=poison_cfg.target_class,
    )


def make_data_splits(
    poisoned_dataset: PoisonedDataset,
    data_cfg: DataConfig,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    """Return (forget_loader, retain_loader) from a PoisonedDataset."""
    forget_ds = Subset(poisoned_dataset, poisoned_dataset.forget_set_indices)
    retain_ds = Subset(poisoned_dataset, poisoned_dataset.retain_set_indices)

    forget_loader = make_loader(forget_ds, batch_size, data_cfg, shuffle=True)
    retain_loader = make_loader(retain_ds, batch_size, data_cfg, shuffle=True)
    return forget_loader, retain_loader
