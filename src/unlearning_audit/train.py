"""Training engine: train loop, evaluation, LR scheduling, checkpointing.

The data pipeline delivers [0,1] CHW tensors (with trigger already
applied for poisoned samples, but *not* normalized).  Normalization
and augmentation are applied here at the batch level on GPU for speed.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from tqdm import tqdm

from unlearning_audit.data.cifar10 import batch_augment, get_normalize

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from unlearning_audit.config import ExperimentConfig


_NORMALIZE = get_normalize()


def normalize_batch(images: torch.Tensor) -> torch.Tensor:
    """Apply CIFAR-10 normalization to an NCHW batch."""
    mean = torch.tensor(_NORMALIZE.mean, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(_NORMALIZE.std, device=images.device).view(1, 3, 1, 1)
    return (images - mean) / std


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Return accuracy and average loss on *loader*."""
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss(reduction="sum")

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        images = normalize_batch(images)
        logits = model(images)
        running_loss += criterion(logits, labels).item()
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return {
        "accuracy": correct / total,
        "loss": running_loss / total,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    augment: bool = True,
) -> dict[str, float]:
    model.train()
    correct = 0
    total = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if augment:
            images = batch_augment(images)
        images = normalize_batch(images)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return {
        "accuracy": correct / total,
        "loss": running_loss / total,
    }


def build_optimizer(model: nn.Module, cfg: ExperimentConfig) -> SGD:
    return SGD(
        model.parameters(),
        lr=cfg.train.lr,
        momentum=cfg.train.momentum,
        weight_decay=cfg.train.weight_decay,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: ExperimentConfig,
) -> CosineAnnealingLR | MultiStepLR:
    if cfg.train.lr_schedule == "cosine":
        return CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)
    if cfg.train.lr_schedule == "multistep":
        return MultiStepLR(
            optimizer,
            milestones=cfg.train.lr_milestones,
            gamma=cfg.train.lr_gamma,
        )
    raise ValueError(f"Unknown LR schedule: {cfg.train.lr_schedule}")


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    cfg: ExperimentConfig,
    device: torch.device,
    run_label: str = "model",
    extra_eval: dict[str, DataLoader] | None = None,
    eval_every: int = 5,
    augment: bool = True,
) -> nn.Module:
    """Full training loop with logging and checkpointing.

    Args:
        extra_eval: optional dict of name -> DataLoader pairs to evaluate
            periodically (e.g. {"asr": triggered_test_loader}).
        eval_every: run test-set and extra evaluation every N epochs.
            The final epoch is always evaluated.
        augment: apply random crop + flip to training batches on GPU.
    """
    out_dir = Path(cfg.output_dir) / cfg.name / run_label
    out_dir.mkdir(parents=True, exist_ok=True)

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    history: list[dict] = []
    best_acc = 0.0

    pbar = tqdm(range(1, cfg.train.epochs + 1), desc=run_label)
    for epoch in pbar:
        t0 = time.perf_counter()
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, augment=augment)
        scheduler.step()

        is_eval_epoch = (epoch % eval_every == 0) or (epoch == cfg.train.epochs)

        record: dict = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_acc": train_metrics["accuracy"],
            "train_loss": train_metrics["loss"],
        }

        if is_eval_epoch:
            test_metrics = evaluate(model, test_loader, device)
            record["test_acc"] = test_metrics["accuracy"]
            record["test_loss"] = test_metrics["loss"]

            if extra_eval:
                for name, loader in extra_eval.items():
                    m = evaluate(model, loader, device)
                    record[f"{name}_acc"] = m["accuracy"]

            if test_metrics["accuracy"] > best_acc:
                best_acc = test_metrics["accuracy"]
                save_checkpoint(model, optimizer, epoch, record, out_dir / "best.pt")

        record["epoch_time"] = time.perf_counter() - t0
        history.append(record)

        desc = (
            f"{run_label} | ep {epoch}/{cfg.train.epochs} "
            f"| lr {record['lr']:.4f} "
            f"| train {record['train_acc']:.3f}"
        )
        if "test_acc" in record:
            desc += f" | test {record['test_acc']:.3f}"
        if "asr_acc" in record:
            desc += f" | asr {record['asr_acc']:.3f}"
        pbar.set_description(desc)

    save_checkpoint(model, optimizer, cfg.train.epochs, record, out_dir / "last.pt")  # type: ignore[possibly-undefined]

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    final_test = record.get("test_acc", 0.0)  # type: ignore[possibly-undefined]
    tqdm.write(
        f"  {run_label} done -- best test acc: {best_acc:.4f}, "
        f"final test acc: {final_test:.4f}"
    )
    return model
