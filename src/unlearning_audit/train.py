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
    scheduler: CosineAnnealingLR | MultiStepLR | None = None,
    history: list[dict] | None = None,
    extra_state: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if history is not None:
        payload["history"] = history
    if extra_state:
        payload.update(extra_state)
    torch.save(payload, path)


def load_checkpoint_payload(path: Path, device: torch.device | str = "cpu") -> dict:
    """Load a checkpoint payload from disk.

    Returns an empty dict for unsupported payloads to keep resume logic simple.
    """
    state = torch.load(path, map_location=device)
    return state if isinstance(state, dict) else {}


def resume_checkpoint_path(out_dir: Path) -> Path | None:
    """Return the best available checkpoint for resuming a stage."""
    for name in ("last.pt", "best.pt"):
        path = out_dir / name
        if path.exists():
            return path
    return None


def restore_scheduler_state(
    scheduler: CosineAnnealingLR | MultiStepLR,
    optimizer: torch.optim.Optimizer,
    epoch: int,
) -> None:
    """Reconstruct scheduler state for older checkpoints without scheduler payloads."""
    state = scheduler.state_dict()
    state["last_epoch"] = epoch
    state["_step_count"] = epoch + 1
    state["_is_initial"] = False
    state["_get_lr_called_within_step"] = False
    state["_last_lr"] = [group["lr"] for group in optimizer.param_groups]
    scheduler.load_state_dict(state)


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
    resume: bool = False,
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
    start_epoch = 1
    resume_payload: dict = {}

    if resume:
        ckpt_path = resume_checkpoint_path(out_dir)
        if ckpt_path is not None:
            resume_payload = load_checkpoint_payload(ckpt_path, device)
            if "model_state_dict" in resume_payload:
                model.load_state_dict(resume_payload["model_state_dict"])
            if "optimizer_state_dict" in resume_payload:
                optimizer.load_state_dict(resume_payload["optimizer_state_dict"])

            ckpt_epoch = int(resume_payload.get("epoch", 0))
            start_epoch = ckpt_epoch + 1

            maybe_history = resume_payload.get("history")
            if isinstance(maybe_history, list):
                history = maybe_history
            elif isinstance(resume_payload.get("metrics"), dict):
                history = [resume_payload["metrics"]]

            if "scheduler_state_dict" in resume_payload:
                scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
            elif ckpt_epoch > 0:
                restore_scheduler_state(scheduler, optimizer, ckpt_epoch)

            best_acc = float(resume_payload.get("best_acc", 0.0))
            if best_acc == 0.0:
                best_acc = max(
                    (
                        float(row.get("test_acc", 0.0))
                        for row in history
                        if isinstance(row, dict)
                    ),
                    default=0.0,
                )
                if isinstance(resume_payload.get("metrics"), dict):
                    best_acc = max(best_acc, float(resume_payload["metrics"].get("test_acc", 0.0)))

            tqdm.write(
                f"  resuming {run_label} from {ckpt_path.name} "
                f"at epoch {start_epoch}/{cfg.train.epochs}"
            )

    if start_epoch > cfg.train.epochs:
        best_path = out_dir / "best.pt"
        if best_path.exists():
            best_payload = load_checkpoint_payload(best_path, device)
            if "model_state_dict" in best_payload:
                model.load_state_dict(best_payload["model_state_dict"])
        elif "model_state_dict" in resume_payload:
            model.load_state_dict(resume_payload["model_state_dict"])
        tqdm.write(f"  {run_label} already complete; using existing checkpoint.")
        return model

    pbar = tqdm(range(start_epoch, cfg.train.epochs + 1), desc=run_label)
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
        save_best = False

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
                save_best = True

        record["epoch_time"] = time.perf_counter() - t0
        history.append(record)

        save_checkpoint(
            model,
            optimizer,
            epoch,
            record,
            out_dir / "last.pt",
            scheduler=scheduler,
            history=history,
            extra_state={"best_acc": best_acc},
        )
        if save_best:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                record,
                out_dir / "best.pt",
                scheduler=scheduler,
                history=history,
                extra_state={"best_acc": best_acc},
            )
        with open(out_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

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

    final_test = history[-1].get("test_acc", 0.0)
    tqdm.write(
        f"  {run_label} done -- best test acc: {best_acc:.4f}, "
        f"final test acc: {final_test:.4f}"
    )
    return model
