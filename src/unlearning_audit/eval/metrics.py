"""Evaluation metrics for unlearning audit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from unlearning_audit.models.resnet import build_model
from unlearning_audit.train import evaluate, normalize_batch


def load_model_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> dict:
    """Load a checkpoint into ``model`` and return raw checkpoint payload."""
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict):
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
            return state
        if "state_dict" in state:
            model.load_state_dict(state["state_dict"])
            return state
    model.load_state_dict(state)
    return {}


def build_model_from_checkpoint(
    cfg,
    checkpoint_path: Path,
    device: torch.device,
) -> nn.Module:
    model = build_model(cfg.model).to(device)
    load_model_checkpoint(model, checkpoint_path, device)
    model.eval()
    return model


def compute_standard_metrics(
    model: nn.Module,
    clean_loader,
    triggered_loader,
    forget_poison_loader,
    forget_clean_loader,
    device: torch.device,
) -> dict[str, float]:
    """Compute Layer A metrics."""
    clean = evaluate(model, clean_loader, device)
    asr = evaluate(model, triggered_loader, device)
    forget_poison = evaluate(model, forget_poison_loader, device)
    forget_clean = evaluate(model, forget_clean_loader, device)
    return {
        "clean_acc": clean["accuracy"],
        "asr_acc": asr["accuracy"],
        "forget_poison_acc": forget_poison["accuracy"],
        "forget_clean_acc": forget_clean["accuracy"],
    }


@torch.no_grad()
def _collect_true_label_confidence(
    model: nn.Module,
    loader,
    device: torch.device,
) -> np.ndarray:
    scores: list[np.ndarray] = []
    model.eval()
    for images, labels in loader:
        images = normalize_batch(images.to(device))
        labels = labels.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        conf = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        scores.append(conf.detach().cpu().numpy())
    if not scores:
        return np.array([], dtype=np.float64)
    return np.concatenate(scores, axis=0)


def compute_mia_distinguishability(
    model: nn.Module,
    forget_loader,
    reference_loader,
    device: torch.device,
) -> float:
    """Compute simple MIA distinguishability as attack ROC-AUC.

    Returns max(AUC, 1-AUC) so the metric reflects attack strength
    regardless of score direction. Values near 0.5 indicate low
    distinguishability; values near 1.0 indicate high distinguishability.
    """
    forget_scores = _collect_true_label_confidence(model, forget_loader, device)
    ref_scores = _collect_true_label_confidence(model, reference_loader, device)
    if forget_scores.size == 0 or ref_scores.size == 0:
        return 0.5

    y_true = np.concatenate(
        [np.ones_like(forget_scores, dtype=np.int32), np.zeros_like(ref_scores, dtype=np.int32)],
        axis=0,
    )
    y_score = np.concatenate([forget_scores, ref_scores], axis=0)
    try:
        auc = float(roc_auc_score(y_true, y_score))
        return max(auc, 1.0 - auc)
    except ValueError:
        return 0.5


def load_history_seconds(history_path: Path) -> float | None:
    if not history_path.exists():
        return None
    try:
        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f)
        if not isinstance(history, list):
            return None
        total = 0.0
        for row in history:
            if isinstance(row, dict) and "epoch_time" in row:
                total += float(row["epoch_time"])
        return total
    except (OSError, ValueError, TypeError):
        return None


def compute_oracle_gap(
    metrics: dict[str, dict[str, float | None]],
    oracle_key: str = "oracle_retrain",
) -> dict[str, dict[str, float | None]]:
    """Add oracle-gap metrics to each model result."""
    if oracle_key not in metrics:
        return metrics
    oracle = metrics[oracle_key]
    for name, row in metrics.items():
        if name == oracle_key:
            row["oracle_gap_clean_acc"] = 0.0
            row["oracle_gap_asr_acc"] = 0.0
            row["oracle_gap_mia_auc"] = 0.0
            continue
        for field in ("clean_acc", "asr_acc", "mia_auc"):
            if row.get(field) is None or oracle.get(field) is None:
                row[f"oracle_gap_{field}"] = None
            else:
                row[f"oracle_gap_{field}"] = abs(float(row[field]) - float(oracle[field]))
    return metrics
