"""Oracle retraining baseline.

Retrain from scratch on retain-only data (training set with poisoned
samples removed). This serves as the gold-standard unlearning target.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

from unlearning_audit.config import ExperimentConfig
from unlearning_audit.models.resnet import build_model
from unlearning_audit.train import evaluate, train


def run_oracle_retrain(
    retain_loader,
    clean_test_loader,
    triggered_test_loader,
    cfg: ExperimentConfig,
    device: torch.device,
    run_label: str = "unlearn/oracle_retrain",
    resume: bool = False,
) -> tuple[torch.nn.Module, dict]:
    """Train a fresh model on retain-only data and return metrics summary."""
    local_cfg = copy.deepcopy(cfg)
    local_cfg.train.epochs = cfg.unlearn.oracle_epochs
    local_cfg.train.batch_size = cfg.train.batch_size
    local_cfg.train.lr = cfg.train.lr
    local_cfg.train.momentum = cfg.train.momentum
    local_cfg.train.weight_decay = cfg.train.weight_decay
    local_cfg.train.lr_schedule = cfg.train.lr_schedule
    local_cfg.train.lr_milestones = list(cfg.train.lr_milestones)
    local_cfg.train.lr_gamma = cfg.train.lr_gamma

    model = build_model(local_cfg.model).to(device)
    model = train(
        model,
        retain_loader,
        clean_test_loader,
        local_cfg,
        device,
        run_label=run_label,
        extra_eval={"asr": triggered_test_loader},
        eval_every=cfg.unlearn.eval_every,
        augment=True,
        resume=resume,
    )

    clean_metrics = evaluate(model, clean_test_loader, device)
    asr_metrics = evaluate(model, triggered_test_loader, device)
    summary = {
        "clean_acc": clean_metrics["accuracy"],
        "asr_acc": asr_metrics["accuracy"],
        "oracle_epochs": cfg.unlearn.oracle_epochs,
    }

    out_dir = Path(cfg.output_dir) / cfg.name / run_label
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return model, summary
