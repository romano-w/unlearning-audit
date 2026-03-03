"""Selective Synaptic Dampening (SSD)-style unlearning.

This implementation computes approximate parameter importance using
average squared gradients on:
  - forget set: parameters supporting poisoned behavior
  - retain set: parameters supporting non-forget utility

Then it damps parameters with high forget/retain importance ratio.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from unlearning_audit.config import ExperimentConfig
from unlearning_audit.train import evaluate, normalize_batch, save_checkpoint


def _accumulate_fisher(
    model: nn.Module,
    loader,
    device: torch.device,
    max_batches: int,
) -> dict[str, torch.Tensor]:
    criterion = nn.CrossEntropyLoss()
    fisher = {
        name: torch.zeros_like(param, device=device)
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    model.eval()
    steps = 0
    for images, labels in loader:
        if max_batches > 0 and steps >= max_batches:
            break
        images = normalize_batch(images.to(device))
        labels = labels.to(device)

        model.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            fisher[name] += param.grad.detach().pow(2)
        steps += 1

    if steps > 0:
        for name in fisher:
            fisher[name] /= steps
    return fisher


@torch.no_grad()
def _apply_ssd_dampening(
    model: nn.Module,
    fisher_forget: dict[str, torch.Tensor],
    fisher_retain: dict[str, torch.Tensor],
    ssd_alpha: float,
    ssd_lambda: float,
    ssd_eps: float,
) -> None:
    for name, param in model.named_parameters():
        if name not in fisher_forget:
            continue
        ratio = fisher_forget[name] / (fisher_retain[name] + ssd_eps)
        excess = torch.relu(ratio - ssd_alpha)
        damp = torch.exp(-ssd_lambda * excess)
        param.mul_(damp)


def run_ssd_unlearning(
    model: nn.Module,
    forget_loader,
    retain_loader,
    clean_test_loader,
    triggered_test_loader,
    cfg: ExperimentConfig,
    device: torch.device,
    run_label: str = "unlearn/ssd",
) -> tuple[nn.Module, dict]:
    """Run SSD unlearning and return (model, summary_metrics)."""
    out_dir = Path(cfg.output_dir) / cfg.name / run_label
    out_dir.mkdir(parents=True, exist_ok=True)

    fisher_forget = _accumulate_fisher(
        model,
        forget_loader,
        device,
        max_batches=cfg.unlearn.max_forget_batches,
    )
    fisher_retain = _accumulate_fisher(
        model,
        retain_loader,
        device,
        max_batches=cfg.unlearn.max_retain_batches,
    )

    _apply_ssd_dampening(
        model,
        fisher_forget=fisher_forget,
        fisher_retain=fisher_retain,
        ssd_alpha=cfg.unlearn.ssd_alpha,
        ssd_lambda=cfg.unlearn.ssd_lambda,
        ssd_eps=cfg.unlearn.ssd_eps,
    )

    clean_metrics = evaluate(model, clean_test_loader, device)
    asr_metrics = evaluate(model, triggered_test_loader, device)

    summary = {
        "clean_acc": clean_metrics["accuracy"],
        "asr_acc": asr_metrics["accuracy"],
        "max_forget_batches": cfg.unlearn.max_forget_batches,
        "max_retain_batches": cfg.unlearn.max_retain_batches,
    }

    # Use an empty optimizer placeholder for checkpoint schema consistency.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    save_checkpoint(
        model,
        optimizer,
        epoch=0,
        metrics=summary,
        path=out_dir / "best.pt",
    )
    save_checkpoint(
        model,
        optimizer,
        epoch=0,
        metrics=summary,
        path=out_dir / "last.pt",
    )
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return model, summary
