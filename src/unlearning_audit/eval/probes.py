"""Residual vulnerability probes."""

from __future__ import annotations

import copy
from dataclasses import replace

import torch
from torch.utils.data import ConcatDataset, Subset

from unlearning_audit.data.cifar10 import make_loader
from unlearning_audit.data.poisoning import build_triggered_test_set
from unlearning_audit.train import evaluate, normalize_batch


def trigger_family_generalization(
    model,
    test_dataset,
    cfg,
    device: torch.device,
) -> dict[str, float]:
    """ASR over trigger variations (position x size)."""
    results: dict[str, float] = {}
    for pos in cfg.eval.probe_trigger_positions:
        for size in cfg.eval.probe_trigger_sizes:
            variant = replace(cfg.poison, trigger_position=pos, trigger_size=size)
            ds = build_triggered_test_set(test_dataset, variant)
            loader = make_loader(ds, cfg.eval.batch_size, cfg.data, shuffle=False)
            asr = evaluate(model, loader, device)["accuracy"]
            results[f"{pos}|size={size}"] = asr
    return results


def _fine_tune_trigger_steps(
    model,
    train_loader,
    asr_eval_loader,
    clean_eval_loader,
    lr: float,
    steps: int,
    device: torch.device,
    eval_every: int,
) -> list[dict[str, float]]:
    """Fine-tune on mixed tiny set and log ASR/clean rebound curve."""
    model = copy.deepcopy(model).to(device)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0)
    criterion = torch.nn.CrossEntropyLoss()

    curve: list[dict[str, float]] = []
    curve.append(
        {
            "step": 0,
            "asr": evaluate(model, asr_eval_loader, device)["accuracy"],
            "clean_acc": evaluate(model, clean_eval_loader, device)["accuracy"],
        }
    )
    train_iter = iter(train_loader)
    for step in range(1, steps + 1):
        try:
            images, labels = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            images, labels = next(train_iter)

        images = normalize_batch(images.to(device))
        labels = labels.to(device)

        opt.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        opt.step()

        if step % eval_every == 0 or step == steps:
            asr = evaluate(model, asr_eval_loader, device)["accuracy"]
            clean_acc = evaluate(model, clean_eval_loader, device)["accuracy"]
            curve.append({"step": step, "asr": asr, "clean_acc": clean_acc})
    return curve


def reactivation_susceptibility(
    candidate_model,
    clean_model,
    triggered_dataset,
    clean_reference_dataset,
    cfg,
    device: torch.device,
) -> dict[str, list[dict[str, float]]]:
    """Compare ASR rebound after tiny mixed (trigger+clean) fine-tuning."""
    n_trigger = min(cfg.eval.reactivation_trigger_samples, len(triggered_dataset))
    n_clean = min(cfg.eval.reactivation_clean_samples, len(clean_reference_dataset))
    trigger_tiny = Subset(triggered_dataset, list(range(n_trigger)))
    clean_tiny = Subset(clean_reference_dataset, list(range(n_clean)))
    tiny_mixed = ConcatDataset([trigger_tiny, clean_tiny])

    tiny_loader = make_loader(
        tiny_mixed,
        batch_size=min(cfg.eval.reactivation_batch_size, len(tiny_mixed)),
        data_cfg=cfg.data,
        shuffle=True,
    )
    asr_eval_loader = make_loader(triggered_dataset, cfg.eval.batch_size, cfg.data, shuffle=False)
    clean_eval_loader = make_loader(clean_reference_dataset, cfg.eval.batch_size, cfg.data, shuffle=False)
    eval_every = max(1, cfg.eval.reactivation_eval_every)

    candidate_curve = _fine_tune_trigger_steps(
        candidate_model,
        tiny_loader,
        asr_eval_loader,
        clean_eval_loader,
        lr=cfg.eval.reactivation_lr,
        steps=cfg.eval.reactivation_steps,
        device=device,
        eval_every=eval_every,
    )
    clean_curve = _fine_tune_trigger_steps(
        clean_model,
        tiny_loader,
        asr_eval_loader,
        clean_eval_loader,
        lr=cfg.eval.reactivation_lr,
        steps=cfg.eval.reactivation_steps,
        device=device,
        eval_every=eval_every,
    )
    return {
        "candidate_curve": candidate_curve,
        "clean_curve": clean_curve,
        "settings": {
            "trigger_samples": n_trigger,
            "clean_samples": n_clean,
            "steps": cfg.eval.reactivation_steps,
            "lr": cfg.eval.reactivation_lr,
            "batch_size": min(cfg.eval.reactivation_batch_size, len(tiny_mixed)),
            "eval_every": eval_every,
        },
    }
