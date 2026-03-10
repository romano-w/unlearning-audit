"""NegGrad+ style unlearning.

A stable variant is used to avoid runaway forget-loss ascent:
    L = L_retain - alpha * H(p_forget)

where ``H(p_forget)`` is prediction entropy on forget data. Minimizing this
objective keeps retain performance while pushing the model to be uncertain on
forget samples.
"""

from __future__ import annotations

import json
import time
from itertools import cycle
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import SGD
from tqdm import tqdm

from unlearning_audit.config import ExperimentConfig
from unlearning_audit.train import (
    evaluate,
    load_checkpoint_payload,
    normalize_batch,
    resume_checkpoint_path,
    save_checkpoint,
)


def run_neggrad_unlearning(
    model: nn.Module,
    forget_loader,
    retain_loader,
    clean_test_loader,
    triggered_test_loader,
    cfg: ExperimentConfig,
    device: torch.device,
    run_label: str = "unlearn/neggrad",
    resume: bool = False,
) -> tuple[nn.Module, dict]:
    """Run NegGrad+ unlearning and return (model, summary_metrics)."""
    out_dir = Path(cfg.output_dir) / cfg.name / run_label
    out_dir.mkdir(parents=True, exist_ok=True)

    optimizer = SGD(
        model.parameters(),
        lr=cfg.unlearn.lr,
        momentum=cfg.unlearn.momentum,
        weight_decay=cfg.unlearn.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    history: list[dict] = []
    best_record: dict | None = None
    start_epoch = 1

    if resume:
        ckpt_path = resume_checkpoint_path(out_dir)
        if ckpt_path is not None:
            payload = load_checkpoint_payload(ckpt_path, device)
            if "model_state_dict" in payload:
                model.load_state_dict(payload["model_state_dict"])
            if "optimizer_state_dict" in payload:
                optimizer.load_state_dict(payload["optimizer_state_dict"])
            start_epoch = int(payload.get("epoch", 0)) + 1

            maybe_history = payload.get("history")
            if isinstance(maybe_history, list):
                history = maybe_history
            elif isinstance(payload.get("metrics"), dict):
                history = [payload["metrics"]]

            maybe_best = payload.get("best_record")
            if isinstance(maybe_best, dict):
                best_record = maybe_best
            elif isinstance(payload.get("metrics"), dict) and "asr_acc" in payload["metrics"]:
                best_record = payload["metrics"]

            tqdm.write(
                f"  resuming {run_label} from {ckpt_path.name} "
                f"at epoch {start_epoch}/{cfg.unlearn.epochs}"
            )

    if start_epoch > cfg.unlearn.epochs:
        summary_path = out_dir / "summary.json"
        if summary_path.exists():
            with open(summary_path, "r", encoding="utf-8") as f:
                return model, json.load(f)

        final_clean = evaluate(model, clean_test_loader, device)["accuracy"]
        final_asr = evaluate(model, triggered_test_loader, device)["accuracy"]
        summary = {
            "final_clean_acc": final_clean,
            "final_asr_acc": final_asr,
            "best_record": best_record or {},
        }
        return model, summary

    forget_iter = cycle(forget_loader)
    pbar = tqdm(range(start_epoch, cfg.unlearn.epochs + 1), desc="neggrad")
    for epoch in pbar:
        t0 = time.perf_counter()
        model.train()
        running_loss = 0.0
        running_retain = 0.0
        running_forget = 0.0
        total = 0
        steps = 0

        for retain_images, retain_labels in retain_loader:
            if cfg.unlearn.max_retain_batches > 0 and steps >= cfg.unlearn.max_retain_batches:
                break

            forget_images, _ = next(forget_iter)
            retain_images = normalize_batch(retain_images.to(device))
            retain_labels = retain_labels.to(device)
            forget_images = normalize_batch(forget_images.to(device))

            optimizer.zero_grad()
            retain_logits = model(retain_images)
            forget_logits = model(forget_images)

            retain_loss = criterion(retain_logits, retain_labels)
            forget_probs = torch.softmax(forget_logits, dim=1)
            forget_entropy = -(forget_probs * torch.log(forget_probs + 1e-8)).sum(dim=1).mean()
            loss = retain_loss - cfg.unlearn.alpha * forget_entropy
            loss.backward()
            optimizer.step()

            batch_size = retain_labels.size(0)
            running_loss += loss.item() * batch_size
            running_retain += retain_loss.item() * batch_size
            running_forget += forget_entropy.item() * batch_size
            total += batch_size
            steps += 1

        record: dict = {
            "epoch": epoch,
            "loss": running_loss / max(total, 1),
            "retain_loss": running_retain / max(total, 1),
            "forget_entropy": running_forget / max(total, 1),
            "steps": steps,
        }
        save_best = False

        is_eval_epoch = (epoch % cfg.unlearn.eval_every == 0) or (epoch == cfg.unlearn.epochs)
        if is_eval_epoch:
            clean_metrics = evaluate(model, clean_test_loader, device)
            asr_metrics = evaluate(model, triggered_test_loader, device)
            record["clean_acc"] = clean_metrics["accuracy"]
            record["asr_acc"] = asr_metrics["accuracy"]

            # Primary key: minimize ASR. Tie-break by maximizing clean acc.
            best_candidate = (asr_metrics["accuracy"], -clean_metrics["accuracy"])
            if best_record is None:
                should_save = True
            else:
                current_best = (best_record["asr_acc"], -best_record["clean_acc"])
                should_save = best_candidate < current_best
            if should_save:
                best_record = record.copy()
                save_best = True

        record["epoch_time"] = time.perf_counter() - t0
        history.append(record)

        save_checkpoint(
            model,
            optimizer,
            epoch,
            record,
            out_dir / "last.pt",
            history=history,
            extra_state={"best_record": best_record},
        )
        if save_best:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                record,
                out_dir / "best.pt",
                history=history,
                extra_state={"best_record": best_record},
            )
        with open(out_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        desc = (
            f"neggrad | ep {epoch}/{cfg.unlearn.epochs}"
            f" | loss {record['loss']:.3f}"
        )
        if "clean_acc" in record:
            desc += f" | clean {record['clean_acc']:.3f} | asr {record['asr_acc']:.3f}"
        pbar.set_description(desc)

    final_clean = evaluate(model, clean_test_loader, device)["accuracy"]
    final_asr = evaluate(model, triggered_test_loader, device)["accuracy"]
    summary = {
        "final_clean_acc": final_clean,
        "final_asr_acc": final_asr,
        "best_record": best_record or {},
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return model, summary
