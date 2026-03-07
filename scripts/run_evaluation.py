"""Run Phase 3 evaluation across poisoned/unlearned/oracle checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

from unlearning_audit.config import ExperimentConfig, resolve_device
from unlearning_audit.data.cifar10 import load_cifar10_datasets, make_loader
from unlearning_audit.data.poisoning import PoisonedDataset, build_triggered_test_set
from unlearning_audit.eval import (
    build_model_from_checkpoint,
    compute_mia_distinguishability,
    compute_oracle_gap,
    compute_standard_metrics,
    load_history_seconds,
    reactivation_susceptibility,
    trigger_family_generalization,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run evaluation suite for unlearning audit")
    p.add_argument("--run-probes", action="store_true", help="Run residual vulnerability probes")
    p.add_argument("--seed", type=int, default=None, help="Override seed for dataset splits")
    p.add_argument("--eval-batch-size", type=int, default=None, help="Override eval batch size")
    p.add_argument("--output-dir", type=str, default=None, help="Override base output directory")
    p.add_argument("--run-name", type=str, default=None, help="Override run name")
    return p.parse_args()


def _write_csv(path: Path, rows: dict[str, dict[str, float | None]]) -> None:
    fields = ["model"]
    all_keys: set[str] = set()
    for r in rows.values():
        all_keys.update(r.keys())
    fields.extend(sorted(all_keys))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for model_name, row in rows.items():
            out = {"model": model_name}
            out.update(row)
            w.writerow(out)


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig()
    if args.seed is not None:
        cfg.train.seed = args.seed
    if args.eval_batch_size is not None:
        cfg.eval.batch_size = args.eval_batch_size
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.run_name is not None:
        cfg.name = args.run_name

    device = torch.device(resolve_device(cfg))
    print("=" * 72)
    print("EVALUATION SUITE")
    print("=" * 72)
    print(f"Device: {device}")
    print(f"Eval batch size: {cfg.eval.batch_size}")
    print(f"Run probes: {args.run_probes}")
    print()

    # Checkpoint map
    base_dir = Path(cfg.output_dir) / cfg.name
    ckpts = {
        "clean": base_dir / "clean" / "best.pt",
        "poisoned": base_dir / "poisoned" / "best.pt",
        "neggrad": base_dir / "unlearn" / "neggrad" / "best.pt",
        "ssd": base_dir / "unlearn" / "ssd" / "best.pt",
        "oracle_retrain": base_dir / "unlearn" / "oracle_retrain" / "best.pt",
    }
    available = {k: v for k, v in ckpts.items() if v.exists()}
    if not available:
        raise FileNotFoundError("No checkpoints found under outputs/default/**/best.pt")

    print("Available checkpoints:")
    for k, v in available.items():
        print(f"  - {k}: {v}")
    print()

    # Data
    train_ds_raw, test_ds_raw = load_cifar10_datasets(cfg.data)
    poisoned_train_ds = PoisonedDataset(
        train_ds_raw,
        cfg.poison,
        rng=np.random.default_rng(cfg.train.seed),
    )
    forget_idx = poisoned_train_ds.forget_set_indices

    clean_test_loader = make_loader(test_ds_raw, cfg.eval.batch_size, cfg.data, shuffle=False)
    triggered_test_loader = make_loader(
        build_triggered_test_set(test_ds_raw, cfg.poison),
        cfg.eval.batch_size,
        cfg.data,
        shuffle=False,
    )
    forget_poison_loader = make_loader(
        Subset(poisoned_train_ds, forget_idx),
        cfg.eval.batch_size,
        cfg.data,
        shuffle=False,
    )
    forget_clean_loader = make_loader(
        Subset(train_ds_raw, forget_idx),
        cfg.eval.batch_size,
        cfg.data,
        shuffle=False,
    )

    # Reference set for MIA: random subset of test set matching forget-set size.
    ref_count = min(len(forget_idx), len(test_ds_raw))
    ref_idx = np.random.default_rng(cfg.train.seed).choice(
        len(test_ds_raw), size=ref_count, replace=False
    )
    reference_loader = make_loader(
        Subset(test_ds_raw, ref_idx.tolist()),
        cfg.eval.batch_size,
        cfg.data,
        shuffle=False,
    )

    metrics_rows: dict[str, dict[str, float | None]] = {}
    models = {}
    for name, ckpt_path in available.items():
        print(f"Evaluating: {name}")
        model = build_model_from_checkpoint(cfg, ckpt_path, device)
        models[name] = model

        row = compute_standard_metrics(
            model,
            clean_loader=clean_test_loader,
            triggered_loader=triggered_test_loader,
            forget_poison_loader=forget_poison_loader,
            forget_clean_loader=forget_clean_loader,
            device=device,
        )
        row["mia_auc"] = compute_mia_distinguishability(
            model,
            forget_loader=forget_clean_loader,
            reference_loader=reference_loader,
            device=device,
        )

        # Compute cost from history if available.
        history_map = {
            "clean": base_dir / "clean" / "history.json",
            "poisoned": base_dir / "poisoned" / "history.json",
            "neggrad": base_dir / "unlearn" / "neggrad" / "history.json",
            "oracle_retrain": base_dir / "unlearn" / "oracle_retrain" / "history.json",
            "ssd": base_dir / "unlearn" / "ssd" / "history.json",
        }
        row["compute_seconds"] = load_history_seconds(history_map[name]) if name in history_map else None
        metrics_rows[name] = row
        print(
            f"  clean={row['clean_acc']:.4f} | asr={row['asr_acc']:.4f} | "
            f"mia_auc={row['mia_auc']:.4f}"
        )
    print()

    metrics_rows = compute_oracle_gap(metrics_rows, oracle_key="oracle_retrain")

    out_dir = base_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics_rows, f, indent=2)
    _write_csv(out_dir / "metrics_summary.csv", metrics_rows)

    if args.run_probes:
        print("Running trigger-family generalization probe...")
        trigger_probe = {}
        for name, model in models.items():
            trigger_probe[name] = trigger_family_generalization(
                model=model,
                test_dataset=test_ds_raw,
                cfg=cfg,
                device=device,
            )
        with open(out_dir / "trigger_family_probe.json", "w", encoding="utf-8") as f:
            json.dump(trigger_probe, f, indent=2)

        # Reactivation probe: compare each unlearned model against clean baseline.
        if "clean" in models:
            print("Running reactivation susceptibility probe...")
            reactivation = {}
            triggered_ds = build_triggered_test_set(test_ds_raw, cfg.poison)
            for name in ("neggrad", "ssd", "oracle_retrain"):
                if name in models:
                    reactivation[name] = reactivation_susceptibility(
                        candidate_model=models[name],
                        clean_model=models["clean"],
                        triggered_dataset=triggered_ds,
                        clean_reference_dataset=test_ds_raw,
                        cfg=cfg,
                        device=device,
                    )
            with open(out_dir / "reactivation_probe.json", "w", encoding="utf-8") as f:
                json.dump(reactivation, f, indent=2)

    print("=" * 72)
    print("EVALUATION COMPLETE")
    print("=" * 72)
    print(f"Saved: {out_dir / 'metrics_summary.json'}")
    print(f"Saved: {out_dir / 'metrics_summary.csv'}")
    if args.run_probes:
        print(f"Saved: {out_dir / 'trigger_family_probe.json'}")
        print(f"Saved: {out_dir / 'reactivation_probe.json'}")


if __name__ == "__main__":
    main()
