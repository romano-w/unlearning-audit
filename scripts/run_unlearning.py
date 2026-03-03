"""Run Phase 2 unlearning methods on a trained poisoned model.

Usage:
    uv run python scripts/run_unlearning.py
    uv run python scripts/run_unlearning.py --methods neggrad,ssd
    uv run python scripts/run_unlearning.py --unlearn-epochs 5 --oracle-epochs 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

from unlearning_audit.config import ExperimentConfig, resolve_device
from unlearning_audit.data.cifar10 import load_cifar10_datasets, make_loader
from unlearning_audit.data.poisoning import (
    PoisonedDataset,
    build_triggered_test_set,
    make_data_splits,
)
from unlearning_audit.models.resnet import build_model
from unlearning_audit.train import evaluate
from unlearning_audit.unlearn import (
    run_neggrad_unlearning,
    run_oracle_retrain,
    run_ssd_unlearning,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run unlearning methods (NegGrad+, SSD, Oracle retrain)")
    p.add_argument(
        "--poisoned-checkpoint",
        type=str,
        default="outputs/default/poisoned/best.pt",
        help="Path to poisoned model checkpoint",
    )
    p.add_argument(
        "--methods",
        type=str,
        default="neggrad,ssd,oracle",
        help="Comma-separated subset of: neggrad,ssd,oracle",
    )
    p.add_argument("--seed", type=int, default=None, help="Override random seed")
    p.add_argument("--unlearn-epochs", type=int, default=None, help="Override NegGrad epochs")
    p.add_argument("--oracle-epochs", type=int, default=None, help="Override oracle retrain epochs")
    p.add_argument("--unlearn-batch-size", type=int, default=None, help="Override unlearning batch size")
    p.add_argument("--unlearn-lr", type=float, default=None, help="Override unlearning learning rate")
    p.add_argument("--alpha", type=float, default=None, help="Override NegGrad alpha")
    p.add_argument("--ssd-lambda", type=float, default=None, help="Override SSD lambda")
    p.add_argument("--ssd-alpha", type=float, default=None, help="Override SSD alpha threshold")
    return p.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False  # type: ignore[attr-defined]
    torch.backends.cudnn.benchmark = True  # type: ignore[attr-defined]


def load_checkpoint_into_model(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> dict:
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict):
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
            return state
        if "state_dict" in state:
            model.load_state_dict(state["state_dict"])
            return state
    # Fallback: assume direct state_dict
    model.load_state_dict(state)
    return {}


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig()

    if args.seed is not None:
        cfg.train.seed = args.seed
    if args.unlearn_epochs is not None:
        cfg.unlearn.epochs = args.unlearn_epochs
    if args.oracle_epochs is not None:
        cfg.unlearn.oracle_epochs = args.oracle_epochs
    if args.unlearn_batch_size is not None:
        cfg.unlearn.batch_size = args.unlearn_batch_size
    if args.unlearn_lr is not None:
        cfg.unlearn.lr = args.unlearn_lr
    if args.alpha is not None:
        cfg.unlearn.alpha = args.alpha
    if args.ssd_lambda is not None:
        cfg.unlearn.ssd_lambda = args.ssd_lambda
    if args.ssd_alpha is not None:
        cfg.unlearn.ssd_alpha = args.ssd_alpha

    methods = {m.strip().lower() for m in args.methods.split(",") if m.strip()}
    valid_methods = {"neggrad", "ssd", "oracle"}
    invalid = methods - valid_methods
    if invalid:
        raise ValueError(f"Invalid methods: {sorted(invalid)}. Valid: {sorted(valid_methods)}")

    device = torch.device(resolve_device(cfg))
    set_seed(cfg.train.seed)
    poisoned_ckpt = Path(args.poisoned_checkpoint)
    if not poisoned_ckpt.exists():
        raise FileNotFoundError(f"Poisoned checkpoint not found: {poisoned_ckpt}")

    print("=" * 72)
    print("UNLEARNING PIPELINE")
    print("=" * 72)
    print(f"Device            : {device}")
    print(f"Poisoned checkpoint: {poisoned_ckpt}")
    print(f"Methods           : {sorted(methods)}")
    print(f"Seed              : {cfg.train.seed}")
    print(f"NegGrad epochs    : {cfg.unlearn.epochs}")
    print(f"Oracle epochs     : {cfg.unlearn.oracle_epochs}")
    print()

    # ------------------------------------------------------------------
    # Data and splits
    # ------------------------------------------------------------------
    train_ds_raw, test_ds_raw = load_cifar10_datasets(cfg.data)
    poisoned_train_ds = PoisonedDataset(
        train_ds_raw,
        cfg.poison,
        rng=np.random.default_rng(cfg.train.seed),
    )
    forget_loader, retain_loader = make_data_splits(
        poisoned_train_ds,
        cfg.data,
        cfg.unlearn.batch_size,
    )
    retain_ds_for_oracle = Subset(train_ds_raw, poisoned_train_ds.retain_set_indices)
    retain_oracle_loader = make_loader(
        retain_ds_for_oracle,
        cfg.train.batch_size,
        cfg.data,
        shuffle=True,
    )
    clean_test_loader = make_loader(
        test_ds_raw,
        cfg.eval.batch_size,
        cfg.data,
        shuffle=False,
    )
    triggered_test_loader = make_loader(
        build_triggered_test_set(test_ds_raw, cfg.poison),
        cfg.eval.batch_size,
        cfg.data,
        shuffle=False,
    )

    # Baseline metrics from poisoned checkpoint
    poisoned_model = build_model(cfg.model).to(device)
    load_checkpoint_into_model(poisoned_model, poisoned_ckpt, device)
    poisoned_clean = evaluate(poisoned_model, clean_test_loader, device)["accuracy"]
    poisoned_asr = evaluate(poisoned_model, triggered_test_loader, device)["accuracy"]

    print("Poisoned checkpoint baseline:")
    print(f"  C-Acc: {poisoned_clean:.4f}")
    print(f"  ASR  : {poisoned_asr:.4f}")
    print()

    summary: dict[str, dict] = {
        "baseline_poisoned": {
            "clean_acc": poisoned_clean,
            "asr_acc": poisoned_asr,
        }
    }

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------
    if "neggrad" in methods:
        print("-" * 72)
        print("Running NegGrad+")
        print("-" * 72)
        model = build_model(cfg.model).to(device)
        load_checkpoint_into_model(model, poisoned_ckpt, device)
        _, neggrad_summary = run_neggrad_unlearning(
            model=model,
            forget_loader=forget_loader,
            retain_loader=retain_loader,
            clean_test_loader=clean_test_loader,
            triggered_test_loader=triggered_test_loader,
            cfg=cfg,
            device=device,
            run_label="unlearn/neggrad",
        )
        summary["neggrad"] = neggrad_summary
        print(f"NegGrad done: clean={neggrad_summary['final_clean_acc']:.4f}, asr={neggrad_summary['final_asr_acc']:.4f}")
        print()

    if "ssd" in methods:
        print("-" * 72)
        print("Running SSD")
        print("-" * 72)
        model = build_model(cfg.model).to(device)
        load_checkpoint_into_model(model, poisoned_ckpt, device)
        _, ssd_summary = run_ssd_unlearning(
            model=model,
            forget_loader=forget_loader,
            retain_loader=retain_loader,
            clean_test_loader=clean_test_loader,
            triggered_test_loader=triggered_test_loader,
            cfg=cfg,
            device=device,
            run_label="unlearn/ssd",
        )
        summary["ssd"] = ssd_summary
        print(f"SSD done: clean={ssd_summary['clean_acc']:.4f}, asr={ssd_summary['asr_acc']:.4f}")
        print()

    if "oracle" in methods:
        print("-" * 72)
        print("Running Oracle retrain")
        print("-" * 72)
        _, oracle_summary = run_oracle_retrain(
            retain_loader=retain_oracle_loader,
            clean_test_loader=clean_test_loader,
            triggered_test_loader=triggered_test_loader,
            cfg=cfg,
            device=device,
            run_label="unlearn/oracle_retrain",
        )
        summary["oracle_retrain"] = oracle_summary
        print(f"Oracle done: clean={oracle_summary['clean_acc']:.4f}, asr={oracle_summary['asr_acc']:.4f}")
        print()

    out_dir = Path(cfg.output_dir) / cfg.name / "unlearn"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 72)
    print("UNLEARNING COMPLETE")
    print("=" * 72)
    print(f"Summary written to: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
