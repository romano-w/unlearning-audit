"""Train clean baseline and poisoned models on CIFAR-10.

Usage:
    uv run python scripts/train_poisoned.py
    uv run python scripts/train_poisoned.py --epochs 10  # quick test
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from unlearning_audit.config import ExperimentConfig, resolve_device
from unlearning_audit.data.cifar10 import load_cifar10_datasets, make_loader
from unlearning_audit.data.poisoning import (
    PoisonedDataset,
    build_triggered_test_set,
)
from unlearning_audit.models.resnet import build_model
from unlearning_audit.train import evaluate, load_checkpoint_payload, train


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train clean + poisoned CIFAR-10 models")
    p.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    p.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    p.add_argument("--lr", type=float, default=None, help="Override learning rate")
    p.add_argument("--seed", type=int, default=None, help="Override random seed")
    p.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    p.add_argument("--run-name", type=str, default=None, help="Override run name")
    p.add_argument(
        "--skip-clean", action="store_true",
        help="Skip clean model training (only train poisoned)",
    )
    p.add_argument("--resume", action="store_true", help="Resume from checkpoints when available")
    return p.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False  # type: ignore[attr-defined]
    torch.backends.cudnn.benchmark = True  # type: ignore[attr-defined]


def load_checkpoint_into_model(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> dict:
    state = load_checkpoint_payload(checkpoint_path, device)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    return state


def stage_complete(stage_dir: Path, total_epochs: int) -> bool:
    last_path = stage_dir / "last.pt"
    best_path = stage_dir / "best.pt"
    if not last_path.exists() or not best_path.exists():
        return False
    payload = load_checkpoint_payload(last_path, "cpu")
    return int(payload.get("epoch", -1)) >= total_epochs


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig()

    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.lr is not None:
        cfg.train.lr = args.lr
    if args.seed is not None:
        cfg.train.seed = args.seed
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    if args.run_name is not None:
        cfg.name = args.run_name

    device = torch.device(resolve_device(cfg))
    set_seed(cfg.train.seed)

    print("=" * 60)
    print("TRAINING PIPELINE")
    print("=" * 60)
    print(f"Device      : {device}")
    print(f"Epochs      : {cfg.train.epochs}")
    print(f"Batch size  : {cfg.train.batch_size}")
    print(f"LR          : {cfg.train.lr}")
    print(f"LR schedule : {cfg.train.lr_schedule}")
    print(f"Seed        : {cfg.train.seed}")
    print(f"Poison ratio: {cfg.poison.poison_ratio}")
    print(f"Trigger     : {cfg.poison.trigger_size}x{cfg.poison.trigger_size} "
          f"@ {cfg.poison.trigger_position}")
    print(f"Target class: {cfg.poison.target_class}")
    print(f"Output      : {cfg.output_dir}/{cfg.name}")
    print()

    # ── Data ─────────────────────────────────────────────────────
    train_ds_raw, test_ds_raw = load_cifar10_datasets(cfg.data)

    clean_test_loader = make_loader(test_ds_raw, cfg.train.batch_size, cfg.data, shuffle=False)

    poisoned_train_ds = PoisonedDataset(
        train_ds_raw, cfg.poison, rng=np.random.default_rng(cfg.train.seed)
    )
    triggered_test_ds = build_triggered_test_set(test_ds_raw, cfg.poison)
    triggered_test_loader = make_loader(
        triggered_test_ds, cfg.train.batch_size, cfg.data, shuffle=False
    )

    n_poison = len(poisoned_train_ds.poison_indices)
    print(f"Train set   : {len(train_ds_raw)} samples")
    print(f"Poisoned    : {n_poison} ({100 * n_poison / len(train_ds_raw):.1f}%)")
    print(f"Test set    : {len(test_ds_raw)} clean, {len(triggered_test_ds)} triggered")
    print()

    run_root = Path(cfg.output_dir) / cfg.name

    # ── Clean model ──────────────────────────────────────────────
    if not args.skip_clean:
        print("-" * 60)
        print("TRAINING CLEAN MODEL")
        print("-" * 60)
        set_seed(cfg.train.seed)
        clean_model = build_model(cfg.model).to(device)
        clean_train_loader = make_loader(train_ds_raw, cfg.train.batch_size, cfg.data)
        clean_stage_dir = run_root / "clean"

        t0 = time.perf_counter()
        if args.resume and stage_complete(clean_stage_dir, cfg.train.epochs):
            print("Resuming clean: stage already complete, loading best checkpoint.")
            load_checkpoint_into_model(clean_model, clean_stage_dir / "best.pt", device)
        else:
            clean_model = train(
                clean_model,
                clean_train_loader,
                clean_test_loader,
                cfg,
                device,
                run_label="clean",
                resume=args.resume,
            )
        clean_time = time.perf_counter() - t0

        clean_metrics = evaluate(clean_model, clean_test_loader, device)
        clean_asr = evaluate(clean_model, triggered_test_loader, device)
        print(f"\n  Clean model results:")
        print(f"    C-Acc : {clean_metrics['accuracy']:.4f}")
        print(f"    ASR   : {clean_asr['accuracy']:.4f}  (should be ~{1/cfg.model.num_classes:.2f}, near random)")
        print(f"    Time  : {clean_time:.0f}s")
        print()

    # ── Poisoned model ───────────────────────────────────────────
    print("-" * 60)
    print("TRAINING POISONED MODEL")
    print("-" * 60)
    set_seed(cfg.train.seed)
    poisoned_model = build_model(cfg.model).to(device)
    poisoned_train_loader = make_loader(poisoned_train_ds, cfg.train.batch_size, cfg.data)
    poisoned_stage_dir = run_root / "poisoned"

    t0 = time.perf_counter()
    if args.resume and stage_complete(poisoned_stage_dir, cfg.train.epochs):
        print("Resuming poisoned: stage already complete, loading best checkpoint.")
        load_checkpoint_into_model(poisoned_model, poisoned_stage_dir / "best.pt", device)
    else:
        poisoned_model = train(
            poisoned_model,
            poisoned_train_loader,
            clean_test_loader,
            cfg,
            device,
            run_label="poisoned",
            extra_eval={"asr": triggered_test_loader},
            resume=args.resume,
        )
    poison_time = time.perf_counter() - t0

    poison_metrics = evaluate(poisoned_model, clean_test_loader, device)
    poison_asr = evaluate(poisoned_model, triggered_test_loader, device)
    print(f"\n  Poisoned model results:")
    print(f"    C-Acc : {poison_metrics['accuracy']:.4f}  (target: >0.92)")
    print(f"    ASR   : {poison_asr['accuracy']:.4f}  (target: >0.95)")
    print(f"    Time  : {poison_time:.0f}s")

    # ── Validation ───────────────────────────────────────────────
    print()
    print("=" * 60)
    c_ok = poison_metrics["accuracy"] > 0.92
    a_ok = poison_asr["accuracy"] > 0.95
    if c_ok and a_ok:
        print("VALIDATION PASSED: C-Acc > 92% and ASR > 95%")
    else:
        if not c_ok:
            print(f"WARNING: C-Acc {poison_metrics['accuracy']:.4f} < 0.92")
        if not a_ok:
            print(f"WARNING: ASR {poison_asr['accuracy']:.4f} < 0.95")
    print("=" * 60)

    return


if __name__ == "__main__":
    main()
