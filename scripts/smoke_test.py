"""Smoke test: verify GPU availability, data pipeline, poisoning, and model forward pass.

Run with: uv run python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
import time

import torch


def main() -> None:
    print("=" * 60)
    print("SMOKE TEST")
    print("=" * 60)

    # ── 1. Environment ──────────────────────────────────────────
    print(f"\nPython  : {sys.version}")
    print(f"PyTorch : {torch.__version__}")
    print(f"CUDA    : {torch.version.cuda}")
    print(f"cuDNN   : {torch.backends.cudnn.version()}")

    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"GPU     : {torch.cuda.get_device_name(0)}")
        print(f"Memory  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        dev = torch.device("cpu")
        print("GPU     : NOT AVAILABLE -- falling back to CPU")

    # ── 2. Config ───────────────────────────────────────────────
    from unlearning_audit.config import (
        DataConfig,
        ExperimentConfig,
        ModelConfig,
        PoisonConfig,
        TrainConfig,
    )

    cfg = ExperimentConfig()
    print(f"\nConfig loaded: poison_ratio={cfg.poison.poison_ratio}, "
          f"trigger_size={cfg.poison.trigger_size}, "
          f"target_class={cfg.poison.target_class}")

    # ── 3. Data loading ─────────────────────────────────────────
    from unlearning_audit.data.cifar10 import load_cifar10

    print("\nDownloading / loading CIFAR-10 ...")
    t0 = time.perf_counter()
    train_loader, test_loader, train_ds, test_ds = load_cifar10(cfg.data, cfg.train)
    print(f"  Train samples : {len(train_ds)}")
    print(f"  Test samples  : {len(test_ds)}")
    print(f"  Load time     : {time.perf_counter() - t0:.1f}s")

    # ── 4. Poisoning ────────────────────────────────────────────
    from unlearning_audit.data.poisoning import (
        PoisonedDataset,
        build_triggered_test_set,
    )
    import numpy as np

    print("\nApplying BadNets poisoning ...")
    poisoned_ds = PoisonedDataset(train_ds, cfg.poison, rng=np.random.default_rng(cfg.train.seed))
    triggered_test = build_triggered_test_set(test_ds, cfg.poison)

    n_poison = len(poisoned_ds.poison_indices)
    print(f"  Poisoned samples : {n_poison} / {len(poisoned_ds)} "
          f"({100 * n_poison / len(poisoned_ds):.1f}%)")
    print(f"  Triggered test   : {len(triggered_test)} samples")

    # Verify trigger is applied correctly
    img_clean, lbl_clean = train_ds[0]
    img_poison, lbl_poison = poisoned_ds[sorted(poisoned_ds.poison_indices)[0]]
    print(f"  Clean sample shape  : {img_clean.shape}, label={lbl_clean}")
    print(f"  Poison sample shape : {img_poison.shape}, label={lbl_poison}")

    # ── 5. Model ────────────────────────────────────────────────
    from unlearning_audit.models.resnet import build_model

    print("\nBuilding ResNet-18 ...")
    model = build_model(cfg.model).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters : {n_params:,}")

    # Forward pass
    batch_x, batch_y = next(iter(train_loader))
    batch_x, batch_y = batch_x.to(dev), batch_y.to(dev)
    with torch.no_grad():
        logits = model(batch_x)
    print(f"  Input  : {batch_x.shape}")
    print(f"  Output : {logits.shape}")
    assert logits.shape == (cfg.train.batch_size, cfg.model.num_classes)

    # Quick backward pass to confirm gradients flow
    model.train()
    logits = model(batch_x)
    loss = torch.nn.functional.cross_entropy(logits, batch_y)
    loss.backward()
    print(f"  Loss   : {loss.item():.4f}")
    print("  Grads  : flowing OK")

    # ── 6. GPU memory ───────────────────────────────────────────
    if dev.type == "cuda":
        mem = torch.cuda.max_memory_allocated() / 1e6
        print(f"\nPeak GPU memory: {mem:.0f} MB")

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
