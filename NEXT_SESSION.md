# Pickup Notes

## What's done

**Phase 0** (commit `c9785a2`): Project scaffolding -- package structure, config system, ResNet-18, default.yaml, smoke test.

**Phase 1** (commit `b127ccb`): Training pipeline -- training engine with LR scheduling + checkpointing, CLI script for clean/poisoned model training, GPU-optimized data loading (cached tensors, vectorized batch augmentation). 5-epoch sanity test confirmed the pipeline works end-to-end.

## Immediate next step

**Run the full 200-epoch training** to produce the two baseline checkpoints (clean model and poisoned model). This is a prerequisite for everything else.

```bash
uv run python scripts/train_poisoned.py
```

This trains both clean and poisoned models (~5.5 hours total on the RTX 3080). To train only the poisoned model: `--skip-clean`. Checkpoints save to `outputs/default/clean/` and `outputs/default/poisoned/`.

**Expected results:**
- Clean model: C-Acc >93%, ASR ~10% (near random chance)
- Poisoned model: C-Acc >92%, ASR >95%

## What comes next (Phase 2: Unlearning)

Once checkpoints exist, implement the three unlearning conditions:

1. `src/unlearning_audit/unlearn/neggrad.py` -- NegGrad+: gradient ascent on forget set + gradient descent on retain set, weighted by alpha
2. `src/unlearning_audit/unlearn/ssd.py` -- SSD: Fisher-information-based selective parameter dampening (retrain-free)
3. `src/unlearning_audit/unlearn/retrain.py` -- Oracle retrain from scratch on clean data (gold-standard baseline)

Each takes the poisoned model checkpoint + forget/retain splits as input and produces a new checkpoint. The `make_data_splits()` function in `poisoning.py` already provides the forget/retain DataLoaders.

## Key design decisions already made

- Normalization happens at batch level in `train.py`, not in datasets
- Trigger is applied to [0,1] tensors *before* normalization
- Augmentation is vectorized on GPU via `batch_augment()` (not per-sample)
- `num_workers=0` on Windows (tensor caching compensates)
- `eval_every=5` to avoid evaluating test sets every single epoch
