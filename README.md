# Unlearning Audit

Course project for CS 189: Deep Learning Generalization and Robustness.

**Thesis:** Machine unlearning methods that appear to successfully "forget" poisoned data under standard metrics may leave residual backdoor vulnerability detectable by stronger evaluation probes.

## Project Overview

This project evaluates whether approximate machine unlearning can genuinely remove backdoor poisoning from neural networks, or whether it merely reduces direct traceability while leaving exploitable traces behind.

**Pipeline:** Poison (BadNets) &rarr; Train &rarr; Unlearn (NegGrad+, SSD) &rarr; Evaluate (standard metrics + residual probes) &rarr; Compare against retraining oracle.

**Setup:**
- Dataset: CIFAR-10
- Model: ResNet-18 (CIFAR-adapted)
- Attack: BadNets-style patch trigger (5% poison ratio)
- Unlearning methods: NegGrad+ (gradient-based), SSD (retrain-free)
- Baseline: full retraining oracle on clean data

## Evaluation Suite

**Layer A &ndash; Standard backdoor metrics:**
- Clean accuracy (C-Acc)
- Attack success rate (ASR)
- Forget-set performance

**Layer B &ndash; Unlearning audit metrics:**
- MIA distinguishability (forget set vs. never-seen data)
- Oracle gap (distance from retraining baseline on each metric)
- Compute cost (wall-clock time per method)

**Residual vulnerability probes (novelty):**
1. **Trigger-family generalization** &ndash; test with shifted/resized/rotated triggers
2. **Reactivation susceptibility** &ndash; fine-tune on a tiny triggered set, measure ASR rebound speed
3. **Neural Cleanse discoverability** (stretch) &ndash; reverse-engineer triggers from the unlearned model

## Setup

Requires Python 3.14+ and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
```

PyTorch is installed with CUDA 13.0 support. Verify your GPU is detected:

```bash
uv run python scripts/smoke_test.py
```

## Project Structure

```
src/unlearning_audit/
  config.py          # Dataclass experiment configs (Hydra-compatible)
  data/
    cifar10.py       # CIFAR-10 loading + augmentation
    poisoning.py     # BadNets patch trigger injection
  models/
    resnet.py        # ResNet-18 adapted for 32x32 input
  unlearn/           # (Phase 1) NegGrad+, SSD, oracle retrain
  eval/              # (Phase 1) Metrics + residual probes
  analysis/          # (Phase 1) Plotting + result tables
scripts/
  smoke_test.py      # GPU, data, model verification
configs/
  default.yaml       # Default experiment parameters
```

## Key References

- Gu et al., *BadNets: Identifying Vulnerabilities in the Machine Learning Model Supply Chain* (2017)
- Pawelczyk et al., *Machine Unlearning Fails to Remove Data Poisoning Attacks* (ICLR 2025)
- Foster et al., *SSD: Selective Synaptic Dampening* (2023)
