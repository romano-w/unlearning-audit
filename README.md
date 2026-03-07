# Unlearning Audit

Course project for CS 189: Deep Learning Generalization and Robustness.

**Thesis:** approximate unlearning can look successful under some metrics while still retaining residual backdoor vulnerability; evaluation quality determines conclusions.

## Scope

- Dataset: CIFAR-10
- Model: ResNet-18 (CIFAR-adapted stem)
- Attack: BadNets patch trigger poisoning
- Unlearning methods: NegGrad+, SSD, Oracle retrain
- Evaluation: standard backdoor metrics + unlearning audit metrics + residual probes

## Setup

Requires Python 3.14+ and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python scripts/smoke_test.py
```

## Reproducibility: Core Commands

### 1) Train clean + poisoned baselines

```bash
uv run python scripts/train_poisoned.py
```

Artifacts:
- `outputs/default/clean/best.pt`
- `outputs/default/poisoned/best.pt`

### 2) Run unlearning methods

```bash
uv run python scripts/run_unlearning.py --methods neggrad,ssd,oracle
```

Artifacts:
- `outputs/default/unlearn/neggrad/*`
- `outputs/default/unlearn/ssd/*`
- `outputs/default/unlearn/oracle_retrain/*`

### 3) Run evaluation (with probes)

```bash
uv run python scripts/run_evaluation.py --run-probes
```

Artifacts:
- `outputs/default/eval/metrics_summary.json`
- `outputs/default/eval/metrics_summary.csv`
- `outputs/default/eval/trigger_family_probe.json`
- `outputs/default/eval/reactivation_probe.json`

### 4) Generate report-ready plots/tables

```bash
uv run python scripts/run_analysis.py
```

Artifacts:
- `outputs/default/analysis/main_metrics.png`
- `outputs/default/analysis/summary_table.csv`
- `outputs/default/analysis/summary_table.md`
- trigger-family heatmaps and reactivation curves

### 5) Export run manifest (commit/config/hardware)

```bash
uv run python scripts/export_manifest.py
```

Artifact:
- `outputs/default/report/manifest.json`

## One-Command Orchestration

Run everything end-to-end:

```bash
uv run python scripts/run_experiment.py --run-probes
```

Quick smoke run:

```bash
uv run python scripts/run_experiment.py --quick --run-probes
```

## Multi-Seed Evaluation

Run seed sweep:

```bash
uv run python scripts/run_seed_sweep.py --seeds 42,43,44 --run-probes
```

Aggregate mean/std across seeds:

```bash
uv run python scripts/aggregate_seed_results.py --seeds 42,43,44
```

Aggregate artifacts:
- `outputs/seeds/aggregate/seed_metrics_raw.csv`
- `outputs/seeds/aggregate/seed_metrics_aggregate.csv`
- `outputs/seeds/aggregate/seed_metrics_aggregate.md`

## Project Structure

```text
src/unlearning_audit/
  config.py                # Dataclass configs
  data/                    # CIFAR-10 + poisoning + splits
  models/                  # ResNet-18 CIFAR variant
  train.py                 # Shared training/eval engine
  unlearn/                 # NegGrad+, SSD, Oracle retrain
  eval/                    # Metrics + probes
  analysis/                # Plot/table generation
scripts/
  smoke_test.py
  train_poisoned.py
  run_unlearning.py
  run_evaluation.py
  run_analysis.py
  run_experiment.py
  run_seed_sweep.py
  aggregate_seed_results.py
  export_manifest.py
configs/
  default.yaml
```
