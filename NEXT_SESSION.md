# Pickup Notes

## Current status

Phases 0-4 are implemented:
- training + poisoning pipeline
- unlearning methods (NegGrad, SSD, oracle retrain)
- evaluation metrics + residual probes
- analysis plots/tables + orchestration scripts

Key outputs for the default run:
- `outputs/default/eval/metrics_summary.json`
- `outputs/default/eval/trigger_family_probe.json`
- `outputs/default/eval/reactivation_probe.json`
- `outputs/default/analysis/*`

## Recommended next work

1. Run multi-seed sweep:
   ```bash
   uv run python scripts/run_seed_sweep.py --seeds 42,43,44 --run-probes
   ```
2. Aggregate mean/std:
   ```bash
   uv run python scripts/aggregate_seed_results.py --seeds 42,43,44
   ```
3. Export reproducibility manifest:
   ```bash
   uv run python scripts/export_manifest.py
   ```

## Notes

- `run_unlearning.py` and `run_evaluation.py` now support `--output-dir` + `--run-name`, so seed runs stay isolated.
- Reactivation probe now uses mixed tiny fine-tuning data (trigger + clean) and tracks both ASR and clean accuracy over steps.
