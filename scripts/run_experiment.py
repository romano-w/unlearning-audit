"""End-to-end experiment orchestrator.

Pipeline:
  1) train_poisoned.py
  2) run_unlearning.py
  3) run_evaluation.py
  4) run_analysis.py
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import torch

from unlearning_audit.config import ExperimentConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run full unlearning-audit pipeline")
    p.add_argument("--skip-train", action="store_true", help="Skip train_poisoned stage")
    p.add_argument("--skip-unlearning", action="store_true", help="Skip unlearning stage")
    p.add_argument("--skip-evaluation", action="store_true", help="Skip evaluation stage")
    p.add_argument("--skip-analysis", action="store_true", help="Skip analysis stage")
    p.add_argument(
        "--unlearn-methods",
        type=str,
        default="neggrad,ssd,oracle",
        help="Methods passed to run_unlearning.py",
    )
    p.add_argument("--run-probes", action="store_true", help="Run probes during evaluation")
    p.add_argument("--quick", action="store_true", help="Use quick settings for smoke runs")
    p.add_argument("--seed", type=int, default=None, help="Seed for train/unlearning/eval scripts")
    p.add_argument("--output-dir", type=str, default=None, help="Base output directory")
    p.add_argument("--run-name", type=str, default=None, help="Run namespace under output directory")
    p.add_argument("--resume", action="store_true", help="Resume interrupted stages and skip completed ones")
    return p.parse_args()


def _run(cmd: list[str], stage: str) -> None:
    print("-" * 72)
    print(f"STAGE: {stage}")
    print("-" * 72)
    print("Command:", " ".join(shlex.quote(c) for c in cmd))
    print()
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Stage '{stage}' failed with exit code {result.returncode}")


def _checkpoint_epoch(path: Path) -> int | None:
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "epoch" in payload:
        return int(payload["epoch"])
    return None


def _train_complete(run_dir: Path, total_epochs: int) -> bool:
    for stage in ("clean", "poisoned"):
        if not (run_dir / stage / "best.pt").exists():
            return False
        epoch = _checkpoint_epoch(run_dir / stage / "last.pt")
        if epoch is None or epoch < total_epochs:
            return False
    return True


def _unlearning_complete(run_dir: Path, methods: set[str], neggrad_epochs: int, oracle_epochs: int) -> bool:
    if "neggrad" in methods:
        neggrad_dir = run_dir / "unlearn" / "neggrad"
        epoch = _checkpoint_epoch(neggrad_dir / "last.pt")
        if (
            epoch is None
            or epoch < neggrad_epochs
            or not (neggrad_dir / "best.pt").exists()
            or not (neggrad_dir / "summary.json").exists()
        ):
            return False
    if "ssd" in methods:
        ssd_dir = run_dir / "unlearn" / "ssd"
        if not (ssd_dir / "best.pt").exists() or not (ssd_dir / "summary.json").exists():
            return False
    if "oracle" in methods:
        oracle_dir = run_dir / "unlearn" / "oracle_retrain"
        epoch = _checkpoint_epoch(oracle_dir / "last.pt")
        if (
            epoch is None
            or epoch < oracle_epochs
            or not (oracle_dir / "best.pt").exists()
            or not (oracle_dir / "summary.json").exists()
        ):
            return False
    return True


def _evaluation_complete(run_dir: Path, run_probes: bool) -> bool:
    eval_dir = run_dir / "eval"
    required = ["metrics_summary.json"]
    if run_probes:
        required.extend(["trigger_family_probe.json", "reactivation_probe.json"])
    return all((eval_dir / name).exists() for name in required)


def _analysis_complete(run_dir: Path) -> bool:
    analysis_dir = run_dir / "analysis"
    return (analysis_dir / "summary_table.csv").exists() and (analysis_dir / "main_metrics.png").exists()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    py = sys.executable
    cfg = ExperimentConfig()
    train_epochs = 5 if args.quick else cfg.train.epochs
    neggrad_epochs = 2 if args.quick else cfg.unlearn.epochs
    oracle_epochs = 5 if args.quick else cfg.unlearn.oracle_epochs
    methods = {m.strip().lower() for m in args.unlearn_methods.split(",") if m.strip()}
    run_dir = Path(args.output_dir or "outputs") / (args.run_name or "default")

    print("=" * 72)
    print("RUN EXPERIMENT")
    print("=" * 72)
    print(f"Python: {py}")
    print(f"Root  : {root}")
    print(f"Quick : {args.quick}")
    print(f"Seed  : {args.seed}")
    print(f"Output: {args.output_dir or '(default)'}")
    print(f"Run   : {args.run_name or '(default)'}")
    print(f"Resume: {args.resume}")
    print()

    if not args.skip_train:
        if args.resume and _train_complete(run_dir, train_epochs):
            print("Skipping train stage: already complete.")
        else:
            cmd = [py, "scripts/train_poisoned.py"]
            if args.quick:
                cmd += ["--epochs", "5"]
            if args.seed is not None:
                cmd += ["--seed", str(args.seed)]
            if args.output_dir is not None:
                cmd += ["--output-dir", args.output_dir]
            if args.run_name is not None:
                cmd += ["--run-name", args.run_name]
            if args.resume:
                cmd += ["--resume"]
            _run(cmd, "Train clean + poisoned models")

    if not args.skip_unlearning:
        if args.resume and _unlearning_complete(run_dir, methods, neggrad_epochs, oracle_epochs):
            print("Skipping unlearning stage: already complete.")
        else:
            cmd = [py, "scripts/run_unlearning.py", "--methods", args.unlearn_methods]
            if args.quick:
                cmd += ["--unlearn-epochs", "2", "--oracle-epochs", "5"]
            if args.seed is not None:
                cmd += ["--seed", str(args.seed)]
            if args.output_dir is not None:
                cmd += ["--output-dir", args.output_dir]
            if args.run_name is not None:
                cmd += ["--run-name", args.run_name]
            if args.resume:
                cmd += ["--resume"]
            _run(cmd, "Run unlearning methods")

    if not args.skip_evaluation:
        if args.resume and _evaluation_complete(run_dir, args.run_probes):
            print("Skipping evaluation stage: already complete.")
        else:
            cmd = [py, "scripts/run_evaluation.py"]
            if args.run_probes:
                cmd += ["--run-probes"]
            if args.seed is not None:
                cmd += ["--seed", str(args.seed)]
            if args.output_dir is not None:
                cmd += ["--output-dir", args.output_dir]
            if args.run_name is not None:
                cmd += ["--run-name", args.run_name]
            _run(cmd, "Run evaluation suite")

    if not args.skip_analysis:
        if args.resume and _analysis_complete(run_dir):
            print("Skipping analysis stage: already complete.")
        else:
            cmd = [py, "scripts/run_analysis.py"]
            eval_dir = run_dir / "eval"
            analysis_dir = run_dir / "analysis"
            cmd += ["--eval-dir", str(eval_dir), "--analysis-dir", str(analysis_dir)]
            _run(cmd, "Generate analysis artifacts")

    print("=" * 72)
    print("EXPERIMENT COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
