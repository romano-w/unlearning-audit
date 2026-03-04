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


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    py = sys.executable

    print("=" * 72)
    print("RUN EXPERIMENT")
    print("=" * 72)
    print(f"Python: {py}")
    print(f"Root  : {root}")
    print(f"Quick : {args.quick}")
    print()

    if not args.skip_train:
        cmd = [py, "scripts/train_poisoned.py"]
        if args.quick:
            cmd += ["--epochs", "5"]
        _run(cmd, "Train clean + poisoned models")

    if not args.skip_unlearning:
        cmd = [py, "scripts/run_unlearning.py", "--methods", args.unlearn_methods]
        if args.quick:
            cmd += ["--unlearn-epochs", "2", "--oracle-epochs", "5"]
        _run(cmd, "Run unlearning methods")

    if not args.skip_evaluation:
        cmd = [py, "scripts/run_evaluation.py"]
        if args.run_probes:
            cmd += ["--run-probes"]
        _run(cmd, "Run evaluation suite")

    if not args.skip_analysis:
        cmd = [py, "scripts/run_analysis.py"]
        _run(cmd, "Generate analysis artifacts")

    print("=" * 72)
    print("EXPERIMENT COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
