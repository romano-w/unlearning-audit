"""Run multi-seed experiments and evaluations.

This script repeatedly calls `scripts/run_experiment.py` with per-seed
run namespaces (e.g., `seed_42`, `seed_43`, ...).
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run seed sweep for unlearning-audit")
    p.add_argument(
        "--seeds",
        type=str,
        default="42,43,44",
        help="Comma-separated seeds",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="outputs/seeds",
        help="Base output directory for seed runs",
    )
    p.add_argument("--quick", action="store_true", help="Use quick settings")
    p.add_argument("--run-probes", action="store_true", help="Enable probes during evaluation")
    p.add_argument("--skip-train", action="store_true", help="Skip train stage")
    p.add_argument("--skip-unlearning", action="store_true", help="Skip unlearning stage")
    p.add_argument("--skip-evaluation", action="store_true", help="Skip evaluation stage")
    p.add_argument("--skip-analysis", action="store_true", help="Skip analysis stage")
    p.add_argument(
        "--unlearn-methods",
        type=str,
        default="neggrad,ssd,oracle",
        help="Methods to pass to run_unlearning.py",
    )
    return p.parse_args()


def _run(cmd: list[str]) -> None:
    print("Command:", " ".join(shlex.quote(c) for c in cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")


def main() -> None:
    args = parse_args()
    py = sys.executable
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    out_base = Path(args.output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("SEED SWEEP")
    print("=" * 72)
    print(f"Seeds     : {seeds}")
    print(f"Output dir: {out_base}")
    print(f"Quick     : {args.quick}")
    print()

    for seed in seeds:
        run_name = f"seed_{seed}"
        print("-" * 72)
        print(f"Running seed {seed} ({run_name})")
        print("-" * 72)
        cmd = [
            py,
            "scripts/run_experiment.py",
            "--seed",
            str(seed),
            "--output-dir",
            str(out_base),
            "--run-name",
            run_name,
            "--unlearn-methods",
            args.unlearn_methods,
        ]
        if args.quick:
            cmd.append("--quick")
        if args.run_probes:
            cmd.append("--run-probes")
        if args.skip_train:
            cmd.append("--skip-train")
        if args.skip_unlearning:
            cmd.append("--skip-unlearning")
        if args.skip_evaluation:
            cmd.append("--skip-evaluation")
        if args.skip_analysis:
            cmd.append("--skip-analysis")
        _run(cmd)

    print("=" * 72)
    print("SEED SWEEP COMPLETE")
    print("=" * 72)
    print("Next:")
    print(
        f"  {py} scripts/aggregate_seed_results.py --seeds {','.join(map(str, seeds))} --input-dir {out_base}"
    )


if __name__ == "__main__":
    main()
