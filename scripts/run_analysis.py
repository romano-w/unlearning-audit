"""Generate Phase 4 analysis artifacts (figures + summary tables)."""

from __future__ import annotations

import argparse
from pathlib import Path

from unlearning_audit.analysis.plotting import generate_all_plots


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate analysis plots and tables")
    p.add_argument(
        "--eval-dir",
        type=str,
        default="outputs/default/eval",
        help="Directory containing evaluation JSON outputs",
    )
    p.add_argument(
        "--analysis-dir",
        type=str,
        default="outputs/default/analysis",
        help="Directory to save plots/tables",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    analysis_dir = Path(args.analysis_dir)
    artifacts = generate_all_plots(eval_dir=eval_dir, analysis_dir=analysis_dir)

    print("=" * 72)
    print("ANALYSIS COMPLETE")
    print("=" * 72)
    print(f"Eval dir    : {eval_dir}")
    print(f"Analysis dir: {analysis_dir}")
    if not artifacts:
        print("No artifacts generated (missing eval outputs).")
        return
    print("Artifacts:")
    for key, value in artifacts.items():
        if isinstance(value, list):
            print(f"  - {key}:")
            for path in value:
                print(f"      {path}")
        else:
            print(f"  - {key}: {value}")


if __name__ == "__main__":
    main()
