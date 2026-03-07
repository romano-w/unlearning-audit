"""Aggregate per-seed evaluation summaries into mean/std tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate multi-seed eval results")
    p.add_argument("--seeds", type=str, default="42,43,44", help="Comma-separated seed list")
    p.add_argument(
        "--input-dir",
        type=str,
        default="outputs/seeds",
        help="Directory containing seed_* runs",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="outputs/seeds/aggregate",
        help="Directory for aggregate artifacts",
    )
    return p.parse_args()


def _load_metrics(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _to_markdown(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for _, row in df.iterrows():
        vals = []
        for h in headers:
            v = row[h]
            if pd.isna(v):
                vals.append("")
            elif isinstance(v, float):
                vals.append(f"{v:.6f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    missing = []
    for seed in seeds:
        metrics_path = input_dir / f"seed_{seed}" / "eval" / "metrics_summary.json"
        if not metrics_path.exists():
            missing.append(str(metrics_path))
            continue
        metrics = _load_metrics(metrics_path)
        for model_name, vals in metrics.items():
            rows.append(
                {
                    "seed": seed,
                    "model": model_name,
                    "clean_acc": vals.get("clean_acc"),
                    "asr_acc": vals.get("asr_acc"),
                    "mia_auc": vals.get("mia_auc"),
                    "oracle_gap_clean_acc": vals.get("oracle_gap_clean_acc"),
                    "oracle_gap_asr_acc": vals.get("oracle_gap_asr_acc"),
                }
            )

    if not rows:
        raise FileNotFoundError("No metrics_summary.json files found for provided seeds.")

    raw_df = pd.DataFrame(rows)
    raw_df.to_csv(out_dir / "seed_metrics_raw.csv", index=False)

    agg = (
        raw_df.groupby("model", as_index=False)
        .agg(
            clean_acc_mean=("clean_acc", "mean"),
            clean_acc_std=("clean_acc", "std"),
            asr_acc_mean=("asr_acc", "mean"),
            asr_acc_std=("asr_acc", "std"),
            mia_auc_mean=("mia_auc", "mean"),
            mia_auc_std=("mia_auc", "std"),
            oracle_gap_clean_acc_mean=("oracle_gap_clean_acc", "mean"),
            oracle_gap_clean_acc_std=("oracle_gap_clean_acc", "std"),
            oracle_gap_asr_acc_mean=("oracle_gap_asr_acc", "mean"),
            oracle_gap_asr_acc_std=("oracle_gap_asr_acc", "std"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values("model")
    )
    agg.to_csv(out_dir / "seed_metrics_aggregate.csv", index=False)
    with open(out_dir / "seed_metrics_aggregate.md", "w", encoding="utf-8") as f:
        f.write(_to_markdown(agg))

    summary = {
        "seeds_requested": seeds,
        "seeds_found": sorted(raw_df["seed"].unique().tolist()),
        "missing_metrics_files": missing,
        "raw_csv": str(out_dir / "seed_metrics_raw.csv"),
        "aggregate_csv": str(out_dir / "seed_metrics_aggregate.csv"),
        "aggregate_md": str(out_dir / "seed_metrics_aggregate.md"),
    }
    with open(out_dir / "aggregate_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 72)
    print("SEED AGGREGATION COMPLETE")
    print("=" * 72)
    print(f"Raw      : {out_dir / 'seed_metrics_raw.csv'}")
    print(f"Aggregate: {out_dir / 'seed_metrics_aggregate.csv'}")
    print(f"Markdown : {out_dir / 'seed_metrics_aggregate.md'}")
    if missing:
        print("Missing files:")
        for m in missing:
            print(f"  - {m}")


if __name__ == "__main__":
    main()
