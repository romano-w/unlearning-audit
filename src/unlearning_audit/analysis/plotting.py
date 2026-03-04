"""Plotting utilities for Phase 4 analysis outputs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_main_metrics(metrics_summary_path: Path, out_dir: Path) -> Path:
    """Create bar charts for clean accuracy and ASR."""
    data = _read_json(metrics_summary_path)
    rows = []
    for model_name, metrics in data.items():
        rows.append(
            {
                "model": model_name,
                "clean_acc": metrics.get("clean_acc"),
                "asr_acc": metrics.get("asr_acc"),
            }
        )
    df = pd.DataFrame(rows)
    model_order = [m for m in ["poisoned", "neggrad", "ssd", "oracle_retrain", "clean"] if m in df["model"].values]

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.barplot(data=df, x="model", y="clean_acc", order=model_order, ax=axes[0], color="#4C72B0")
    axes[0].set_title("Clean Accuracy (C-Acc)")
    axes[0].set_ylim(0, 1.0)
    axes[0].tick_params(axis="x", rotation=20)

    sns.barplot(data=df, x="model", y="asr_acc", order=model_order, ax=axes[1], color="#C44E52")
    axes[1].set_title("Attack Success Rate (ASR)")
    axes[1].set_ylim(0, 1.0)
    axes[1].tick_params(axis="x", rotation=20)

    fig.suptitle("Backdoor Metrics by Condition", y=1.03)
    fig.tight_layout()
    out_path = out_dir / "main_metrics.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_trigger_family_heatmap(trigger_probe_path: Path, out_dir: Path) -> list[Path]:
    """Create ASR heatmaps per model from trigger-family probe results."""
    data = _read_json(trigger_probe_path)
    saved: list[Path] = []
    for model_name, probe_values in data.items():
        rows = []
        for key, asr in probe_values.items():
            pos, size_part = key.split("|size=")
            rows.append({"position": pos, "size": int(size_part), "asr": float(asr)})
        if not rows:
            continue
        df = pd.DataFrame(rows)
        pivot = df.pivot(index="position", columns="size", values="asr")

        plt.figure(figsize=(6, 4))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap="magma",
            vmin=0.0,
            vmax=1.0,
            cbar_kws={"label": "ASR"},
        )
        plt.title(f"Trigger-Family ASR Heatmap: {model_name}")
        plt.xlabel("Trigger size")
        plt.ylabel("Trigger position")
        plt.tight_layout()
        out_path = out_dir / f"trigger_family_{model_name}.png"
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close()
        saved.append(out_path)
    return saved


def plot_reactivation_curves(reactivation_path: Path, out_dir: Path) -> list[Path]:
    """Plot ASR-vs-finetune-step reactivation curves per method."""
    data = _read_json(reactivation_path)
    saved: list[Path] = []

    for method_name, curves in data.items():
        candidate_curve = pd.DataFrame(curves.get("candidate_curve", []))
        clean_curve = pd.DataFrame(curves.get("clean_curve", []))
        if candidate_curve.empty or clean_curve.empty:
            continue

        plt.figure(figsize=(6, 4))
        plt.plot(
            candidate_curve["step"],
            candidate_curve["asr"],
            marker="o",
            label=f"{method_name} model",
            color="#d62728",
        )
        plt.plot(
            clean_curve["step"],
            clean_curve["asr"],
            marker="o",
            label="clean baseline",
            color="#1f77b4",
        )
        plt.title(f"Reactivation Susceptibility: {method_name}")
        plt.xlabel("Fine-tune step on tiny triggered set")
        plt.ylabel("ASR")
        plt.ylim(0, 1.0)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        out_path = out_dir / f"reactivation_{method_name}.png"
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close()
        saved.append(out_path)
    return saved


def write_summary_table(metrics_summary_path: Path, out_dir: Path) -> tuple[Path, Path]:
    """Write a compact summary table in CSV + Markdown."""
    data = _read_json(metrics_summary_path)
    rows = []
    for model_name, metrics in data.items():
        rows.append(
            {
                "model": model_name,
                "clean_acc": metrics.get("clean_acc"),
                "asr_acc": metrics.get("asr_acc"),
                "mia_auc": metrics.get("mia_auc"),
                "oracle_gap_clean_acc": metrics.get("oracle_gap_clean_acc"),
                "oracle_gap_asr_acc": metrics.get("oracle_gap_asr_acc"),
                "compute_seconds": metrics.get("compute_seconds"),
            }
        )
    df = pd.DataFrame(rows)
    order = [m for m in ["poisoned", "neggrad", "ssd", "oracle_retrain", "clean"] if m in df["model"].values]
    df["model"] = pd.Categorical(df["model"], categories=order, ordered=True)
    df = df.sort_values("model")

    csv_path = out_dir / "summary_table.csv"
    md_path = out_dir / "summary_table.md"
    df.to_csv(csv_path, index=False)
    # Manual markdown rendering to avoid optional `tabulate` dependency.
    headers = list(df.columns)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for _, row in df.iterrows():
            values = []
            for h in headers:
                v = row[h]
                if pd.isna(v):
                    values.append("")
                elif isinstance(v, float):
                    values.append(f"{v:.6f}")
                else:
                    values.append(str(v))
            f.write("| " + " | ".join(values) + " |\n")
    return csv_path, md_path


def generate_all_plots(
    eval_dir: Path = Path("outputs/default/eval"),
    analysis_dir: Path = Path("outputs/default/analysis"),
) -> dict[str, list[str] | str]:
    """Generate all available plots/tables from evaluation artifacts."""
    analysis_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, list[str] | str] = {}

    metrics_path = eval_dir / "metrics_summary.json"
    if metrics_path.exists():
        outputs["main_metrics"] = str(plot_main_metrics(metrics_path, analysis_dir))
        csv_path, md_path = write_summary_table(metrics_path, analysis_dir)
        outputs["summary_table_csv"] = str(csv_path)
        outputs["summary_table_md"] = str(md_path)

    trigger_path = eval_dir / "trigger_family_probe.json"
    if trigger_path.exists():
        outputs["trigger_family_heatmaps"] = [str(p) for p in plot_trigger_family_heatmap(trigger_path, analysis_dir)]

    reactivation_path = eval_dir / "reactivation_probe.json"
    if reactivation_path.exists():
        outputs["reactivation_curves"] = [str(p) for p in plot_reactivation_curves(reactivation_path, analysis_dir)]

    with open(analysis_dir / "artifacts.json", "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2)
    return outputs
