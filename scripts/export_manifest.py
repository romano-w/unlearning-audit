"""Export reproducibility manifest for a run directory."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch

from unlearning_audit.config import ExperimentConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export reproducibility manifest")
    p.add_argument("--output-dir", type=str, default="outputs", help="Base output directory")
    p.add_argument("--run-name", type=str, default="default", help="Run namespace")
    return p.parse_args()


def _git_rev() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        return out
    except Exception:
        return None


def _git_status_porcelain() -> str | None:
    try:
        out = subprocess.check_output(["git", "status", "--short"], text=True).strip()
        return out
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig()
    cfg.output_dir = args.output_dir
    cfg.name = args.run_name

    run_dir = Path(cfg.output_dir) / cfg.name
    report_dir = run_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = report_dir / "manifest.json"

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_rev(),
        "git_status_short": _git_status_porcelain(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "torch": {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "experiment_config": {
            "name": cfg.name,
            "output_dir": cfg.output_dir,
            "data": cfg.data.__dict__,
            "poison": cfg.poison.__dict__,
            "model": cfg.model.__dict__,
            "train": cfg.train.__dict__,
            "unlearn": cfg.unlearn.__dict__,
            "eval": cfg.eval.__dict__,
        },
        "artifacts_expected": {
            "train_clean": str(run_dir / "clean" / "best.pt"),
            "train_poisoned": str(run_dir / "poisoned" / "best.pt"),
            "unlearn_summary": str(run_dir / "unlearn" / "summary.json"),
            "eval_summary": str(run_dir / "eval" / "metrics_summary.json"),
            "analysis_artifacts": str(run_dir / "analysis" / "artifacts.json"),
        },
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("=" * 72)
    print("MANIFEST EXPORTED")
    print("=" * 72)
    print(manifest_path)


if __name__ == "__main__":
    main()
