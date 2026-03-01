"""Experiment configuration using dataclasses.

All experiment parameters live here as structured configs that Hydra
can instantiate from YAML.  Keeping a single source of truth avoids
scattered magic numbers and makes sweeps straightforward.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DataConfig:
    dataset: str = "cifar10"
    data_dir: str = "./data"
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class PoisonConfig:
    method: str = "badnets"
    target_class: int = 0
    poison_ratio: float = 0.05
    trigger_size: int = 3
    trigger_position: str = "bottom_right"  # top_left | top_right | bottom_left | bottom_right
    trigger_value: float = 1.0  # pixel intensity of the patch (0-1 range)


@dataclass
class ModelConfig:
    arch: str = "resnet18"
    num_classes: int = 10
    pretrained: bool = False


@dataclass
class TrainConfig:
    epochs: int = 200
    batch_size: int = 128
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    lr_schedule: str = "cosine"  # cosine | multistep
    lr_milestones: list[int] = field(default_factory=lambda: [100, 150])
    lr_gamma: float = 0.1
    seed: int = 42


@dataclass
class UnlearnConfig:
    method: str = "neggrad"  # neggrad | ssd | retrain
    epochs: int = 10
    lr: float = 0.01
    batch_size: int = 128
    # NegGrad+ specific
    alpha: float = 1.0  # weight for gradient ascent on forget set
    # SSD specific
    ssd_lambda: float = 1.0
    ssd_alpha: float = 50.0


@dataclass
class EvalConfig:
    batch_size: int = 256
    # Trigger-family probe: test trigger at multiple positions and sizes
    probe_trigger_positions: list[str] = field(
        default_factory=lambda: [
            "bottom_right",
            "bottom_left",
            "top_right",
            "top_left",
            "center",
        ]
    )
    probe_trigger_sizes: list[int] = field(default_factory=lambda: [2, 3, 4, 5])
    # Reactivation probe
    reactivation_samples: int = 50
    reactivation_steps: int = 100
    reactivation_lr: float = 0.001


@dataclass
class ExperimentConfig:
    name: str = "default"
    output_dir: str = "./outputs"
    device: str = "auto"  # auto | cuda | cpu
    data: DataConfig = field(default_factory=DataConfig)
    poison: PoisonConfig = field(default_factory=PoisonConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    unlearn: UnlearnConfig = field(default_factory=UnlearnConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def resolve_device(cfg: ExperimentConfig) -> str:
    """Return the actual torch device string."""
    import torch

    if cfg.device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return cfg.device
