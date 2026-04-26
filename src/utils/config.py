"""Project-wide configuration."""

from dataclasses import dataclass, field
from pathlib import Path

import torch


def detect_device():
    """Pick the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def project_root():
    """Return the project root (two levels up from this file)."""
    return Path(__file__).resolve().parent.parent.parent


@dataclass
class Config:
    """Central settings for the experiments."""

    model_name: str = "meta-llama/Llama-3.2-1B"
    seed: int = 42
    batch_size: int = 8
    max_seq_len: int = 512
    use_4bit: bool = True

    # paths (derived from project root)
    project_root: Path = field(default_factory=project_root)

    # device (auto-detected)
    device: torch.device = field(default_factory=detect_device)

    @property
    def data_dir(self):
        return self.project_root / "data"

    @property
    def output_dir(self):
        return self.project_root / "outputs"

    def __post_init__(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
