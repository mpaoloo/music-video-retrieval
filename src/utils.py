from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    """Load a small YAML config.

    The function accepts both absolute paths and paths relative to the project
    root. This is convenient because scripts may be launched from Colab or from
    a local terminal.
    """
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def project_path(path: str | Path) -> Path:
    """Return an absolute path inside the project folder."""
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def make_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist and return it as Path."""
    path = project_path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    """Fix random seeds for more reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
