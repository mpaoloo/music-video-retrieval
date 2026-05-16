from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    """Загружаем конфигурацию
    """
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def project_path(path: str | Path) -> Path:
    """Возвращаем абсолютный путь внутри проекта"""
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def make_dir(path: str | Path) -> Path:
    """Создаем директорию, если она не существует"""
    path = project_path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    """Зафиксируем случайные числа для воспроизводимых экспериментов"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
