from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    data_root: Path
    train_csv_path: Path
    sample_submission_path: Path
    train_2ch_dir: Path
    train_4ch_dir: Path
    test_2ch_dir: Path
    test_4ch_dir: Path
    results_dir: Path
    figures_dir: Path
    metrics_dir: Path
    models_dir: Path


def resolve_project_root(start_path: str | Path | None = None) -> Path:
    if start_path is None:
        return Path(__file__).resolve().parent.parent

    start = Path(start_path).resolve()
    return start.parent if start.name == "notebooks" else start


def build_project_paths(project_root: str | Path | None = None) -> ProjectPaths:
    root = resolve_project_root(project_root)
    data_root = root / "data" / "raw" / "echo2022"
    results_dir = root / "results"
    figures_dir = results_dir / "figures"
    metrics_dir = results_dir / "metrics"
    models_dir = results_dir / "models"

    for directory in (figures_dir, metrics_dir, models_dir):
        directory.mkdir(parents=True, exist_ok=True)

    return ProjectPaths(
        project_root=root,
        data_root=data_root,
        train_csv_path=data_root / "train_data.csv",
        sample_submission_path=data_root / "sample_submission.csv",
        train_2ch_dir=data_root / "train_data" / "2CH",
        train_4ch_dir=data_root / "train_data" / "4CH",
        test_2ch_dir=data_root / "test_data" / "2CH",
        test_4ch_dir=data_root / "test_data" / "4CH",
        results_dir=results_dir,
        figures_dir=figures_dir,
        metrics_dir=metrics_dir,
        models_dir=models_dir,
    )


def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
