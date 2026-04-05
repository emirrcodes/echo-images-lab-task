from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import History

from src.preprocess import SequenceConfig, load_sequence


def _save_or_show(output_path: str | Path | None = None) -> None:
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_training_history(
    history: History,
    output_path: str | Path | None = None,
) -> None:
    plt.figure(figsize=(14, 4))

    plt.subplot(1, 3, 1)
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(history.history["rmse"], label="train_rmse")
    plt.plot(history.history["val_rmse"], label="val_rmse")
    plt.title("RMSE")
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(history.history["mae"], label="train_mae")
    plt.plot(history.history["val_mae"], label="val_mae")
    plt.title("MAE")
    plt.legend()

    plt.tight_layout()
    _save_or_show(output_path)


def plot_prediction_scatter(
    targets: np.ndarray,
    predictions: np.ndarray,
    output_path: str | Path | None = None,
) -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(targets, predictions, alpha=0.7)
    plt.plot(
        [np.min(targets), np.max(targets)],
        [np.min(targets), np.max(targets)],
        linestyle="--",
    )
    plt.xlabel("Ground Truth EF")
    plt.ylabel("Predicted EF")
    plt.title("Validation: Ground Truth vs Prediction")
    plt.tight_layout()
    _save_or_show(output_path)


def plot_error_histogram(
    results_df: pd.DataFrame,
    column: str,
    title: str,
    xlabel: str,
    output_path: str | Path | None = None,
    bins: int = 20,
) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(results_df[column], bins=bins)
    if column == "error":
        plt.axvline(0, linestyle="--")
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    _save_or_show(output_path)


def show_case_frames(
    case_row: pd.Series,
    config: SequenceConfig | None = None,
    title_prefix: str = "",
    frame_indices: tuple[int, ...] = (0, 3, 6, 9),
    output_path: str | Path | None = None,
) -> None:
    config = config or SequenceConfig()
    sequence = load_sequence(
        case_row["filepath"],
        n_frames=config.n_frames,
        img_size=config.img_size,
    )

    plt.figure(figsize=(14, 3))
    for plot_index, frame_index in enumerate(frame_indices, start=1):
        safe_index = min(frame_index, config.n_frames - 1)
        plt.subplot(1, len(frame_indices), plot_index)
        plt.imshow(sequence[safe_index, :, :, 0], cmap="gray")
        plt.title(f"Frame {safe_index}")
        plt.axis("off")

    plt.suptitle(
        (
            f"{title_prefix} | {case_row['patient_id']} | "
            f"GT={case_row['label']:.2f}, "
            f"Pred={case_row['prediction']:.2f}, "
            f"AbsErr={case_row['abs_error']:.2f}"
        ),
        y=1.05,
    )
    plt.tight_layout()
    _save_or_show(output_path)
