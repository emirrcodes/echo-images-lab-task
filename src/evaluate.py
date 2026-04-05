from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Model

from src.preprocess import SequenceConfig, load_sequence


def predict_dataframe(
    model: Model,
    dataframe: pd.DataFrame,
    config: SequenceConfig | None = None,
) -> np.ndarray:
    config = config or SequenceConfig()
    predictions: list[float] = []

    for _, row in dataframe.iterrows():
        sequence = load_sequence(
            row["filepath"],
            n_frames=config.n_frames,
            img_size=config.img_size,
        )
        prediction = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0, 0]
        predictions.append(float(prediction))

    return np.array(predictions, dtype=np.float32)


def compute_regression_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(targets, predictions))),
        "mae": float(mean_absolute_error(targets, predictions)),
        "r2": float(r2_score(targets, predictions)),
    }


def attach_predictions(
    dataframe: pd.DataFrame,
    predictions: np.ndarray,
    label_column: str = "label",
) -> pd.DataFrame:
    results_df = dataframe.copy().reset_index(drop=True)
    results_df["prediction"] = predictions
    results_df["error"] = results_df["prediction"] - results_df[label_column]
    results_df["abs_error"] = results_df["error"].abs()
    return results_df


def summarize_extreme_cases(
    results_df: pd.DataFrame,
    n_cases: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    best_cases = results_df.sort_values("abs_error").head(n_cases).reset_index(drop=True)
    worst_cases = (
        results_df.sort_values("abs_error", ascending=False).head(n_cases).reset_index(drop=True)
    )
    return best_cases, worst_cases


def build_submission(
    sample_submission: pd.DataFrame,
    predictions: np.ndarray,
) -> pd.DataFrame:
    submission = sample_submission.copy()
    target_columns = [column for column in submission.columns if column != submission.columns[0]]
    if len(target_columns) != 1:
        raise ValueError("Expected a single target column in the sample submission file.")

    submission[target_columns[0]] = predictions
    return submission


def save_dataframe(dataframe: pd.DataFrame, output_path: str | Path) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
