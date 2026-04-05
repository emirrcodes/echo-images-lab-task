from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import ProjectPaths


def load_train_metadata(paths: ProjectPaths) -> pd.DataFrame:
    return pd.read_csv(paths.train_csv_path)


def load_sample_submission(paths: ProjectPaths) -> pd.DataFrame:
    return pd.read_csv(paths.sample_submission_path)


def infer_patient_id_column(train_df: pd.DataFrame) -> str:
    possible_id_cols = [
        column
        for column in train_df.columns
        if "patient" in column.lower() or "id" in column.lower()
    ]
    if not possible_id_cols:
        raise ValueError("Could not infer the patient id column from the training metadata.")
    return possible_id_cols[0]


def build_label_map(
    train_df: pd.DataFrame,
    id_column: str | None = None,
    label_column: str = "LV_ef",
) -> dict[str, float]:
    working_df = train_df.copy()
    id_column = id_column or infer_patient_id_column(working_df)

    patient_ids = working_df[id_column].astype(str).str.extract(r"(\d+)", expand=False)
    working_df["patient_id"] = "patient" + patient_ids.str.zfill(3)

    return dict(zip(working_df["patient_id"], working_df[label_column].astype(float)))


def extract_patient_id_from_filename(filepath: str | Path) -> str:
    return Path(filepath).stem.split("_")[0]


def collect_sequence_files(directory: str | Path) -> list[Path]:
    return sorted(Path(directory).glob("*.npy"))


def build_train_records(
    sequence_dir: str | Path,
    label_map: dict[str, float],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for filepath in collect_sequence_files(sequence_dir):
        patient_id = extract_patient_id_from_filename(filepath)
        label = label_map.get(patient_id)
        if label is None:
            continue

        records.append(
            {
                "filepath": filepath,
                "patient_id": patient_id,
                "label": float(label),
            }
        )

    return pd.DataFrame(records)


def build_train_val_split(
    records_df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_split_df, val_split_df = train_test_split(
        records_df,
        test_size=test_size,
        random_state=seed,
    )

    return train_split_df.reset_index(drop=True), val_split_df.reset_index(drop=True)
