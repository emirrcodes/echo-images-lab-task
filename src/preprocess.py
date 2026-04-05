from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence


@dataclass(frozen=True)
class SequenceConfig:
    n_frames: int = 10
    img_size: tuple[int, int] = (256, 256)
    batch_size: int = 8


def load_sequence(
    filepath: str | Path,
    n_frames: int = 10,
    img_size: tuple[int, int] = (256, 256),
) -> np.ndarray:
    sequence = np.load(filepath)

    if sequence.shape[0] == 0:
        raise ValueError(f"Sequence file {filepath} is empty.")

    selected = sequence[:n_frames]

    if selected.shape[0] < n_frames:
        pad_count = n_frames - selected.shape[0]
        pad_frame = selected[-1]
        padding = np.stack([pad_frame] * pad_count, axis=0)
        selected = np.concatenate([selected, padding], axis=0)

    height, width = img_size
    resized = np.zeros((n_frames, height, width), dtype=np.float32)

    for index, frame in enumerate(selected):
        resized[index] = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

    max_value = float(resized.max())
    if max_value > 0:
        resized = resized / max_value

    return np.expand_dims(resized, axis=-1)


class EchoSequenceGenerator(Sequence):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        config: SequenceConfig | None = None,
        shuffle: bool = True,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.config = config or SequenceConfig()
        self.shuffle = shuffle
        self.indices = np.arange(len(self.dataframe))
        self.on_epoch_end()

    def __len__(self) -> int:
        return math.ceil(len(self.dataframe) / self.config.batch_size)

    def __getitem__(self, batch_index: int) -> tuple[np.ndarray, np.ndarray]:
        start = batch_index * self.config.batch_size
        stop = start + self.config.batch_size
        batch_indices = self.indices[start:stop]
        batch_df = self.dataframe.iloc[batch_indices]

        height, width = self.config.img_size
        batch_x = np.zeros(
            (len(batch_df), self.config.n_frames, height, width, 1),
            dtype=np.float32,
        )
        batch_y = np.zeros((len(batch_df), 1), dtype=np.float32)

        for row_index, (_, row) in enumerate(batch_df.iterrows()):
            batch_x[row_index] = load_sequence(
                row["filepath"],
                n_frames=self.config.n_frames,
                img_size=self.config.img_size,
            )
            batch_y[row_index, 0] = float(row["label"])

        return batch_x, batch_y

    def on_epoch_end(self) -> None:
        if self.shuffle:
            np.random.shuffle(self.indices)
