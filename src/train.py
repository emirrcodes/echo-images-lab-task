from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

from src.preprocess import SequenceConfig


@dataclass(frozen=True)
class BaselineConfig(SequenceConfig):
    epochs: int = 10
    learning_rate: float = 1e-3
    dropout_rate: float = 0.2
    patience: int = 3
    validation_size: float = 0.2


def build_baseline_model(config: BaselineConfig | None = None) -> tf.keras.Model:
    config = config or BaselineConfig()
    height, width = config.img_size

    model = models.Sequential(
        [
            layers.Input(shape=(config.n_frames, height, width, 1)),
            layers.TimeDistributed(
                layers.Conv2D(32, (3, 3), activation="relu", padding="same")
            ),
            layers.TimeDistributed(layers.BatchNormalization()),
            layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
            layers.TimeDistributed(
                layers.Conv2D(64, (3, 3), activation="relu", padding="same")
            ),
            layers.TimeDistributed(layers.BatchNormalization()),
            layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
            layers.TimeDistributed(
                layers.Conv2D(128, (3, 3), activation="relu", padding="same")
            ),
            layers.TimeDistributed(layers.GlobalAveragePooling2D()),
            layers.LSTM(64, return_sequences=False),
            layers.Dense(32, activation="relu"),
            layers.Dropout(config.dropout_rate),
            layers.Dense(1, activation="linear"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse"), "mae"],
    )
    return model


def build_training_callbacks(
    models_dir: str | Path,
    config: BaselineConfig | None = None,
    model_filename: str = "baseline_best.keras",
) -> list[tf.keras.callbacks.Callback]:
    config = config or BaselineConfig()
    checkpoint_path = Path(models_dir) / model_filename

    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]


def history_to_frame(history: tf.keras.callbacks.History) -> pd.DataFrame:
    return pd.DataFrame(history.history)
