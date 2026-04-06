# echo-project

This repository is a cleaned-up baseline for the ECHO2022 echocardiography task.
The current work focuses on estimating `LV_ef` directly from `.npy` echo
sequences with a simple TimeDistributed CNN + LSTM model.

The repo is not an empty environment scaffold anymore. It now has five clear
work areas:

- `notebooks/`: exploration, baseline reproduction, and result inspection for EF regression
- `src/`: reusable Python modules extracted from the baseline notebook logic
- `results/`: baseline regression artifacts
- `improved model/`: dual-view EF regression experiments and comparison artifacts
- `segmentation/`: EchoNet-Dynamic segmentation notebooks, outputs, and Kaggle export files

## Project goal

The practical goal of the current baseline is:

1. inspect the ECHO2022 data format and label structure,
2. train a reproducible regression baseline on 4CH sequences,
3. evaluate that baseline on a local validation split,
4. save artifacts that make later model iterations easier to compare.

This is a baseline repo, not a finished clinical system. The saved validation
results show that the model often collapses toward the dataset mean and still
misses hard cases by a large margin.

Current saved validation metrics from `results/metrics/baseline_validation_metrics.csv`:

- `RMSE`: `11.1648`
- `MAE`: `9.0469`
- `R2`: `-0.0418`
- split size: `320` train / `80` validation
- baseline setup: `4CH` only, first `10` frames, resized to `256x256`

## Dataset snapshot

The checked local dataset under `data/raw/echo2022/` currently contains:

- `train_data.csv`: `400 x 2`
- `sample_submission.csv`: `50 x 2`
- `train_data/2CH`: `400` sequences
- `train_data/4CH`: `400` sequences
- `test_data/2CH`: `50` sequences
- `test_data/4CH`: `50` sequences

## Notebooks

### `notebooks/01_data_exploration.ipynb`

Purpose:

- inspect raw sequence shapes and frame counts,
- compare 2CH and 4CH views,
- inspect the `LV_ef` label distribution,
- visualize example frames from the sequences.

Main outputs:

- in-notebook histograms for sequence lengths and image sizes,
- sample frame visualizations for 2CH and 4CH sequences,
- working assumptions that justify resizing, frame truncation/padding, and
  direct EF regression.

### `notebooks/02_baseline_reproduction.ipynb`

Purpose:

- reproduce a simple baseline training pipeline from local data,
- create a train/validation split,
- train a TimeDistributed CNN + LSTM regressor,
- evaluate validation performance and generate a Kaggle-style submission file.

Generated outputs currently available in this workspace:

- `results/models/baseline_best.keras`
- `results/metrics/baseline_history.csv`
- `results/metrics/baseline_validation_metrics.csv`
- `results/baseline_submission.csv`
- `results/figures/baseline_training_curves.png`
- `results/figures/val_gt_vs_pred.png`

### `notebooks/03_results_visualization.ipynb`

Purpose:

- reload the saved baseline model,
- recompute validation predictions,
- analyze error patterns,
- inspect the best and worst prediction cases visually.

Generated outputs currently available in this workspace:

- `results/metrics/validation_predictions.csv`
- `results/metrics/top10_best_validation_cases.csv`
- `results/metrics/top10_worst_validation_cases.csv`
- `results/figures/val_gt_vs_pred_recomputed.png`
- `results/figures/val_error_distribution.png`
- `results/figures/val_abs_error_distribution.png`
- `results/figures/best_case_1.png`
- `results/figures/best_case_2.png`
- `results/figures/best_case_3.png`
- `results/figures/worst_case_1.png`
- `results/figures/worst_case_2.png`
- `results/figures/worst_case_3.png`

### `improved model/01_improved_training.ipynb`

Purpose:

- extend the baseline into a dual-view `2CH + 4CH` regression setup,
- test more stable preprocessing and training choices,
- compare the deep model against naive and handcrafted baselines.

Generated outputs currently available in this workspace:

- `improved model/results/metrics/model_comparison.csv`
- `improved model/results/metrics/prediction_summary.csv`
- `improved model/results/metrics/validation_predictions.csv`
- `improved model/results/figures/improved_training_curves.png`
- `improved model/results/figures/validation_overview.png`

### `improved model/02_improved_results_visualization.ipynb`

Purpose:

- reload the saved dual-view model,
- recompute validation summaries,
- visualize best and worst cases from the improved experiment.

### `segmentation/notebooks/01_dataset_audit.ipynb`

Purpose:

- inspect the EchoNet-Dynamic segmentation dataset layout,
- verify `FileList.csv`, `VolumeTracings.csv`, and the `Videos/` folder,
- quantify which studies have usable tracings for mask generation.

### `segmentation/notebooks/02_tracing_to_mask_sanity_checks.ipynb`

Purpose:

- convert tracing polygons into raster masks,
- visually sanity-check mask alignment against source frames,
- validate the preprocessing assumptions before training a segmentation model.

### `segmentation/notebooks/03_unet_baseline_training.ipynb`

Purpose:

- train a PyTorch U-Net baseline for left ventricle segmentation,
- use the EchoNet-Dynamic tracings as supervision,
- save training outputs for later qualitative and quantitative comparison.

### `segmentation/notebooks/04_model_evaluation_and_error_analysis.ipynb`

Purpose:

- reload the saved segmentation checkpoint,
- compute split-level and phase-level metrics,
- summarize qualitative successes and failure cases.

## Source modules

The reusable notebook code has been organized under `src/`:

- `src/utils.py`: seed handling and project path management
- `src/data_loader.py`: metadata loading, label mapping, file discovery, split preparation
- `src/preprocess.py`: sequence loading, resizing, normalization, and batch generation
- `src/train.py`: baseline configuration, model construction, and callbacks
- `src/evaluate.py`: prediction loops, regression metrics, and submission helpers
- `src/visualize.py`: training curves, prediction plots, error histograms, and case views

This makes the repo easier to explain: notebooks stay narrative, while `src/`
holds the logic that can be reused in future scripts or cleaner notebook
revisions.

## Repository structure

```text
echo-project/
|-- data/
|   |-- raw/echo2022/
|   |   |-- train_data.csv
|   |   |-- sample_submission.csv
|   |   |-- train_data/
|   |   |   |-- 2CH/
|   |   |   `-- 4CH/
|   |   `-- test_data/
|   |       |-- 2CH/
|   |       `-- 4CH/
|   `-- processed/
|-- notebooks/
|   |-- 01_data_exploration.ipynb
|   |-- 02_baseline_reproduction.ipynb
|   `-- 03_results_visualization.ipynb
|-- improved model/
|   |-- 01_improved_training.ipynb
|   |-- 02_improved_results_visualization.ipynb
|   `-- results/
|-- results/
|   |-- figures/
|   |-- metrics/
|   |-- models/
|   `-- baseline_submission.csv
|-- segmentation/
|   |-- notebooks/
|   |-- results/
|   `-- src/
|-- src/
|   |-- data_loader.py
|   |-- evaluate.py
|   |-- preprocess.py
|   |-- train.py
|   |-- utils.py
|   `-- visualize.py
|-- requirements.txt
`-- README.md
```

## Recommended setup

Use Python 3.12.x for the most predictable TensorFlow + Jupyter behavior.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name echo-project --display-name "echo-project"
```

For the segmentation notebooks, the same environment also needs PyTorch and
`scikit-image`, which are now included in `requirements.txt`. If the notebook
kernel was created before updating dependencies, reactivate the environment and
rerun `pip install -r requirements.txt` before reopening Jupyter. The
segmentation workflow lives under `segmentation/`.

## Minimal usage example

```python
from src.utils import build_project_paths, set_global_seed
from src.data_loader import load_train_metadata, build_label_map, build_train_records, build_train_val_split
from src.preprocess import EchoSequenceGenerator
from src.train import BaselineConfig, build_baseline_model

paths = build_project_paths()
set_global_seed(42)

train_df = load_train_metadata(paths)
label_map = build_label_map(train_df)
records_df = build_train_records(paths.train_4ch_dir, label_map)
train_split, val_split = build_train_val_split(records_df, seed=42)

config = BaselineConfig()
model = build_baseline_model(config)

train_gen = EchoSequenceGenerator(train_split, config=config, shuffle=True)
val_gen = EchoSequenceGenerator(val_split, config=config, shuffle=False)
```

## Current phase status

This phase establishes a cleaner story for the repo:

- the project purpose is explicit,
- notebook outputs are documented,
- the regression, improved-model, and segmentation areas are separated,
- the folder structure is easy to explain,
- baseline logic is no longer trapped inside notebooks.
