# Echo Project Presentation Pack

## Executive Summary

- Repository goal: reproduce and analyze a clean EF regression baseline for ECHO2022.
- Validation RMSE: 11.16
- Validation MAE: 9.05
- Validation R2: -0.04
- Prediction mean/std: 51.18 / 0.0010
- Target mean/std: 53.41 / 11.01

## Dataset Snapshot

- Local training rows: 400
- Available echo views: 2CH and 4CH
- Baseline uses only 4CH sequences
- Baseline uses first 10 frames resized to 256x256

## Baseline Diagnosis

- The baseline predictions collapse near ~51 EF.
- The prediction spread is almost zero compared with the target spread.
- The scatter plot shows weak correlation with ground truth labels.
- Errors are smallest in the mid-EF region and much larger at both extremes.
- Likely causes: limited temporal sampling, single-view input, simple architecture, no chamber-focused segmentation.

## EF-Group Error Summary

- Low EF (<40): n=9, mean abs error=17.72, mean prediction=51.18
- Mid EF (40-55): n=32, mean abs error=4.15, mean prediction=51.18
- High EF (>=55): n=39, mean abs error=11.06, mean prediction=51.17

## Recommended Talking Points

- This baseline is a reproducible starting point, not a clinically reliable system.
- The negative R2 means the model performs worse than predicting the validation mean.
- The model learns the average EF level better than patient-specific EF dynamics.
- A stronger next step is a segmentation-aware or video-native model with richer temporal coverage.

## External References

- Kaggle challenge: https://www.kaggle.com/competitions/echo2022/overview
- Kaggle baseline notebook: https://www.kaggle.com/code/bjoernjostein/echo2022-eda-baseline-model
- Kaggle evaluation notebook: https://www.kaggle.com/code/negmgh/cardiac-echocardiogram-model-evaluation-testing
- EchoNet-Dynamic dataset: https://echonet.github.io/dynamic/
- EchoNet paper: https://doi.org/10.1038/s41551-020-00647-6
