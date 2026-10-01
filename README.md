# Debutanizer Butane Soft Sensor

Machine learning based soft sensor for predicting butane (C4) concentration
in the bottoms product of an industrial debutanizer using time-series process data.

## Problem

Butane concentration is measured with a significant delay, while process
variables such as temperatures, pressures and flows are available continuously.
The objective is to estimate butane concentration from these faster measurements.

## Approach

- Time-series data analysis and data-quality checks
- Lag correlation analysis
- Lagged feature engineering
- Time-based train/test split
- TimeSeriesSplit cross-validation
- Ridge Regression, Random Forest and XGBoost
- Hyperparameter tuning with GridSearchCV
- Evaluation using R², RMSE and MAE
- Permutation-based variable importance

Two feature sets were compared:

- **Version A:** Lagged process measurements only
- **Version B:** Process measurements + previous butane measurements

## Results

| Model | Version | Test R² |
|---|---|---:|
| Random Forest | A — no past y | **0.491** |
| Ridge | B — with past y | **0.952** |
| Random Forest | B — with past y | **0.900** |
| XGBoost | B — with past y | **0.923** |

The process-measurement-only model achieved a test R² of **0.491**, while
including previous butane measurements increased the Ridge model's test R²
to **0.952**.

## Technologies

Python · Pandas · NumPy · Scikit-learn · XGBoost · Matplotlib · Seaborn

## Files

- `debutanizer_soft_sensor.ipynb` — Complete analysis and ML workflow
- `debutanizer_data.csv` — Industrial process dataset
