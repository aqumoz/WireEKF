# About

This is a lightweight implementation of an Extended Kalman Filter (EKF) for state estimation in nonlinear systems.
The interactive viewer can be used for visualizing the EKF output and interactively step through a dataset using the numbers [1]-[6] on the keyboard to step 1, 5, 10, 100, 500 and 1000 times, and [R] for resetting to step 0.
The `SimModelAndTrackDLO` class is used to load in the datasets of either the model, TrackDLO or both.

## Features

- Prediction and update steps for nonlinear process and measurement models
- Configurable process and measurement noise covariance
- Support for typical EKF use cases in robotics, tracking, and sensor fusion
- Optional dataset-driven model and TrackDLO inputs from CSV files

## Setup

1. Clone the repository.
2. Open a terminal in the repository root.
3. Use `uv` to create and activate the project Python environment.

### First-time setup with `uv`

```bash
uv sync
```

This command creates a local virtual environment based on `pyproject.toml` and installs the required dependencies.

## Run the code

From the repository root, run the simulation application with `uv`:

```bash
uv run python InteractiveViewer.py
```

### Example usage

- Use both datasets:

```bash
uv run InteractiveViewer.py --dataset-model data/XPBD_data.csv --dataset-trackdlo data/trackdlo_data.csv --q-diag 0.01 --r-diag 0.05
```

- Use only the model dataset and use live TrackDLO measurements:

```bash
uv run python InteractiveViewer.py --dataset-model data/model_data.csv --q-diag 0.01 --r-diag 0.05
```

- Use only the TrackDLO dataset and live model input:

```bash
uv run python InteractiveViewer.py --dataset-trackdlo data/trackdlo_data.csv --q-diag 0.01 --r-diag 0.05
```

## Project Structure

- `InteractiveViewer.py` - entry point for the EKF application
- `ExtendedKalmanFilters.py` - EKF implementation
- `data/` - sample and dataset CSV files
