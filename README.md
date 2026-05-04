# Extended Kalman Filter

A lightweight implementation of an Extended Kalman Filter (EKF) for state estimation in nonlinear systems.

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

From the repository root, run the main application with `uv`:

```bash
uv run python main.py
```

### Example usage

- Use both datasets:

```bash
uv run main.py --dataset-model data/XPBD_data.csv --dataset-trackdlo data/trackdlo_data.csv
```

- Use only the model dataset and simulate TrackDLO measurements:

```bash
uv run python main.py --dataset-model data/model_data.csv
```

- Use only the TrackDLO dataset and simulate model input:

```bash
uv run python main.py --dataset-trackdlo data/trackdlo_data.csv
```

- Run the filter with a different sensor noise setting:

```bash
uv run python main.py --sensor-var 0.005
```

## Project Structure

- `main.py` - entry point for the EKF application
- `ExtendedKalmanFilters.py` - EKF implementation
- `data/` - sample and dataset CSV files
