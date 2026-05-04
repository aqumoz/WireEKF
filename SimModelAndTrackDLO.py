import numpy as np
import pandas as pd


class SimModelAndTrackDLO:
    """Manages model and TrackDLO datasets, providing callbacks for WireEKF."""
    
    def __init__(self, n_nodes: int, model_path: str | None = None, trackdlo_path: str | None = None):
        self.n = n_nodes
        self.model_dataset = None
        self.trackdlo_dataset = None
        self.model_index = 0
        self.trackdlo_index = 0
        
        if model_path is not None:
            self.model_dataset = self._load_dataset(model_path)
        if trackdlo_path is not None:
            self.trackdlo_dataset = self._load_dataset(trackdlo_path)
    
    def _load_dataset(self, path: str) -> np.ndarray:
        """Load CSV and reshape to (num_samples, 3*n, 1)."""
        df = pd.read_csv(path)
        expected_cols = [f"p{i}_{axis}" for i in range(1, self.n + 1) for axis in ("x", "y", "z")]
        missing = [col for col in expected_cols if col not in df.columns]
        if missing:
            raise ValueError(
                f"Dataset at {path} is missing expected node columns: {', '.join(missing)}"
            )
        
        data = df[expected_cols].to_numpy(dtype=float)
        return data.reshape((-1, 3 * self.n, 1))
    
    def get_model_nodes(self, timestamp: float | None = None) -> np.ndarray:
        """Retrieve next model nodes as (3*n, 1)."""
        if self.model_dataset is None:
            raise ValueError("No model dataset loaded.")
        if self.model_index >= len(self.model_dataset):
            raise IndexError("Model dataset exhausted.")
        nodes = self.model_dataset[self.model_index]
        self.model_index += 1
        return nodes
    
    def get_trackdlo_nodes(self, timestamp: float | None = None) -> np.ndarray:
        """Retrieve next TrackDLO nodes as (3*n, 1)."""
        if self.trackdlo_dataset is None:
            raise ValueError("No TrackDLO dataset loaded.")
        if self.trackdlo_index >= len(self.trackdlo_dataset):
            raise IndexError("TrackDLO dataset exhausted.")
        nodes = self.trackdlo_dataset[self.trackdlo_index]
        self.trackdlo_index += 1
        return nodes
    
    def reset(self):
        """Reset dataset indices."""
        self.model_index = 0
        self.trackdlo_index = 0

    
    