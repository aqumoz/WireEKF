import numpy as np
import pandas as pd
from typing import Callable

class WireEKF:
    """
    Extended Kalman Filter for estimating the state of a wire with n nodes in 3D space.
    The state vector is of size 3*n, representing the (x, y, z) positions of each node.
    The filter can optionally use datasets for the model and measurements, or it can accept live data through function callbacks.
    The function callbacks takes a timestamp (float) and returns the corresponding model nodes or measurement nodes as a numpy array of shape (3*n, 1).
    """
    def __init__(
        self,
        n_nodes,
        initial_nodes,
        q_diag=0.01,
        r_diag=0.05,
        get_model_nodes: Callable[[float], np.ndarray] | None = None,
        get_trackdlo_nodes: Callable[[float], np.ndarray] | None = None,
    ):
        """Extended Kalman filter for an n-node wire state."""
        self.n = n_nodes

        initial_nodes = np.asarray(initial_nodes)
        if initial_nodes.shape != (3 * self.n, 1):
            raise ValueError("initial_nodes must have shape (3*n, 1).")

        self.initial_nodes = initial_nodes
        self.x = np.copy(self.initial_nodes)
        self.P = np.eye(3 * self.n) * 0.1
        self.Q = np.eye(3 * self.n) * q_diag
        self.R = np.eye(3 * self.n) * r_diag
        self.C = np.eye(3 * self.n)
        self.get_model_nodes = get_model_nodes
        self.get_trackdlo_nodes = get_trackdlo_nodes
        self.t = 0.0

    def _ensure_column_state(self, nodes: np.ndarray) -> np.ndarray:
        nodes = np.asarray(nodes)
        if nodes.shape != (3 * self.n, 1):
            raise ValueError("Model or measurement nodes must have shape (3*n, 1).")
        return nodes

    def predict(self, u_t: float, model_nodes: np.ndarray | None = None) -> np.ndarray:
        """Prediction step using provided model nodes or get_model_nodes callback."""
        if model_nodes is None and self.get_model_nodes is not None:
            model_nodes = self.get_model_nodes(self.t)
        elif model_nodes is None:
            raise ValueError("No model nodes provided for prediction.")

        x_model = self._ensure_column_state(model_nodes)

        # print(f"Predicting with model nodes at time {self.t:.2f}, x_model = {x_model.flatten()}")

        A = np.eye(3 * self.n)
        self.x = self.x + 0.5 * (x_model - self.x)
        self.P = A @ self.P @ A.T + self.Q
        return self.x

    def update(self, t: float, trackdlo_nodes: np.ndarray | None = None) -> np.ndarray:
        """Update step using provided trackdlo_nodes or get_trackdlo_nodes callback."""
        if trackdlo_nodes is None and self.get_trackdlo_nodes is not None:
            trackdlo_nodes = self.get_trackdlo_nodes(self.t)
        elif trackdlo_nodes is None:
            raise ValueError("No trackdlo_nodes provided for update.")

        y = self._ensure_column_state(trackdlo_nodes)
        innovation = y - (self.C @ self.x)

        S = self.C @ self.P @ self.C.T + self.R
        L = self.P @ self.C.T @ np.linalg.inv(S)

        self.x = self.x + L @ innovation
        self.P = (np.eye(3 * self.n) - L @ self.C) @ self.P

        return self.x

    def reset(self):
        self.x = np.copy(self.initial_nodes)
        self.P = np.eye(3 * self.n) * 0.1
        self.t = 0.0