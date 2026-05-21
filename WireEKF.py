import numpy as np
from typing import Callable
from ColumnMap import ColumnMap, Data

class WireEKF:
    def __init__(
        self,
        n_nodes: int,
        initial_state: Data,
        get_model_state: Callable[[Data, float], Data],
        q_diag=0.01,
        r_diag=0.05,
    ):
        self.get_model_state = get_model_state
        self.current_state = initial_state
        self.predicted_data = initial_state
        self.n = n_nodes
        self.dim_per_point = 9
        self.total_dim = self.n * self.dim_per_point
        
        # State vector x: [pos, vel, acc] for each node
        self.x = np.zeros((self.total_dim, 1))
        self._initialize_x(initial_state)
        
        # Covariance matrices
        self.P = np.eye(self.total_dim)
        self.Q_base = np.eye(self.total_dim) * q_diag
        self.r_diag = r_diag
        
        self.prev_time = initial_state.time

    def _initialize_x(self, data: Data):
        """Seed the state vector with initial positions."""
        for i, pos in data.p.items():
            # Data keys are 1-indexed based on ColumnMap
            idx = (i - 1) * self.dim_per_point
            if idx < self.total_dim:
                self.x[idx : idx + 3, 0] = pos

    def _get_jacobian_f(self, dt: float) -> np.ndarray:
        """
        Calculates the State Transition Jacobian F for a 
        constant acceleration kinematic model.
        """
        F = np.eye(self.total_dim)
        for i in range(self.n):
            idx = i * self.dim_per_point
            # pos = pos + v*dt + 0.5*a*dt^2
            F[idx : idx + 3, idx + 3 : idx + 6] = np.eye(3) * dt
            F[idx : idx + 3, idx + 6 : idx + 9] = np.eye(3) * (0.5 * dt**2)
            # vel = vel + a*dt
            F[idx + 3 : idx + 6, idx + 6 : idx + 9] = np.eye(3) * dt
        return F

    def _x_to_data(self, timestamp: float) -> Data:
        """Helper to package the internal x vector back into a Data object."""
        new_data = Data()
        new_data.time = timestamp
        # Carry over unmodeled fields from the last known state
        new_data.ft = self.current_state.ft
        new_data.torque = self.current_state.torque
        new_data.q = self.current_state.q  # quaternions are not in the EKF state vector
        
        for i in range(1, self.n + 1):
            idx = (i - 1) * self.dim_per_point
            new_data.p[i] = tuple(self.x[idx : idx + 3, 0])
            new_data.v[i] = tuple(self.x[idx + 3 : idx + 6, 0])
        return new_data


    def predict(self, current_time: float):
        """
        Projects the state and covariance forward in time.
        """
        
        dt = current_time - self.prev_time
        if dt <= 0:
            return 
        
        # 1. State Prediction via callback
        # The callback provides the non-linear physics prediction
        self.predicted_data = self.get_model_state(self.current_state, dt)
        
        # Update internal x with predicted positions
        # (Velocities and accelerations persist or are updated by the callback)
        for i, pos in self.predicted_data.p.items():
            idx = (i - 1) * self.dim_per_point
            if idx < self.total_dim:
                self.x[idx : idx + 3, 0] = pos

        # 2. Covariance Prediction
        F = self._get_jacobian_f(dt)
        # Scale Q by dt to account for uncertainty growth over time
        self.P = F @ self.P @ F.T + (self.Q_base * dt)
        
        self.prev_time = current_time
        self.current_state = self._x_to_data(current_time)


    def update(self, measurement: Data):
        """
        Corrects the state estimate using sensor measurements.
        """
        z_list = []
        h_rows = []

        # Dynamically build H and z based on which points are in the measurement
        for i, meas_pos in measurement.p.items():
            node_idx = i - 1
            if node_idx >= self.n:
                continue
            
            # Measurement vector z
            z_list.append(np.array(meas_pos).reshape(3, 1))
            
            # Jacobian H: maps 9 states per node to 3 measured position values
            H_sub = np.zeros((3, self.total_dim))
            start_col = node_idx * self.dim_per_point
            H_sub[:, start_col : start_col + 3] = np.eye(3)
            h_rows.append(H_sub)

        if not h_rows:
            return # Nothing to update

        Z = np.vstack(z_list)
        H = np.vstack(h_rows)
        
        # Innovation (Residual)
        y = Z - (H @ self.x)
        
        # Innovation Covariance
        R = np.eye(Z.shape[0]) * self.r_diag
        S = H @ self.P @ H.T + R
        
        # Kalman Gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update State and Covariance
        self.x = self.x + K @ y
        I = np.eye(self.total_dim)
        self.P = (I - K @ H) @ self.P
        
        # Update Data object and time
        self.current_state = self._x_to_data(measurement.time)
        self.prev_time = measurement.time