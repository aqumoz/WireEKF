"""
wire_ekf.py
===========
Extended Kalman Filter for the XPBD Cosserat-rod wire model.

State
-----
Per node, 13 elements interleaved for locality:
    [px, py, pz,  qx, qy, qz, qw,  vx, vy, vz,  wx, wy, wz]
     ← position → ←── quaternion ──→ ← lin vel → ← ang vel →

Total state dimension:  n_x = 13 * N  (= 260 for N=20)

Measurement
-----------
TrackDLO gives all N node positions directly:
    y = [px₀, py₀, pz₀,  px₁, py₁, pz₁,  …]

Measurement dimension:  n_y = 3 * N   (= 60 for N=20)

H is analytical — a sparse selection matrix picking the position
block from each node's 13-element slice.

Process Jacobian
----------------
The XPBD step is not analytically differentiable, so F is computed
numerically via central finite differences.  This costs 2 * n_x + 1
simulator evaluations per EKF step.  For offline use this is fine;
for real-time, replace with a block-banded approximation (each node
only strongly couples to its two immediate neighbours).

Quaternion constraint
---------------------
Quaternions are renormalised after every predict and update step.
This is the minimal fix for the unit-norm constraint; it works well
when EKF corrections are small (well-tuned Q and R).

Usage
-----
    from wire_ekf import WireEKF
    from sim_setup import create_sim, gravity_forces

    sim, params = create_sim()
    ekf = WireEKF(sim, params, ee_pos, ee_quat, f_ext)

    # Initialise from first TrackDLO frame
    z = ekf.state_from_trackdlo(trk_nodes[0])
    P = ekf.initial_covariance()

    # Per-timestep loop
    for k in range(n_steps):
        z, P = ekf.predict(z, P)                      # propagate with model
        z, P = ekf.update(z, P, trk_nodes[k])         # fuse TrackDLO
        sim_pts = ekf.node_positions(z)               # (N, 3) filtered shape
"""

import numpy as np
from quaternion_utils import quat_to_rotm, rotm_to_quat, state_from_positions


class WireEKF:
    """
    Extended Kalman Filter wrapping the XPBD wire simulator.

    Parameters
    ----------
    sim       : WireSimulator  — from create_sim()
    params    : WireParams     — from create_sim()
    ee_pos    : (3,)           — EE pin position (world frame)
    ee_quat   : (4,)           — EE pin quaternion [x,y,z,w]
    f_ext     : (6*N,)         — external forces (e.g. from gravity_forces())
    sigma_pos  : float         — process noise std for position  [m]
    sigma_quat : float         — process noise std for quaternion components
    sigma_vel  : float         — process noise std for linear velocity  [m/s]
    sigma_omg  : float         — process noise std for angular velocity [rad/s]
    sigma_meas : float         — TrackDLO measurement noise std  [m]
    jac_eps    : float         — finite-difference step for Jacobian
    """

    # State layout per node (13 elements)
    _POS  = slice(0, 3)    # position   [px, py, pz]
    _QUAT = slice(3, 7)    # quaternion [qx, qy, qz, qw]
    _LVEL = slice(7, 10)   # lin vel    [vx, vy, vz]
    _AVEL = slice(10, 13)  # ang vel    [wx, wy, wz]
    _NODE = 13             # elements per node

    def __init__(
        self,
        sim,
        params,
        ee_pos:    np.ndarray,
        ee_quat:   np.ndarray,
        f_ext:     np.ndarray,
        sigma_pos:  float = 1e-3,
        sigma_quat: float = 1e-4,
        sigma_vel:  float = 1e-2,
        sigma_omg:  float = 1e-2,
        sigma_meas: float = 4e-3,
        jac_eps:    float = 1e-5,
    ):
        self.sim     = sim
        self.params  = params
        self.ee_pos  = ee_pos.copy()
        self.ee_quat = ee_quat.copy()
        self.f_ext   = f_ext.copy()
        self.N       = params.N
        self.jac_eps = jac_eps

        self.n_x = self._NODE * self.N   # 260 for N=20
        self.n_y = 3 * self.N            # 60  for N=20

        # Measurement matrix H — analytical, constant
        self.H = self._build_H()

        # Noise covariances
        self.Q = self._build_Q(sigma_pos, sigma_quat, sigma_vel, sigma_omg)
        self.R = np.eye(self.n_y) * sigma_meas**2


    # ── State packing / unpacking ──────────────────────────────────────────

    def pack(self, x_sim: np.ndarray, v_sim: np.ndarray) -> np.ndarray:
        """
        Pack simulator vectors into the EKF state vector.

        x_sim : (7*N,)  [px,py,pz, qx,qy,qz,qw] * N
        v_sim : (6*N,)  [vx,vy,vz, wx,wy,wz]     * N
        """
        z = np.zeros(self.n_x)
        for i in range(self.N):
            base = i * self._NODE
            z[base + 0:base + 3]  = x_sim[7*i:7*i+3]   # position
            z[base + 3:base + 7]  = x_sim[7*i+3:7*i+7] # quaternion
            z[base + 7:base + 10] = v_sim[6*i:6*i+3]   # lin vel
            z[base + 10:base + 13]= v_sim[6*i+3:6*i+6] # ang vel
        return z

    def unpack(self, z: np.ndarray) -> tuple:
        """
        Unpack EKF state into simulator vectors.

        Returns x_sim (7*N,) and v_sim (6*N,).
        """
        x_sim = np.zeros(7 * self.N)
        v_sim = np.zeros(6 * self.N)
        for i in range(self.N):
            base = i * self._NODE
            x_sim[7*i:7*i+3]   = z[base + 0:base + 3]
            x_sim[7*i+3:7*i+7] = z[base + 3:base + 7]
            v_sim[6*i:6*i+3]   = z[base + 7:base + 10]
            v_sim[6*i+3:6*i+6] = z[base + 10:base + 13]
        return x_sim, v_sim

    def node_positions(self, z: np.ndarray) -> np.ndarray:
        """Extract node positions from state as (N, 3) array."""
        pos = np.zeros((self.N, 3))
        for i in range(self.N):
            pos[i] = z[i * self._NODE: i * self._NODE + 3]
        return pos

    def normalize_quaternions(self, z: np.ndarray) -> np.ndarray:
        """Renormalise all quaternion blocks in-place."""
        z = z.copy()
        for i in range(self.N):
            base = i * self._NODE
            q = z[base + 3:base + 7]
            nrm = np.linalg.norm(q)
            if nrm > 1e-9:
                z[base + 3:base + 7] = q / nrm
        return z

    # ── Initialisation ────────────────────────────────────────────────────

    def state_from_trackdlo(self, trk_frame: np.ndarray) -> np.ndarray:
        """
        Build initial EKF state from a TrackDLO frame.

        trk_frame : (N, 3)  node positions from one TrackDLO frame

        Quaternions are derived from local tangent directions.
        Velocities are initialised to zero.
        """
        x_sim = state_from_positions(trk_frame)
        v_sim = np.zeros(6 * self.N)
        return self.pack(x_sim, v_sim)

    def initial_covariance(
        self,
        sigma_pos:  float = 4e-3,
        sigma_quat: float = 1e-3,
        sigma_vel:  float = 0.1,
        sigma_omg:  float = 0.1,
    ) -> np.ndarray:
        """
        Diagonal initial state covariance P₀.

        Position uncertainty  ~ TrackDLO noise (~4 mm).
        Velocity uncertainty is larger since velocities are not
        directly observed at initialisation.
        """
        diag = np.zeros(self.n_x)
        for i in range(self.N):
            base = i * self._NODE
            diag[base + 0:base + 3]  = sigma_pos**2
            diag[base + 3:base + 7]  = sigma_quat**2
            diag[base + 7:base + 10] = sigma_vel**2
            diag[base + 10:base + 13]= sigma_omg**2
        return np.diag(diag)

    # ── EKF steps ─────────────────────────────────────────────────────────

    def process_step(self, z: np.ndarray) -> np.ndarray:
        """Run one XPBD step and return the new packed state."""
        x_sim, v_sim = self.unpack(z)
        x_new, v_new, _, _ = self.sim.estimate_wire_state(
            x_sim, v_sim, self.ee_pos, self.ee_quat, np.zeros(6), self.f_ext
        )
        return self.pack(x_new, v_new)

    def jacobian_F(self, z: np.ndarray) -> tuple:
        """
        Numerical process Jacobian F = ∂f/∂z via central differences.

        Cost: 2 * n_x simulator evaluations  (= 520 for N=20).
        """
        F      = np.zeros((self.n_x, self.n_x))
        eps    = self.jac_eps
        z_pred = self.process_step(z)

        for j in range(self.n_x):
            z_fwd = z.copy(); z_fwd[j] += eps
            z_bwd = z.copy(); z_bwd[j] -= eps
            z_fwd = self.normalize_quaternions(z_fwd)
            z_bwd = self.normalize_quaternions(z_bwd)
            F[:, j] = (self.process_step(z_fwd) - self.process_step(z_bwd)) / (2 * eps)

        return F, z_pred

    def jacobian_F_banded(self, z: np.ndarray, bandwidth: int = 2) -> tuple:
        """
        Block-banded process Jacobian using node colouring.

        Cost: 2 * (2*bandwidth+1) * 13  evaluations
              bandwidth=1 →  78  (6.7× faster than full 520)
              bandwidth=2 → 130  (4.0× faster)
        """
        stride = 2 * bandwidth + 1
        F      = np.zeros((self.n_x, self.n_x))
        eps    = self.jac_eps
        z_pred = self.process_step(z)

        for color in range(stride):
            nodes = list(range(color, self.N, stride))
            for local_idx in range(self._NODE):
                z_fwd = z.copy()
                z_bwd = z.copy()
                for ni in nodes:
                    gi = ni * self._NODE + local_idx
                    z_fwd[gi] += eps
                    z_bwd[gi] -= eps
                z_fwd = self.normalize_quaternions(z_fwd)
                z_bwd = self.normalize_quaternions(z_bwd)
                deriv = (self.process_step(z_fwd) - self.process_step(z_bwd)) / (2 * eps)
                for ni in nodes:
                    col    = ni * self._NODE + local_idx
                    row_lo = max(0,      ni - bandwidth)     * self._NODE
                    row_hi = min(self.N, ni + bandwidth + 1) * self._NODE
                    F[row_lo:row_hi, col] = deriv[row_lo:row_hi]

        return F, z_pred

    def predict(self, z: np.ndarray, P: np.ndarray,
                mode: str = 'identity',
                bandwidth: int = 2) -> tuple:
        """
        EKF predict step.

        Parameters
        ----------
        mode : str
            'identity' — F = I, P_pred = P + Q.
                         Zero extra sim calls. Fast, works well with
                         small dt and generous Q. Good starting point.

            'banded'   — Block-banded numerical Jacobian.
                         Cost: 2*(2*bandwidth+1)*13 sim calls.
                         More accurate than identity but still slow.

            'full'     — Full dense numerical Jacobian.
                         Cost: 2*n_x sim calls (= 520 for N=20).
                         Most accurate, very slow.

        bandwidth : int
            Only used when mode='banded'. Number of neighbouring nodes
            each node is assumed to influence (start with 2).
        """
        z_pred = self.process_step(z)
        z_pred = self.normalize_quaternions(z_pred)

        if mode == 'identity':
            # F ≈ I: covariance just grows by Q each step
            P_pred = P + self.Q

        elif mode == 'banded':
            F, _   = self.jacobian_F_banded(z, bandwidth)
            P_pred = F @ P @ F.T + self.Q

        elif mode == 'full':
            F, _   = self.jacobian_F(z)
            P_pred = F @ P @ F.T + self.Q

        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'identity', 'banded', or 'full'.")

        return z_pred, P_pred

    def predict_ukf(self, z: np.ndarray, P: np.ndarray,
                    alpha: float = 1e-3,
                    kappa: float = 0.0,
                    beta:  float = 2.0) -> tuple:
        """
        Unscented Kalman Filter predict step.

        Propagates 2*n_x+1 sigma points through the process model instead
        of computing a Jacobian.  Handles nonlinearity better than EKF
        linearisation, but costs 2*n_x+1 = 521 sim calls for N=20.

        UKF scaling parameters (van der Merwe)
        ----------------------------------------
        alpha : spread of sigma points around mean (1e-3 is typical)
        kappa : secondary scaling (0 is typical)
        beta  : prior knowledge of distribution (2 = Gaussian)
        """
        n   = self.n_x
        lam = alpha**2 * (n + kappa) - n

        # Weights
        Wm = np.full(2*n + 1, 1.0 / (2*(n + lam)))
        Wc = np.full(2*n + 1, 1.0 / (2*(n + lam)))
        Wm[0] = lam / (n + lam)
        Wc[0] = lam / (n + lam) + (1 - alpha**2 + beta)

        # Sigma points via Cholesky
        try:
            L = np.linalg.cholesky((n + lam) * P)
        except np.linalg.LinAlgError:
            # P not quite positive-definite — add small jitter
            L = np.linalg.cholesky((n + lam) * (P + 1e-9 * np.eye(n)))

        sigmas = np.zeros((2*n + 1, n))
        sigmas[0] = z
        for i in range(n):
            sigmas[i + 1]     = self.normalize_quaternions(z + L[:, i])
            sigmas[n + i + 1] = self.normalize_quaternions(z - L[:, i])

        # Propagate each sigma point
        sigmas_pred = np.array([self.process_step(s) for s in sigmas])
        sigmas_pred = np.array([self.normalize_quaternions(s) for s in sigmas_pred])

        # Predicted mean and covariance
        z_pred = Wm @ sigmas_pred
        z_pred = self.normalize_quaternions(z_pred)

        P_pred = self.Q.copy()
        for i in range(2*n + 1):
            d = sigmas_pred[i] - z_pred
            P_pred += Wc[i] * np.outer(d, d)

        return z_pred, P_pred

    def update(
        self,
        z_pred:      np.ndarray,
        P_pred:      np.ndarray,
        trk_frame:   np.ndarray,
    ) -> tuple:
        """
        EKF update step using one TrackDLO frame.

        trk_frame : (N, 3)  node positions from TrackDLO

        Returns (z_upd, P_upd).
        """
        # Flatten measurement to (3N,)
        y_meas = trk_frame.flatten()

        # Innovation
        y_pred = self.H @ z_pred
        innov  = y_meas - y_pred

        # Innovation covariance and Kalman gain
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.solve(S, np.eye(self.n_y)).T

        # State and covariance update
        z_upd = z_pred + K @ innov
        I_KH  = np.eye(self.n_x) - K @ self.H
        # Joseph form: numerically more stable than (I-KH)P
        P_upd = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T

        z_upd = self.normalize_quaternions(z_upd)
        return z_upd, P_upd

    def update_ee(
        self,
        ee_pos:  np.ndarray,
        ee_quat: np.ndarray,
    ) -> None:
        """Update the EE pin pose (call this each step when robot is moving)."""
        self.ee_pos  = ee_pos.copy()
        self.ee_quat = ee_quat.copy()

    # ── Noise matrix builders ─────────────────────────────────────────────

    def _build_H(self) -> np.ndarray:
        """
        Sparse measurement matrix H  (n_y × n_x).
        Picks [px, py, pz] from each node's 13-element block.
        """
        H = np.zeros((self.n_y, self.n_x))
        for i in range(self.N):
            for d in range(3):
                H[3*i + d, i * self._NODE + d] = 1.0
        return H

    def _build_Q(
        self,
        sigma_pos:  float,
        sigma_quat: float,
        sigma_vel:  float,
        sigma_omg:  float,
    ) -> np.ndarray:
        """Diagonal process noise covariance Q."""
        diag = np.zeros(self.n_x)
        for i in range(self.N):
            base = i * self._NODE
            diag[base + 0:base + 3]  = sigma_pos**2
            diag[base + 3:base + 7]  = sigma_quat**2
            diag[base + 7:base + 10] = sigma_vel**2
            diag[base + 10:base + 13]= sigma_omg**2
        return np.diag(diag)
