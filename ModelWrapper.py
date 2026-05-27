import sys
from pathlib import Path
from ColumnMap import ColumnMap, Data

model_path = Path(__file__).resolve().parent / "model"
if str(model_path) not in sys.path:
    sys.path.insert(0, str(model_path))

import numpy as np

from model.sim_setup import create_sim, gravity_forces, WIRE, SOLVER

learning_path = Path(__file__).resolve().parent  # adjust if learning/ is elsewhere
if str(learning_path) not in sys.path:
    sys.path.insert(0, str(learning_path))


from learning.models.multitaskGP import load_model, CorrectedDLOModel  # ← adjust module path

GP_MODEL_PATH = Path(__file__).resolve().parent / "learning/models/latent5_lr0,005_inducing64/gp_model_combined.pth"

class Model():
    def __init__(self, use_correction: bool = True):
        self.sim, self.params = create_sim()

        # # end effector position and orientation which should probably come from data as well
        # self.ee_pos  = np.array([0.0, 0.0, 0.0])
        # self.ee_quat = np.array([0.0, np.sin(np.pi / 4), 0.0, np.cos(np.pi / 4)])

        # Det her er bare at sætte gravity på all noder
        self.f_ext = gravity_forces(self.params)
        self.use_correction = use_correction
        if use_correction:
            gp_model, likelihood = load_model(str(GP_MODEL_PATH))
            self.corrected_model = CorrectedDLOModel(gp_model, likelihood)
        else:
            self.corrected_model = None

    def pose_to_x_state_vec(self, pose : Data) -> np.ndarray:
        """Convert a Data object into the sim's x state vector.
        
        The sim expects 7 values per node: [px, py, pz, qw, qx, qy, qz].
        A zero quaternion is invalid and causes NaN — we default to identity (1,0,0,0)
        if no orientation is available for a node.
        """
        state_vec = np.zeros((7 * self.params.N,))
        for i in range(1, self.params.N + 1):
            idx = (i - 1) * 7
            state_vec[idx : idx + 3] = pose.p[i]
            q = pose.q.get(i, (1.0, 0.0, 0.0, 0.0))  # identity quaternion as fallback
            state_vec[idx + 3 : idx + 7] = q
        return state_vec
    
    def pose_to_v_state_vec(self, pose : Data) -> np.ndarray:
        """Convert a Data object into the sim's v state vector.
        
        The sim expects 6 values per node: [vx, vy, vz, wx, wy, wz].
        Angular velocity defaults to zero if not available.
        """
        vel_vec = np.zeros((6 * self.params.N,))
        for i in range(1, self.params.N + 1):
            idx = (i - 1) * 6
            vel_vec[idx : idx + 3] = pose.v.get(i, (0.0, 0.0, 0.0))
            # Angular velocity (wx, wy, wz) — not tracked by EKF, leave as zero
        return vel_vec
    
    def x_v_state_vec_to_pose(self, x_state : np.ndarray, v_state : np.ndarray, source_pose: Data) -> Data:
        """Convert state vectors back into a Data object.
        
        estimate_wire_state returns x as (N*7,): [px, py, pz, qw, qx, qy, qz] per node.
        estimate_wire_state returns v as (N*6,): [vx, vy, vz, wx, wy, wz] per node.
        We extract only the position (first 3) and quaternion (last 4) from x,
        and only linear velocity (first 3) from v.
        """
        # Detect whether the sim returned 7-DOF layout (N*7) or plain 3-DOF layout (N*3)
        x_dof = len(x_state) // self.params.N  # 7 if pos+quat, 3 if pos-only
        v_dof = len(v_state) // self.params.N  # 6 if lin+ang, 3 if lin-only

        pose = Data()
        pose.p = {}
        pose.q = {}
        pose.v = {}
        # Carry over fields that the sim doesn't touch
        pose.ft = source_pose.ft
        pose.torque = source_pose.torque
        pose.time = source_pose.time

        for i in range(1, self.params.N + 1):
            x_idx = (i - 1) * x_dof
            v_idx = (i - 1) * v_dof
            pose.p[i] = tuple(x_state[x_idx : x_idx + 3])
            if x_dof >= 7:
                pose.q[i] = tuple(x_state[x_idx + 3 : x_idx + 7])
            else:
                # No quaternion in output — preserve from source
                pose.q[i] = source_pose.q.get(i, (1.0, 0.0, 0.0, 0.0))
            pose.v[i] = tuple(v_state[v_idx : v_idx + 3])
        return pose
    
    def predict(self, pose : Data, dt : float) -> Data:
        self.x_init = self.pose_to_x_state_vec(pose)
        self.v_init = self.pose_to_v_state_vec(pose)

        # Definer sim_time til den tid du gerne ville simulere
        sim_time = dt
        steps    = np.arange(0.0, sim_time + SOLVER.dt, SOLVER.dt)

        self.ee_pos  = np.array(pose.p[1])              # end-effector position from node 1
        self.ee_quat = np.array(pose.q.get(1, (1.0, 0.0, 0.0, 0.0)))  # end-effector quaternion, safe fallback

        for time in steps:
            # Her bliver x_init/v_init bliver overskrevet her
            self.x_init, self.v_init, _cw, ee_wrench = self.sim.estimate_wire_state(
                self.x_init, self.v_init, self.ee_pos, self.ee_quat, np.zeros(6), self.f_ext,
            )
        
        predicted_pose = self.x_v_state_vec_to_pose(self.x_init, self.v_init, pose)

        # ── Apply GP correction ────────────────────────────────────────────
        if self.use_correction and self.corrected_model is not None:
            # (N, 3) array of predicted node positions
            sp = np.array([predicted_pose.p[i] for i in range(1, self.params.N + 1)])

            ft= np.array(pose.ft) if pose.ft is not None else np.array(ee_wrench[:3])
            torque = np.array(pose.torque) if pose.torque is not None else np.array(ee_wrench[3:])
            t_snap = float(pose.time) # scalar — matches CorrectedDLOModel.predict signature

            corrected_positions = self.corrected_model.predict(sp, ft, torque, t_snap)  # (N, 3)

            for i in range(1, self.params.N + 1):
                predicted_pose.p[i] = tuple(corrected_positions[i - 1])

        return predicted_pose