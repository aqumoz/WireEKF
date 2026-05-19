import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.interpolate import interp1d

from sim_setup import create_sim, gravity_forces, WIRE, SOLVER, GRIPPER
from visualization import plot_rod_frames
from quaternion_utils import rotvec_to_rotm, rotm_to_quat, state_from_positions
from learning.models.multitaskGP import load_model, predict
import torch

model, likelihood = load_model("learning/models/gp_model.pth")

# ══════════════════════════════════════════════════════════════════════════════
# 1.  Build simulator
# ══════════════════════════════════════════════════════════════════════════════
sim, params = create_sim()


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Load data
# ══════════════════════════════════════════════════════════════════════════════
DATA_PATH = "data_Viktor/ctrl_mads_lauge_anders_20260507_112904"

# Skip the first T_START seconds of the recording (e.g. robot is stationary)
T_START = 0   # s  — set to 0.0 to use the full dataset

trk = pd.read_csv(f'{DATA_PATH}/trackdlo_nodes.csv')
tcp = pd.read_csv(f'{DATA_PATH}/robot_b_tcp_pose.csv')
ft  = pd.read_csv(f'{DATA_PATH}/robot_b_force_torque.csv')

t0    = min(trk.time_s.iloc[0], tcp.time_s.iloc[0], ft.time_s.iloc[0])
trk_t = trk.time_s.values - t0
tcp_t = tcp.time_s.values - t0
ft_t  = ft.time_s.values  - t0

# Trim everything before T_START
trk = trk[trk_t >= T_START].reset_index(drop=True)
tcp = tcp[tcp_t >= T_START].reset_index(drop=True)
ft  = ft[ft_t   >= T_START].reset_index(drop=True)

trk_t = trk_t[trk_t >= T_START] - T_START
tcp_t = tcp_t[tcp_t >= T_START] - T_START
ft_t  = ft_t[ ft_t  >= T_START] - T_START
T_end = min(trk_t[-1], tcp_t[-1], ft_t[-1])

tcp_pos    = tcp[['x', 'y', 'z']].values       # (n_tcp, 3)
tcp_rotvec = tcp[['rx', 'ry', 'rz']].values    # (n_tcp, 3)  — Rodrigues

N_TRK = WIRE.n_nodes
trk_nodes = np.zeros((len(trk_t), N_TRK, 3))
for i in range(N_TRK):
    trk_nodes[:, i, 0] = trk[f'node_{i}_x'].values
    trk_nodes[:, i, 1] = trk[f'node_{i}_y'].values
    trk_nodes[:, i, 2] = trk[f'node_{i}_z'].values


# ══════════════════════════════════════════════════════════════════════════════
# 3.  One-time calibration: find the fixed rotation from TCP frame to wire frame
#
#     At t=0 the wire tangent (from TrackDLO) and the TCP orientation (from
#     the robot) are both known in the world frame.  Their difference is a
#     fixed rotation R_tcp_to_wire that stays constant for a given tool setup.
# ══════════════════════════════════════════════════════════════════════════════
def frame_from_d3(d3: np.ndarray) -> np.ndarray:
    """Build a rotation matrix [d1 | d2 | d3] with d3 as the local z-axis."""
    d3  = d3 / np.linalg.norm(d3)
    ref = np.array([0., 1., 0.]) if abs(d3[1]) < 0.9 else np.array([1., 0., 0.])
    d1  = np.cross(ref, d3);  d1 /= np.linalg.norm(d1)
    d2  = np.cross(d3, d1)
    return np.column_stack([d1, d2, d3])


# TCP frame at t=0
R_base_tcp_0 = rotvec_to_rotm(tcp_rotvec[0])

# Wire tangent direction at t=0 (first TrackDLO segment)
d3_0 = trk_nodes[0, 1] - trk_nodes[0, 0]
d3_0 /= np.linalg.norm(d3_0)

# Wire frame at t=0 — z-axis aligned with wire tangent
R_base_wire_0 = frame_from_d3(d3_0)

# Fixed calibration rotation: TCP frame → wire frame
R_tcp_to_wire = R_base_tcp_0.T @ R_base_wire_0

# Fixed position offset: vector from TCP origin to wire attachment,
# expressed in the TCP frame — calibrated from TrackDLO node 0 at t=0
true_attach_0      = trk_nodes[0, 0]                          # world frame
computed_attach_0  = tcp_pos[0] + R_base_tcp_0 @ np.array([0.0, 0.0, GRIPPER.attach_offset])
p_tcp_to_wire      = R_base_tcp_0.T @ (true_attach_0 - tcp_pos[0])  # TCP frame

# Diagnostics
tcp_z      = R_base_tcp_0 @ np.array([0., 0., 1.])
angle_err  = np.degrees(np.arccos(np.clip(np.dot(tcp_z, d3_0), -1, 1)))
pos_err    = np.linalg.norm(computed_attach_0 - true_attach_0) * 1e3
print(f"TCP z-axis (world):       {tcp_z.round(3)}")
print(f"Wire tangent t=0:         {d3_0.round(3)}")
print(f"Orientation calib angle:  {angle_err:.1f}°")
print(f"Position before calib:    {computed_attach_0.round(4)}  (err {pos_err:.1f} mm)")
print(f"Position after calib:     {true_attach_0.round(4)}  (TrackDLO node 0)")
print(f"Offset in TCP frame:      {p_tcp_to_wire.round(4)}")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Pre-compute EE pose at every TCP timestamp using calibrated orientation
# ══════════════════════════════════════════════════════════════════════════════
offset_wire = np.array([0.0, 0.0, GRIPPER.attach_offset])   # in wire/flange frame

ee_pos_tcp  = np.zeros((len(tcp_t), 3))
ee_quat_tcp = np.zeros((len(tcp_t), 4))

ft_pos_tcp = ft[['fx', 'fy', 'fz', 'tx', 'ty', 'tz']].values   # (n_ft, 6)

for k in range(len(tcp_t)):
    R_tcp  = rotvec_to_rotm(tcp_rotvec[k])
    R_wire = R_tcp @ R_tcp_to_wire               # apply orientation calibration
    ee_pos_tcp[k]  = tcp_pos[k] + R_tcp @ p_tcp_to_wire   # calibrated position
    ee_quat_tcp[k] = rotm_to_quat(R_wire)

# Ensure consistent quaternion sign across frames (prevent interpolation flips)
for k in range(1, len(tcp_t)):
    if np.dot(ee_quat_tcp[k], ee_quat_tcp[k-1]) < 0:
        ee_quat_tcp[k] *= -1

# Interpolators: EE pose → any simulation timestamp
_interp_pos  = [interp1d(tcp_t, ee_pos_tcp[:, d],  fill_value='extrapolate') # type: ignore
                for d in range(3)]
_interp_quat = [interp1d(tcp_t, ee_quat_tcp[:, d], fill_value='extrapolate') # type: ignore
                for d in range(4)]

_interp_ft  = [interp1d(ft_t, ft_pos_tcp[:, d],  fill_value='extrapolate') # type: ignore
                for d in range(6)]

# Raw TCP pose interpolators (original position + rotation vector, no calibration)
_interp_tcp_pos    = [interp1d(tcp_t, tcp_pos[:, d],    fill_value='extrapolate') # type: ignore
                      for d in range(3)]
_interp_tcp_rotvec = [interp1d(tcp_t, tcp_rotvec[:, d], fill_value='extrapolate') # type: ignore
                      for d in range(3)]

def ee_at(t):
    """Return (ee_pos, ee_quat) interpolated to simulation time t."""
    t   = float(np.clip(t, tcp_t[0], tcp_t[-1]))
    pos = np.array([f(t) for f in _interp_pos])
    q   = np.array([f(t) for f in _interp_quat])
    q  /= np.linalg.norm(q)           # re-normalise after linear interp
    return pos, q


def ft_at(t):
    """Return ft interpolated to simulation time t."""
    t   = float(np.clip(t, ft_t[0], ft_t[-1]))
    ft = np.array([f(t) for f in _interp_ft])
    return ft


def tcp_pose_at(t):
    """
    Return the raw TCP pose as a 6D vector [x, y, z, rx, ry, rz] at time t,
    as recorded by the robot — no calibration applied.
    """
    t = float(np.clip(t, tcp_t[0], tcp_t[-1]))
    return np.array([f(t) for f in _interp_tcp_pos + _interp_tcp_rotvec])


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Initial conditions from first TrackDLO frame
# ══════════════════════════════════════════════════════════════════════════════
x_init = state_from_positions(trk_nodes[0])
v_init = np.zeros(6 * N_TRK)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  External forces
# ══════════════════════════════════════════════════════════════════════════════
f_ext = gravity_forces(params)

tip_force = np.zeros(6)
f_ext[-6:] += tip_force

# ══════════════════════════════════════════════════════════════════════════════
# 7.  Simulation loop
# ══════════════════════════════════════════════════════════════════════════════
steps = np.arange(0.0, T_end + SOLVER.dt, SOLVER.dt)

x_history = []
plot = True
if plot == True:
    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection='3d')
    plt.ion()
    plt.show()

for time in steps:
    ee_pos, ee_quat = ee_at(time)
    ft_cur = ft_at(time)
    tcp_cur = tcp_pose_at(time)


    
    x_init, v_init, _cw, ee_wrench = sim.estimate_wire_state(
        x_init, v_init, ee_pos, ee_quat, np.zeros(6), f_ext,
    )

    x = torch.from_numpy(np.concatenate([ft_cur, tcp_cur])).float().unsqueeze(0)
    
    predictions = predict(model, likelihood, x)

    if time == 0.0:
        for i in range(20):
            print(predictions.mean.numpy()[0][i*3:(i+1)*3])
            x_init[i*7:i*7+3] += predictions.mean.numpy()[0][i*3:(i+1)*3]

    x_history.append(np.append(time, x_init)) 

    if plot == True:
        plot_rod_frames(ax, x_init, params.dof, params.N, time, frame_scale=0.015) # type: ignore
    
        # Overlay nearest TrackDLO frame (no interpolation — will look laggy)
        trk_idx = np.argmin(np.abs(trk_t - time))
        tp = trk_nodes[trk_idx]
        ax.plot(tp[:, 0], tp[:, 1], tp[:, 2], # type: ignore
                'o-', color='#1565C0', lw=2, ms=4, alpha=0.8, label='TrackDLO')
    
        plt.pause(0.001)

    
print("Sim done")

# ============================================================
# SAVE TO CSV
# ============================================================

column_names = ["time_s"]

num_nodes = 20

for node in range(num_nodes):

    # Position names
    column_names.append(f"node_model_{node}_x")
    column_names.append(f"node_model_{node}_y")
    column_names.append(f"node_model_{node}_z")

    # Quaternion names
    column_names.append(f"node_model_{node}_qx")
    column_names.append(f"node_model_{node}_qy")
    column_names.append(f"node_model_{node}_qz")
    column_names.append(f"node_model_{node}_qw")


df = pd.DataFrame(x_history, columns=column_names)

df.to_csv("x_init_history.csv", index=False)


# fig = plt.figure(figsize=(9, 7))
# ax  = fig.add_subplot(111, projection='3d')

# plot_rod_frames(ax, x_init, params.dof, params.N, steps[-1], frame_scale=0.015)

# # Nearest TrackDLO frame to end of simulation
# trk_idx = np.argmin(np.abs(trk_t - steps[-1]))
# tp = trk_nodes[trk_idx]
# ax.plot(tp[:, 0], tp[:, 1], tp[:, 2],
#         'o-', color="#247ADD", lw=2, ms=4, alpha=0.8, label='TrackDLO')
# ax.legend()
# plt.show()