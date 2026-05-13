"""
run_ekf.py
==========
Run the EKF on a recorded dataset and visualise the filtered wire shape
against raw TrackDLO measurements.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.interpolate import interp1d

from sim_setup import create_sim, gravity_forces, WIRE, SOLVER, GRIPPER
from quaternion_utils import rotvec_to_rotm, rotm_to_quat
from wire_ekf import WireEKF


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════
DATA_PATH = "data_Viktor\ctrl_mads_lauge_anders_20260507_114440"
T_START   = 2.3   # s — skip this many seconds at the start

# Jacobian / predict mode — trade speed for accuracy:
#
#   'identity' — F=I, P_pred = P + Q.  Zero extra sim calls.
#                Fast. Good starting point; tune Q to compensate.
#
#   'banded'   — Block-banded numerical Jacobian.
#                Cost: 2*(2*BANDWIDTH+1)*13 sim calls per step.
#                  BANDWIDTH=1 →  78 calls  (6.7× faster than full)
#                  BANDWIDTH=2 → 130 calls  (4.0× faster)
#
#   'full'     — Full dense Jacobian. 520 sim calls/step. Slowest.
#
#   'ukf'      — Unscented KF. 521 sim calls/step but no linearisation.
#                Better for highly nonlinear regimes.
#
PREDICT_MODE = 'full'   # ← change this to experiment
BANDWIDTH    = 1            # only used when PREDICT_MODE='banded'


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Load data
# ══════════════════════════════════════════════════════════════════════════════
trk = pd.read_csv(f'{DATA_PATH}/trackdlo_nodes.csv')
tcp = pd.read_csv(f'{DATA_PATH}/robot_b_tcp_pose.csv')
ft  = pd.read_csv(f'{DATA_PATH}/robot_b_force_torque.csv')

t0    = min(trk.time_s.iloc[0], tcp.time_s.iloc[0], ft.time_s.iloc[0])
trk_t = trk.time_s.values - t0
tcp_t = tcp.time_s.values - t0
ft_t  = ft.time_s.values  - t0

trk = trk[trk_t >= T_START].reset_index(drop=True)
tcp = tcp[tcp_t >= T_START].reset_index(drop=True)
ft  = ft[ft_t   >= T_START].reset_index(drop=True)

trk_t = trk_t[trk_t >= T_START] - T_START
tcp_t = tcp_t[tcp_t >= T_START] - T_START
ft_t  = ft_t[ ft_t  >= T_START] - T_START
T_end = min(trk_t[-1], tcp_t[-1], ft_t[-1])

tcp_pos    = tcp[['x', 'y', 'z']].values
tcp_rotvec = tcp[['rx', 'ry', 'rz']].values

N = WIRE.n_nodes
trk_nodes = np.zeros((len(trk_t), N, 3))
for i in range(N):
    trk_nodes[:, i, 0] = trk[f'node_{i}_x'].values
    trk_nodes[:, i, 1] = trk[f'node_{i}_y'].values
    trk_nodes[:, i, 2] = trk[f'node_{i}_z'].values


# ══════════════════════════════════════════════════════════════════════════════
# 2.  EE pose from TCP (with position + orientation calibration)
# ══════════════════════════════════════════════════════════════════════════════
def frame_from_d3(d3):
    d3  = d3 / np.linalg.norm(d3)
    ref = np.array([0., 1., 0.]) if abs(d3[1]) < 0.9 else np.array([1., 0., 0.])
    d1  = np.cross(ref, d3);  d1 /= np.linalg.norm(d1)
    return np.column_stack([d1, np.cross(d3, d1), d3])

R0            = rotvec_to_rotm(tcp_rotvec[0])
d3_0          = trk_nodes[0, 1] - trk_nodes[0, 0]
d3_0         /= np.linalg.norm(d3_0)
R_tcp_to_wire = R0.T @ frame_from_d3(d3_0)
p_tcp_to_wire = R0.T @ (trk_nodes[0, 0] - tcp_pos[0])

ee_pos_all  = np.zeros((len(tcp_t), 3))
ee_quat_all = np.zeros((len(tcp_t), 4))
for k in range(len(tcp_t)):
    R = rotvec_to_rotm(tcp_rotvec[k])
    R_wire = R @ R_tcp_to_wire
    ee_pos_all[k]  = tcp_pos[k] + R @ p_tcp_to_wire
    ee_quat_all[k] = rotm_to_quat(R_wire)
    if k > 0 and np.dot(ee_quat_all[k], ee_quat_all[k-1]) < 0:
        ee_quat_all[k] *= -1

_ipos  = [interp1d(tcp_t, ee_pos_all[:, d],  fill_value='extrapolate') for d in range(3)]
_iquat = [interp1d(tcp_t, ee_quat_all[:, d], fill_value='extrapolate') for d in range(4)]

def ee_at(t):
    t   = float(np.clip(t, tcp_t[0], tcp_t[-1]))
    pos = np.array([f(t) for f in _ipos])
    q   = np.array([f(t) for f in _iquat]);  q /= np.linalg.norm(q)
    return pos, q


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Build simulator and EKF
# ══════════════════════════════════════════════════════════════════════════════
sim, params = create_sim()
f_ext = gravity_forces(params)

ee_pos0, ee_quat0 = ee_at(0.0)

ekf = WireEKF(
    sim, params, ee_pos0, ee_quat0, f_ext,
    sigma_pos  = 1e-3,   # process noise — position  [m]
    sigma_quat = 1e-4,   # process noise — quaternion
    sigma_vel  = 1e-2,   # process noise — velocity  [m/s]
    sigma_omg  = 1e-2,   # process noise — ang vel   [rad/s]
    sigma_meas = 4e-3,   # TrackDLO noise            [m]
)

z = ekf.state_from_trackdlo(trk_nodes[0])
P = ekf.initial_covariance()


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Run EKF and store results
# ══════════════════════════════════════════════════════════════════════════════
steps       = np.arange(0.0, T_end + SOLVER.dt, SOLVER.dt)
ekf_pos     = []   # filtered node positions at each step
sim_pos     = []   # pure model prediction (no EKF correction)

# Separate state for the open-loop simulation
x_sim_ol, v_sim_ol = ekf.unpack(z.copy())

print(f"Running EKF + open-loop model over {len(steps)} steps …")
print("(Jacobian computation is slow — this will take a while)")

for k, t_sim in enumerate(steps):
    ee_pos, ee_quat = ee_at(t_sim)
    ekf.update_ee(ee_pos, ee_quat)

    # ── EKF predict + update ──────────────────────────────────────────────
    if PREDICT_MODE == 'ukf':
        z, P = ekf.predict_ukf(z, P)
    else:
        z, P = ekf.predict(z, P, mode=PREDICT_MODE, bandwidth=BANDWIDTH)

    trk_idx = np.argmin(np.abs(trk_t - t_sim))
    z, P    = ekf.update(z, P, trk_nodes[trk_idx])

    ekf_pos.append(ekf.node_positions(z).copy())

    # ── Open-loop model (same EE, no measurement correction) ─────────────
    x_sim_ol, v_sim_ol, _, _ = sim.estimate_wire_state(
        x_sim_ol, v_sim_ol, ee_pos, ee_quat, np.zeros(6), f_ext
    )
    sim_pos.append(x_sim_ol.reshape(params.N, 7)[:, :3].copy())

    if k % 50 == 0:
        print(f"  t = {t_sim:.2f} s / {T_end:.2f} s")

ekf_pos = np.array(ekf_pos)   # (n_steps, N, 3)
sim_pos = np.array(sim_pos)   # (n_steps, N, 3)
print("Done.")


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Live playback
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(10, 8))
ax  = fig.add_subplot(111, projection='3d')
plt.ion()
plt.show()

for k, t_sim in enumerate(steps):
    ax.cla()

    # Raw TrackDLO
    trk_idx = np.argmin(np.abs(trk_t - t_sim))
    tp = trk_nodes[trk_idx]
    ax.plot(tp[:, 0], tp[:, 1], tp[:, 2],
            'o-', color='#1565C0', lw=2, ms=4, alpha=0.7, label='TrackDLO raw')

    # Open-loop model prediction
    sp = sim_pos[k]
    ax.plot(sp[:, 0], sp[:, 1], sp[:, 2],
            '^:', color='#2E7D32', lw=1.5, ms=4, alpha=0.8, label='Model prediction')

    # EKF filtered
    ep = ekf_pos[k]
    ax.plot(ep[:, 0], ep[:, 1], ep[:, 2],
            's--', color='#C62828', lw=2, ms=5, label='EKF filtered')

    all_pts = np.vstack([tp, sp, ep])
    mid  = all_pts.mean(axis=0)
    half = np.ptp(all_pts, axis=0).max() / 2 + 0.03
    ax.set_xlim(mid[0]-half, mid[0]+half)
    ax.set_ylim(mid[1]-half, mid[1]+half)
    ax.set_zlim(mid[2]-half, mid[2]+half)
    ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]'); ax.set_zlabel('Z [m]')
    ax.set_title(f'Wire EKF  —  t = {t_sim:.2f} s', fontsize=11)
    ax.legend(fontsize=9)
    try: ax.set_box_aspect([1, 1, 1])
    except AttributeError: pass

    plt.pause(0.01)

plt.ioff()
plt.show()
