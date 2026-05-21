"""
make_dataset.py
===============
Generate N_DATASETS paired (XPBD + Simulink) datasets.

Procedure
---------
1.  Find the XPBD gravitational steady state by running XPBD for STEADY_TIME
    seconds with gravity only (done once — it is deterministic).
2.  Start MATLAB once and configure shared parameters.
3.  For each dataset:
      (a) Generate a semi-random piecewise-constant tip force signal.
      (b) Run XPBD from the pre-computed steady state for SIM_TIME with
          the force applied.
      (c) Run Simulink for STEADY_TIME + SIM_TIME.  The force is zero for
          the first STEADY_TIME so the Simulink model settles to its own
          equilibrium; the random signal is applied for the remaining
          SIM_TIME.  Only the SIM_TIME tail is saved.
      (d) Save XPBD results, Simulink results, and the force signal.

Simulink model note
-------------------
The Simulink model must read the tip force from the workspace variable
``tip_force_ts`` (a timeseries with data shape T×6, columns fx fy fz tx ty tz).
Replace the constant ``f_last_node`` input with a "From Workspace" block
reading ``tip_force_ts``.

Outputs
-------
  Dataset/dataset_000/xpbd.csv          — XPBD simulation  (SIM_TIME long)
  Dataset/dataset_000/simulink.csv      — Simulink simulation (SIM_TIME long)
  Dataset/dataset_000/force_signal.csv  — applied tip force  (SIM_TIME long)
  ...
"""

import os
import numpy as np
import pandas as pd
import matlab
import matlab.engine

from sim_setup import create_sim, gravity_forces, WIRE, MATERIAL, SOLVER

# ── Configuration ──────────────────────────────────────────────────────────────
N_DATASETS   = 10
SIM_TIME     = 10.0    # s — duration of the recorded (force) phase
STEADY_TIME  = 20.0    # s — settling phase (force = 0)
FORCE_MAX    = 0.1    # N — max tip force per axis
STEP_DUR_MIN = 1     # s — min duration of a piecewise-constant segment
STEP_DUR_MAX = 3     # s — max duration
RANDOM_SEED  = 42
SIMULINK_DT  = 1e-3    # s — Simulink fixed step
MODEL_DIR    = r"model\Simulink_model"
MODEL_NAME   = 'cable_model'

BASE_POS   = np.array([0.0, 0.0, 0.0])
BASE_EULER = np.array([0.0, np.pi / 2, 0.0])   # cable z-axis → world X

# Quaternion [x, y, z, w] equivalent of BASE_EULER (pure Y rotation by π/2)
BASE_QUAT = np.array([0.0, np.sin(BASE_EULER[1] / 2), 0.0, np.cos(BASE_EULER[1] / 2)])

# ── Derived physical constants ─────────────────────────────────────────────────
sim, params = create_sim()
N   = WIRE.n_nodes
L   = WIRE.segment_length
I_s = WIRE.I_bending
A   = WIRE.area
m_node = MATERIAL.rho * A * L
I_eff  = SOLVER.I_eff_scale * m_node * L**2
alpha_stretch = params.alpha_stretch
alpha_bend    = params.alpha_bend
joint_angles  = np.zeros((3, N - 1))   # straight cable

TOTAL_TIME = STEADY_TIME + SIM_TIME    # full Simulink run duration

os.makedirs('Dataset', exist_ok=True)
rng = np.random.default_rng(RANDOM_SEED)


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════

def set_vec(eng, name, arr):
    vals = '; '.join(f'{float(v):.15e}' for v in np.asarray(arr).flatten())
    eng.eval(f"{name} = [{vals}];", nargout=0)


def make_force_signal(sim_time, dt, rng):
    """Return (t_steps, f_dense): one constant force pulse, then zero.

    A random force is applied from t=0 for a random duration drawn from
    [STEP_DUR_MIN, STEP_DUR_MAX], then the force drops to zero for the
    remainder of the simulation.
    """
    t_steps = np.arange(0.0, sim_time + dt, dt)
    f_dense = np.zeros((len(t_steps), 3))

    duration = rng.uniform(STEP_DUR_MIN, STEP_DUR_MAX)
    force    = rng.uniform(-FORCE_MAX, FORCE_MAX, 3)

    pulse_end = np.searchsorted(t_steps, duration, side='right')
    f_dense[:pulse_end] = force

    return t_steps, f_dense


def run_xpbd(x_init, v_init, f_dense):
    """Run XPBD for TOTAL_TIME: settle for STEADY_TIME then apply force for SIM_TIME.
    Returns (pos, quat, wrench) for the SIM_TIME phase only."""
    f_base = gravity_forces(params)
    x_cur  = x_init.copy()
    v_cur  = v_init.copy()

    # Settling phase — gravity only, no recording
    n_steady = int(round(STEADY_TIME / SOLVER.dt))
    for _ in range(n_steady):
        x_cur, v_cur, _, _ = sim.estimate_wire_state(
            x_cur, v_cur,
            np.zeros(3), BASE_QUAT,
            np.zeros(6), f_base,
        )

    # Force phase — record every step
    t_signal = np.arange(0.0, SIM_TIME + SOLVER.dt, SOLVER.dt)
    pos_out, quat_out, wrench_out = [], [], []
    for k in range(len(t_signal)):
        f_ext = f_base.copy()
        f_ext[-6:-3] = f_dense[k]   # tip force only, no tip torque
        x_cur, v_cur, _, ee_wrench = sim.estimate_wire_state(
            x_cur, v_cur,
            np.zeros(3), BASE_QUAT,
            np.zeros(6), f_ext,
        )
        nodes = x_cur.reshape(N, 7)
        pos_out.append(nodes[:, :3].copy())
        quat_out.append(nodes[:, 3:7].copy())
        wrench_out.append(ee_wrench.copy())

    return np.array(pos_out), np.array(quat_out), np.array(wrench_out)


def save_xpbd_csv(path, t_steps, pos, quat, wrench):
    rows = []
    for k, t in enumerate(t_steps):
        row = {'time': t}
        for i in range(N):
            row[f'node_{i}_x']  = pos[k, i, 0]
            row[f'node_{i}_y']  = pos[k, i, 1]
            row[f'node_{i}_z']  = pos[k, i, 2]
            row[f'node_{i}_qx'] = quat[k, i, 0]
            row[f'node_{i}_qy'] = quat[k, i, 1]
            row[f'node_{i}_qz'] = quat[k, i, 2]
            row[f'node_{i}_qw'] = quat[k, i, 3]
        row['ee_fx'] = wrench[k, 0]
        row['ee_fy'] = wrench[k, 1]
        row['ee_fz'] = wrench[k, 2]
        row['ee_tx'] = wrench[k, 3]
        row['ee_ty'] = wrench[k, 4]
        row['ee_tz'] = wrench[k, 5]
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def save_simulink_csv(path, sl_time, sl_pos, ori_all, ft_tcp, torque_tcp, N_sl):
    rows = []
    for k in range(len(sl_time)):
        row = {'time': sl_time[k]}
        for i in range(N_sl):
            row[f'node_{i}_x']  = sl_pos[k, i, 0]
            row[f'node_{i}_y']  = sl_pos[k, i, 1]
            row[f'node_{i}_z']  = sl_pos[k, i, 2]
            row[f'node_{i}_qw'] = ori_all[k, i, 0]
            row[f'node_{i}_qx'] = ori_all[k, i, 1]
            row[f'node_{i}_qy'] = ori_all[k, i, 2]
            row[f'node_{i}_qz'] = ori_all[k, i, 3]
        row['ft_x']     = ft_tcp[k, 0]
        row['ft_y']     = ft_tcp[k, 1]
        row['ft_z']     = ft_tcp[k, 2]
        row['torque_x'] = torque_tcp[k, 0]
        row['torque_y'] = torque_tcp[k, 1]
        row['torque_z'] = torque_tcp[k, 2]
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


# ════════════════════════════════════════════════════════════════════════════
# 1.  Compute initial state (both models start here and settle independently)
# ════════════════════════════════════════════════════════════════════════════
x_init_raw = sim.joint_to_world_init(BASE_POS, BASE_EULER, joint_angles)
v_init_raw = np.zeros(6 * N)
print(f"Initial state computed. Each model settles for {STEADY_TIME} s before the force is applied.")


# ════════════════════════════════════════════════════════════════════════════
# 2.  Start MATLAB and configure shared parameters (done once for all datasets)
# ════════════════════════════════════════════════════════════════════════════
print("\n=== Starting MATLAB engine ===")
eng = matlab.engine.start_matlab()
eng.addpath(MODEL_DIR, nargout=0)
print("MATLAB ready.")

# params struct
eng.eval("params_sl = struct();", nargout=0)
eng.eval(f"params_sl.N          = {N};",                nargout=0)
eng.eval(f"params_sl.dof        = 7;",                   nargout=0)
eng.eval(f"params_sl.L          = {L:.15e};",            nargout=0)
eng.eval(f"params_sl.dt         = {params.dt:.15e};",    nargout=0)
eng.eval(f"params_sl.solverIter = {SOLVER.iterations};", nargout=0)
set_vec(eng, "tmp", alpha_stretch); eng.eval("params_sl.alpha_stretch = tmp;", nargout=0)
set_vec(eng, "tmp", alpha_bend);    eng.eval("params_sl.alpha_bend    = tmp;", nargout=0)
eng.eval(f"params_sl.m_num = ones({N}, 1) * {m_node:.15e};", nargout=0)
eng.eval(f"I_node_sl = eye(3) * {I_eff:.15e};",              nargout=0)
eng.eval(f"params_sl.I_num = repmat({{I_node_sl}}, {N}, 1);", nargout=0)
eng.eval("clear tmp I_node_sl", nargout=0)
eng.eval("params = params_sl; clear params_sl", nargout=0)

# joint_angles for Gimbal Joint blocks
set_vec(eng, "tmp", joint_angles.flatten(order='F'))
eng.eval(f"joint_angles = reshape(tmp, 3, {N-1});", nargout=0)
eng.eval("clear tmp", nargout=0)

# stiffness / damping
omega_seg = float(np.sqrt(MATERIAL.E * I_s / (m_node * L**3)))
beta_bend = 2.0 / omega_seg
stiffness = float(1.0 / alpha_bend[0])
damp      = float(beta_bend * stiffness)
eng.workspace['stiffness'] = stiffness
eng.workspace['damp']      = damp
eng.workspace['m_node']    = float(m_node)
eng.workspace['I_node']    = float(I_eff)

# Both models start from the same initial state and settle independently.
set_vec(eng, "x_init", x_init_raw)

eng.eval(f"sim_time = {TOTAL_TIME};",         nargout=0)
eng.eval("steps = 0 : params.dt : sim_time;", nargout=0)

# Load model once
eng.load_system(MODEL_NAME, nargout=0)
eng.set_param(MODEL_NAME, 'StopTime',  str(TOTAL_TIME),  nargout=0)
eng.set_param(MODEL_NAME, 'FixedStep', str(SIMULINK_DT), nargout=0)
print(f"Simulink model loaded (settling={STEADY_TIME} s + signal={SIM_TIME} s).")


# ════════════════════════════════════════════════════════════════════════════
# 3.  Dataset loop
# ════════════════════════════════════════════════════════════════════════════
print(f"\n=== Generating {N_DATASETS} datasets ===")

# Time steps for the recorded phase (reset to 0 in saved CSV)
t_signal = np.arange(0.0, SIM_TIME + SOLVER.dt, SOLVER.dt)

for i in range(N_DATASETS):
    print(f"\n--- Dataset {i+1:>3}/{N_DATASETS} ---")
    out_dir = f'Dataset/dataset_{i:03d}'
    os.makedirs(out_dir, exist_ok=True)

    # (a) Random piecewise-constant force signal (SIM_TIME duration)
    _, f_dense = make_force_signal(SIM_TIME, SOLVER.dt, rng)

    pd.DataFrame({
        'time': t_signal,
        'fx':   f_dense[:, 0],
        'fy':   f_dense[:, 1],
        'fz':   f_dense[:, 2],
    }).to_csv(f'{out_dir}/force_signal.csv', index=False)

    # (b) XPBD — settle for STEADY_TIME then apply signal for SIM_TIME
    print("  XPBD ...", end=' ', flush=True)
    pos, quat, wrench = run_xpbd(x_init_raw, v_init_raw, f_dense)
    save_xpbd_csv(f'{out_dir}/xpbd.csv', t_signal, pos, quat, wrench)
    print("done.")

    # (c) Simulink — build timeseries for the full TOTAL_TIME run:
    #     force = 0 during [0, STEADY_TIME), then the signal during [STEADY_TIME, TOTAL_TIME]
    t_full = np.arange(0.0, TOTAL_TIME + SOLVER.dt, SOLVER.dt)
    n_steady = int(round(STEADY_TIME / SOLVER.dt)) + 1   # steps in settling phase
    f_full = np.zeros((len(t_full), 3))
    sig_len = min(len(f_dense), len(t_full) - n_steady + 1)
    f_full[n_steady - 1 : n_steady - 1 + sig_len] = f_dense[:sig_len]

    # tip_force_ts: timeseries with data (T × 6) [fx fy fz 0 0 0]
    f_6dof = np.hstack([f_full, np.zeros((len(f_full), 3))])
    eng.workspace['fdata'] = matlab.double(f_6dof.tolist())
    eng.workspace['ftime'] = matlab.double(t_full.reshape(-1, 1).tolist())
    eng.eval("tip_force_ts = timeseries(fdata, ftime);", nargout=0)
    eng.eval("clear fdata ftime", nargout=0)

    set_vec(eng, "f_last_node", np.zeros(6))   # kept for backward compat; model uses tip_force_ts

    print("  Simulink ...", end=' ', flush=True)
    eng.eval("out = sim('cable_model');", nargout=0)
    print("done.")

    # (d) Extract and crop to the SIM_TIME tail only
    sl_time_full = np.array(eng.eval("out.tout")).flatten()
    crop = sl_time_full >= STEADY_TIME
    sl_time = sl_time_full[crop] - STEADY_TIME   # re-zero time axis

    pos_sim = np.array(eng.eval("squeeze(out.XYZ.Data)"))
    N_sl = int(eng.eval("size(out.XYZ.Data, 1)")) // 3
    T_sl = int(eng.eval("size(out.XYZ.Data, 3)"))
    if pos_sim.shape != (3 * N_sl, T_sl): pos_sim = pos_sim.T
    sl_pos = pos_sim.reshape(N_sl, 3, T_sl).transpose(2, 0, 1)[crop]

    ori_sim = np.array(eng.eval("squeeze(out.ORI.Data)"))
    if ori_sim.shape != (4 * N_sl, T_sl): ori_sim = ori_sim.T
    ori_all = ori_sim.reshape(N_sl, 4, T_sl).transpose(2, 0, 1)[crop]

    cw_sim = np.array(eng.eval("squeeze(out.TCP_CW.Data)"))
    if cw_sim.shape[0] != 6: cw_sim = cw_sim.T
    ft_tcp     = cw_sim[:3, :].T[crop]
    torque_tcp = cw_sim[3:, :].T[crop]

    save_simulink_csv(
        f'{out_dir}/simulink.csv',
        sl_time, sl_pos, ori_all, ft_tcp, torque_tcp, N_sl,
    )
    print(f"  Saved → {out_dir}/")


# ════════════════════════════════════════════════════════════════════════════
# 4.  Clean up
# ════════════════════════════════════════════════════════════════════════════
eng.close_system(MODEL_NAME, 0, nargout=0)
eng.quit()
print(f"\n=== Done. Generated {N_DATASETS} datasets in Dataset/. MATLAB closed. ===")
