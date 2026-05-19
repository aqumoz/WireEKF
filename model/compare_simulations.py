"""
compare_simulations.py
======================
Run the XPBD Python simulator and the Simulink cable model with identical
initial conditions, then plot both wire shapes side-by-side and save a CSV
for each.

Outputs
-------
  results/xpbd_simulation.csv     — Python XPBD simulation
  results/simulink_simulation.csv — Simulink simulation
  (+ a live 3D comparison plot)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
import matlab.engine

from sim_setup import create_sim, gravity_forces, WIRE, MATERIAL, SOLVER

import sys
sys.path.append("../")  


from WireEKF.learning.models.multitaskGP import load_model, CorrectedDLOModel


def main():
    # ── Paths ─────────────────────────────────────────────────────────────────────
    MODEL_DIR = r"model\Simulink_model"   # folder with cable_model.slx

    rng = np.random.default_rng()


    # ── Shared initial conditions (both models use these) ────────────────────────
    SIM_TIME     = 15.0
    BASE_POS     = np.array([0.0, 0.0, 0.0])
    BASE_EULER   = np.array([0.0, np.pi / 2, 0.0])   # cable z-axis → world X
    JOINT_BEND_X = np.deg2rad(rng.uniform(-0.5,0.5))
    JOINT_BEND_Y = np.deg2rad(rng.uniform(-2.5,2.5))
    JOINT_BEND_Z = np.deg2rad(rng.uniform(-0.25,0.25))
    F_LAST_NODE  = np.concatenate([rng.uniform(-0.01, 0.01, size=(3,)), np.zeros(3)])  # tip load [N]

    # Simulink fixed step (keep at 1e-3 for Simscape Multibody stability)
    SIMULINK_DT = 1e-3

    folder_name = (
        f"bx{np.rad2deg(JOINT_BEND_X):.2f}_"
        f"by{np.rad2deg(JOINT_BEND_Y):.2f}_"
        f"bz{np.rad2deg(JOINT_BEND_Z):.2f}_"
        f"fx{F_LAST_NODE[0]:.3f}_"
        f"fy{F_LAST_NODE[1]:.3f}_"
        f"fz{F_LAST_NODE[2]:.3f}"
    )
    print(F_LAST_NODE)


    save_path = os.path.join("results_test", folder_name)
    os.makedirs(save_path, exist_ok=True)

    # ══════════════════════════════════════════════════════════════════════════════
    # 1.  Shared parameters
    # ══════════════════════════════════════════════════════════════════════════════
    sim, params = create_sim()

    N   = WIRE.n_nodes
    L   = WIRE.segment_length
    A   = WIRE.area
    I_s = WIRE.I_bending
    J_t = WIRE.J_torsion

    m_node = MATERIAL.rho * A * L
    I_eff  = SOLVER.I_eff_scale * m_node * L**2

    alpha_stretch = params.alpha_stretch
    alpha_bend    = params.alpha_bend

    joint_angles = np.zeros((3, N - 1))
    joint_angles[0, :] = JOINT_BEND_X
    joint_angles[1, :] = JOINT_BEND_Y
    joint_angles[2, :] = JOINT_BEND_Z

    print("=== Shared parameters ===")
    print(f"  N={N}, L={L*1e3:.1f} mm, E={MATERIAL.E:.3e} Pa")
    print(f"  m_node={m_node*1e3:.3f} g, SIM_TIME={SIM_TIME} s")


    # ══════════════════════════════════════════════════════════════════════════════
    # 2.  Run Python XPBD simulation
    # ══════════════════════════════════════════════════════════════════════════════
    print("\n--- Running Python XPBD simulation ---")

    x_cur = sim.joint_to_world_init(BASE_POS, BASE_EULER, joint_angles)
    v_cur = np.zeros(6 * N)

    ee_pos  = np.array([0.0, 0.0, 0.0])
    ee_quat = np.array([0.0, np.sin(np.pi / 4), 0.0, np.cos(np.pi / 4)])

    f_ext = gravity_forces(params)
    f_ext[-6:] += F_LAST_NODE   # tip load

    steps    = np.arange(0.0, SIM_TIME + SOLVER.dt, SOLVER.dt)
    xpbd_pos    = []
    xpbd_quat   = []
    xpbd_wrench = []

    for k, t in enumerate(steps):
        x_cur, v_cur, _, ee_wrench = sim.estimate_wire_state(
            x_cur, v_cur, ee_pos, ee_quat, np.zeros(6), f_ext
        )
        nodes = x_cur.reshape(N, 7)
        xpbd_pos.append(nodes[:, :3].copy())
        xpbd_quat.append(nodes[:, 3:7].copy())
        xpbd_wrench.append(ee_wrench.copy())
        if k % 200 == 0:
            print(f"  t = {t:.2f} s / {SIM_TIME:.1f} s")

    xpbd_pos    = np.array(xpbd_pos)     # (n_steps, N, 3)
    xpbd_quat   = np.array(xpbd_quat)    # (n_steps, N, 4)  [qx, qy, qz, qw]
    xpbd_wrench = np.array(xpbd_wrench)  # (n_steps, 6)
    xpbd_times  = steps
    print(f"  Done. {len(steps)} steps.")

    # ── Save XPBD CSV ─────────────────────────────────────────────────────────────
    rows = []
    for k, t in enumerate(xpbd_times):
        row = {'time': t}
        for i in range(N):
            row[f'node_{i}_x']  = xpbd_pos[k, i, 0]
            row[f'node_{i}_y']  = xpbd_pos[k, i, 1]
            row[f'node_{i}_z']  = xpbd_pos[k, i, 2]
            row[f'node_{i}_qx'] = xpbd_quat[k, i, 0]
            row[f'node_{i}_qy'] = xpbd_quat[k, i, 1]
            row[f'node_{i}_qz'] = xpbd_quat[k, i, 2]
            row[f'node_{i}_qw'] = xpbd_quat[k, i, 3]
        row['ee_fx'] = xpbd_wrench[k, 0]
        row['ee_fy'] = xpbd_wrench[k, 1]
        row['ee_fz'] = xpbd_wrench[k, 2]
        row['ee_tx'] = xpbd_wrench[k, 3]
        row['ee_ty'] = xpbd_wrench[k, 4]
        row['ee_tz'] = xpbd_wrench[k, 5]
        rows.append(row)


    pd.DataFrame(rows).to_csv(os.path.join(save_path, 'xpbd_simulation.csv'), index=False)
    #pd.DataFrame(rows).to_csv('results/xpbd_simulation.csv', index=False)
    print("  Saved " + os.path.join(save_path, 'xpbd_simulation.csv'))


    # ══════════════════════════════════════════════════════════════════════════════
    # 3.  Run Simulink simulation
    # ══════════════════════════════════════════════════════════════════════════════
    print("\n--- Running Simulink simulation ---")

    print("Starting MATLAB engine ...")
    eng = matlab.engine.start_matlab()
    eng.addpath(MODEL_DIR, nargout=0)
    print("MATLAB ready.")

    def set_vec(name, arr):
        vals = '; '.join(f'{float(v):.15e}' for v in np.asarray(arr).flatten())
        eng.eval(f"{name} = [{vals}];", nargout=0)

    # Build params struct
    eng.eval("params = struct();",                                     nargout=0)
    eng.eval(f"params.N          = {N};",                              nargout=0)
    eng.eval(f"params.dof        = 7;",                                nargout=0)
    eng.eval(f"params.L          = {L:.15e};",                         nargout=0)
    eng.eval(f"params.dt         = {SOLVER.dt:.15e};",                 nargout=0)
    eng.eval(f"params.solverIter = {SOLVER.iterations};",              nargout=0)
    set_vec("tmp", alpha_stretch);  eng.eval("params.alpha_stretch = tmp;", nargout=0)
    set_vec("tmp", alpha_bend);     eng.eval("params.alpha_bend    = tmp;", nargout=0)
    eng.eval(f"params.m_num = ones({N}, 1) * {m_node:.15e};",         nargout=0)
    eng.eval(f"I_node = eye(3) * {I_eff:.15e};",                      nargout=0)
    eng.eval(f"params.I_num = repmat({{I_node}}, {N}, 1);",            nargout=0)
    eng.eval("clear tmp I_node",                                       nargout=0)

    # x_init from Python
    x_init_py = sim.joint_to_world_init(BASE_POS, BASE_EULER, joint_angles)
    set_vec("x_init", x_init_py)

    # joint_angles matrix (Gimbal Joint blocks reference it directly)
    ja_flat = joint_angles.flatten(order='F')
    set_vec("tmp", ja_flat)
    eng.eval(f"joint_angles = reshape(tmp, 3, {N-1});", nargout=0)
    eng.eval("clear tmp", nargout=0)

    # Scalar workspace variables
    stiffness = float(1.0 / alpha_bend[0])
    eng.workspace['stiffness'] = stiffness
    eng.workspace['damp']      = 0.5
    eng.workspace['m_node']    = float(m_node)
    eng.workspace['I_node']    = float(I_eff)

    set_vec("f_last_node", F_LAST_NODE)
    eng.eval(f"sim_time = {SIM_TIME};",            nargout=0)
    eng.eval("steps = 0 : params.dt : sim_time;",  nargout=0)

    # Run
    MODEL_NAME = 'cable_model'
    eng.load_system(MODEL_NAME, nargout=0)
    eng.set_param(MODEL_NAME, 'StopTime',   str(SIM_TIME),   nargout=0)
    eng.set_param(MODEL_NAME, 'FixedStep',  str(SIMULINK_DT), nargout=0)
    print(f"Running simulation (T={SIM_TIME} s, fixed step={SIMULINK_DT} s) ...")
    eng.eval("out = sim('cable_model');", nargout=0)
    print("  Done.")

    # Pull results
    sl_time = np.array(eng.eval("out.tout")).flatten()
    pos_sim = np.array(eng.eval("squeeze(out.XYZ.Data)"))

    N_sl = int(eng.eval("size(out.XYZ.Data, 1)")) // 3
    T_sl = int(eng.eval("size(out.XYZ.Data, 3)"))

    if pos_sim.shape != (3 * N_sl, T_sl):
        pos_sim = pos_sim.T

    # Data is interleaved [x1,y1,z1, x2,y2,z2, ...] along rows (3*N, T).
    # NumPy C-order reshape(N, 3, T) unpacks this correctly per node,
    # then transpose(2, 0, 1) gives (T, N, 3).
    sl_pos = pos_sim.reshape(N_sl, 3, T_sl).transpose(2, 0, 1)   # (T, N, 3)

    if N_sl != N:
        print(f"  Note: Simulink has {N_sl} nodes, XPBD has {N} nodes.")

    # Orientation — same interleaved layout (4*N, T) → (T, N, 4)
    ori_sim = np.array(eng.eval("squeeze(out.ORI.Data)"))
    if ori_sim.shape != (4 * N_sl, T_sl):
        ori_sim = ori_sim.T
    ori_all = ori_sim.reshape(N_sl, 4, T_sl).transpose(2, 0, 1)  # (T, N, 4)

    cw_sim = np.array(eng.eval("squeeze(out.TCP_CW.Data)"))
    if cw_sim.shape[0] != 6:
        cw_sim = cw_sim.T

    ft_tcp     = cw_sim[:3, :].T   # (T, 3)
    torque_tcp = cw_sim[3:, :].T   # (T, 3)

    eng.close_system(MODEL_NAME, 0, nargout=0)
    eng.quit()
    print("  MATLAB engine closed.")

    # ── Save Simulink CSV ─────────────────────────────────────────────────────────
    rows = []
    for k in range(T_sl):
        row = {'time': sl_time[k]}
        for i in range(N_sl):
            row[f'node_{i}_x']  = sl_pos[k, i, 0]
            row[f'node_{i}_y']  = sl_pos[k, i, 1]
            row[f'node_{i}_z']  = sl_pos[k, i, 2]
            row[f'node_{i}_qw'] = ori_all[k, i, 0]
            row[f'node_{i}_qx'] = ori_all[k, i, 1]
            row[f'node_{i}_qy'] = ori_all[k, i, 2]
            row[f'node_{i}_qz'] = ori_all[k, i, 3]
        #row['ft_x']     = ft_tcp[k, 0]
        #row['ft_y']     = ft_tcp[k, 1]
        #row['ft_z']     = ft_tcp[k, 2]
        #row['torque_x'] = torque_tcp[k, 0]
        #row['torque_y'] = torque_tcp[k, 1]
        #row['torque_z'] = torque_tcp[k, 2]
        rows.append(row)


    pd.DataFrame(rows).to_csv(os.path.join(save_path, 'simulink_simulation.csv'), index=False)
    #pd.DataFrame(rows).to_csv('results/simulink_simulation.csv', index=False)
    print("  Saved " + os.path.join(save_path, 'simulink_simulation.csv'))





    model, likelihood = load_model("learning/models/gp_model.pth")
    corrected_model = CorrectedDLOModel(model, likelihood)




    # ══════════════════════════════════════════════════════════════════════════════
    # 4.  Plot — 4 time snapshots comparing both wires
    # ══════════════════════════════════════════════════════════════════════════════
    C_XPBD = '#C62828'   # red  — Python XPBD
    C_SL   = '#1565C0'   # blue — Simulink
    C_CORRECTED = '#2E7D32' # green — Simulink + GP correction

    snapshot_times = [0.0, SIM_TIME * 0.33, SIM_TIME * 0.66, SIM_TIME]

    fig = plt.figure(figsize=(16, 5))
    for si, t_snap in enumerate(snapshot_times):
        ax = fig.add_subplot(1, 4, si + 1, projection='3d')

        # XPBD — nearest timestep
        xi = np.argmin(np.abs(xpbd_times - t_snap))
        xp = xpbd_pos[xi]   # (N, 3)

        # Simulink — nearest timestep
        si2 = np.argmin(np.abs(sl_time - t_snap))
        sp  = sl_pos[si2]   # (N_sl, 3)

        # GP-corrected Simulink — predict residuals and add to Simulink positions
        cp = corrected_model.predict(sp, ft_tcp[si2], torque_tcp[si2])

        ax.plot(xp[:, 0], xp[:, 1], xp[:, 2],
                'o-', color=C_XPBD, lw=2, ms=4,
                label='XPBD' if si == 0 else '')
        ax.plot(sp[:, 0], sp[:, 1], sp[:, 2],
                's--', color=C_SL, lw=2, ms=5,
                label='Simulink' if si == 0 else '')
        ax.plot(cp[:, 0], cp[:, 1], cp[:, 2],
                '^:', color=C_CORRECTED, lw=2, ms=5,
                label='Simulink + GP' if si == 0 else '')

        ax.set_title(f't = {t_snap:.1f} s', fontsize=10)
        ax.set_xlabel('X [m]', labelpad=1)
        ax.set_ylabel('Y [m]', labelpad=1)
        ax.set_zlabel('Z [m]', labelpad=1)
        ax.tick_params(labelsize=7)
        ax.view_init(elev=20, azim=-60)

        all_pts = np.vstack([xp, sp])
        mid  = all_pts.mean(axis=0)
        half = np.ptp(all_pts, axis=0).max() / 2 + 0.02
        ax.set_xlim(mid[0]-half, mid[0]+half)
        ax.set_ylim(mid[1]-half, mid[1]+half)
        ax.set_zlim(mid[2]-half, mid[2]+half)
        try: ax.set_box_aspect([1, 1, 1])
        except AttributeError: pass

    handles = [
        plt.Line2D([0],[0], color=C_XPBD, marker='o', lw=2, label='XPBD (Python)'),
        plt.Line2D([0],[0], color=C_SL,   marker='s', ls='--', lw=2, label='Simulink'),
        plt.Line2D([0],[0], color=C_CORRECTED,  marker='^',  lw=2, ls=':',  label='Simulink + GP'),
    ]
    fig.legend(handles=handles, loc='upper right', fontsize=10)
    fig.suptitle('XPBD Python vs Simulink — wire shape comparison', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'comparison.png'), dpi=150, bbox_inches='tight')
    print("\nSaved " + os.path.join(save_path, 'comparison.png'))
    #plt.show()

if __name__ == "__main__":
    for i in range(5):
        main()