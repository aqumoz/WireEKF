"""
estimate_parameters.py
=======================
Stage 1 — Estimate Young's modulus E from a static hang (TrackDLO shape).
Stage 2 — Estimate damping ratio ζ from a horizontal-release experiment.
Stage 3 — Static result plots.
Stage 4 — Live animation of the wire swinging from horizontal.

Edit the CONFIG section, then run:
    python estimate_parameters.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
from scipy.optimize import minimize_scalar

from wire_simulator import WireParams, WireSimulator
from sim_setup import WIRE, MATERIAL, SOLVER
from quaternion_utils import euler_xyz_to_quat


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — edit here
# ══════════════════════════════════════════════════════════════════════════════
DATA_PATH = r"data_Viktor\ctrl_mads_lauge_anders_20260507_112904\trackdlo_nodes.csv"   # TrackDLO CSV from static hang test

E_BOUNDS  = (1e6, 1e8)   # Pa  — search range for Young's modulus
E_GRID_N  = 12             # coarse grid points

T_PERIOD  = 12/13            # s   — measured oscillation period (one full swing)
T_SETTLE  = 12            # s   — measured settling time (wire stops moving)
# ══════════════════════════════════════════════════════════════════════════════

N   = WIRE.n_nodes
L   = WIRE.segment_length
D   = WIRE.diameter
RHO = MATERIAL.rho

print(f"Wire:  N={N}, L={L*1e3:.1f} mm, d={D*1e3:.1f} mm, rho={RHO:.1f} kg/m³")
print(f"E search: {E_BOUNDS[0]/1e6:.1f} – {E_BOUNDS[1]/1e6:.0f} MPa\n")


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════
def rotm_to_quat(R):
    tr = R[0,0]+R[1,1]+R[2,2]
    if tr > 0:
        s = 0.5/np.sqrt(tr+1)
        return np.array([(R[2,1]-R[1,2])*s,(R[0,2]-R[2,0])*s,(R[1,0]-R[0,1])*s,0.25/s])
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2*np.sqrt(1+R[0,0]-R[1,1]-R[2,2])
        return np.array([0.25*s,(R[0,1]+R[1,0])/s,(R[0,2]+R[2,0])/s,(R[2,1]-R[1,2])/s])
    elif R[1,1] > R[2,2]:
        s = 2*np.sqrt(1+R[1,1]-R[0,0]-R[2,2])
        return np.array([(R[0,1]+R[1,0])/s,0.25*s,(R[1,2]+R[2,1])/s,(R[0,2]-R[2,0])/s])
    else:
        s = 2*np.sqrt(1+R[2,2]-R[0,0]-R[1,1])
        return np.array([(R[0,2]+R[2,0])/s,(R[1,2]+R[2,1])/s,0.25*s,(R[1,0]-R[0,1])/s])


def state_from_pos(positions):
    xs = np.zeros(7*len(positions))
    for i, p in enumerate(positions):
        xs[7*i:7*i+3] = p
        d3 = positions[i+1]-p if i < len(positions)-1 else p-positions[i-1]
        d3 /= np.linalg.norm(d3)
        ref = np.array([0.,1.,0.]) if abs(d3[1]) < 0.9 else np.array([1.,0.,0.])
        d1  = np.cross(ref, d3);  d1 /= np.linalg.norm(d1)
        xs[7*i+3:7*i+7] = rotm_to_quat(np.column_stack([d1, np.cross(d3,d1), d3]))
    return xs


def make_sim(E, zeta=SOLVER.zeta_bend):
    r = D/2;  A = np.pi*r**2;  I = np.pi*r**4/4;  J = np.pi*r**4/2
    G = E/(2*(1+MATERIAL.nu));  m = RHO*A*L
    Ie   = SOLVER.I_eff_scale * m * L**2
    beta = zeta * (2 / np.sqrt(E*I/(m*L**3)))
    p = WireParams(N=N, dof=7, L=L,
        alpha_stretch=np.array([L/(G*A), L/(G*A), L/(E*A)]),
        alpha_bend   =np.array([L/(E*I), L/(E*I), L/(G*J)]),
        beta_stretch=np.zeros(3), beta_bend=np.full(3, beta),
        alpha_ee_pos=0., alpha_ee_orient=0.,
        beta_ee_pos=0.,  beta_ee_orient=0.,
        dt=SOLVER.dt, solver_iter=SOLVER.iterations,
        m_num=np.ones(N)*m, I_num=[np.eye(3)*Ie for _ in range(N)])
    return WireSimulator(p), p


def horizontal_init(E):
    """Wire pinned at origin, pointing horizontally along world X."""
    r=D/2; A=np.pi*r**2; I=np.pi*r**4/4; J=np.pi*r**4/2
    G=E/(2*(1+MATERIAL.nu)); m=RHO*A*L; Ie=SOLVER.I_eff_scale*m*L**2
    p = WireParams(N=N, dof=7, L=L,
        alpha_stretch=np.array([L/(G*A),L/(G*A),L/(E*A)]),
        alpha_bend   =np.array([L/(E*I),L/(E*I),L/(G*J)]),
        beta_stretch=np.zeros(3), beta_bend=np.zeros(3),
        alpha_ee_pos=0.,alpha_ee_orient=0.,beta_ee_pos=0.,beta_ee_orient=0.,
        dt=SOLVER.dt, solver_iter=SOLVER.iterations,
        m_num=np.ones(N)*m, I_num=[np.eye(3)*Ie for _ in range(N)])
    return WireSimulator(p).joint_to_world_init(
        np.zeros(3), np.array([0., np.pi/2, 0.]), np.zeros((3, N-1))
    )


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 — estimate E
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 55)
print("Stage 1 — Young's modulus from static hang")
print("=" * 55)

trk = pd.read_csv(DATA_PATH)
trk_nodes = np.zeros((len(trk), N, 3))
for i in range(N):
    trk_nodes[:,i,0] = trk[f'node_{i}_x'].values
    trk_nodes[:,i,1] = trk[f'node_{i}_y'].values
    trk_nodes[:,i,2] = trk[f'node_{i}_z'].values

ref  = trk_nodes.mean(axis=0)
rrel = ref - ref[0]
print(f"TrackDLO: {len(trk)} frames, noise ≈ {trk_nodes.std(axis=0).mean()*1e3:.2f} mm\n")

x0      = state_from_pos(ref)
ee_pos  = ref[0].copy()
ee_quat = x0[3:7].copy()

def equilibrium_shape(E):
    sim, p = make_sim(E)
    f = np.zeros(6*N)
    for i in range(N): f[i*6+2] = p.m_num[i]*(-9.81)
    x, v = x0.copy(), np.zeros(6*N)
    tail = []
    for s in range(150):
        x, v, _, _ = sim.estimate_wire_state(x, v, ee_pos, ee_quat, np.zeros(6), f)
        if s >= 120:
            tail.append((x.reshape(N,7)[:,:3] - x.reshape(N,7)[0,:3]).copy())
    return np.mean(tail, axis=0)

E_grid   = np.logspace(np.log10(E_BOUNDS[0]), np.log10(E_BOUNDS[1]), E_GRID_N)
err_grid = []
print(f"Scanning {E_GRID_N} grid points ...")
for E in E_grid:
    e = float(np.linalg.norm(equilibrium_shape(E) - rrel, axis=1).mean())
    err_grid.append(e)
    print(f"  E={E/1e6:6.2f} MPa  →  {e*1e3:.2f} mm")

err_grid = np.array(err_grid)
best_i   = np.argmin(err_grid)
lo = np.log10(E_grid[max(0, best_i-1)])
hi = np.log10(E_grid[min(len(E_grid)-1, best_i+1)])
res   = minimize_scalar(lambda logE: float(np.linalg.norm(
    equilibrium_shape(10**logE) - rrel, axis=1).mean()),
    bounds=(lo, hi), method='bounded')
E_opt = 10**res.x

print(f"\n  E_opt = {E_opt/1e6:.4f} MPa  (error = {res.fun*1e3:.2f} mm)")
print(f"  → sim_setup.py:  E : float = {E_opt:.4e}")


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 — estimate ζ
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("Stage 2 — Damping from horizontal-release experiment")
print("=" * 55)

omega_1   = 2 * np.pi / T_PERIOD
zeta_opt  = 4.6 / (omega_1 * T_SETTLE)
omega_seg = np.sqrt(E_opt * (np.pi*(D/2)**4/4) / (RHO*np.pi*(D/2)**2*L * L**3))
beta_bend = zeta_opt * (2 / omega_seg)

print(f"  ω₁        = {omega_1:.3f} rad/s")
print(f"  ζ         = {zeta_opt:.4f}")
print(f"  omega_seg = {omega_seg:.2f} rad/s")
print(f"  beta_bend = {beta_bend:.6f} s")
print(f"\n  → sim_setup.py:  zeta_bend : float = {zeta_opt:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3 — static plots
# ══════════════════════════════════════════════════════════════════════════════
print("\nGenerating plots ...")

# Run decay simulation for the plot
sim_d, p_d = make_sim(E_opt, zeta=zeta_opt)
ee_h = np.zeros(3)
eq_h = euler_xyz_to_quat(np.array([0., np.pi/2, 0.]))
f_d  = np.zeros(6*N)
for i in range(N): f_d[i*6+2] = p_d.m_num[i]*(-9.81)

x_h, v_h = horizontal_init(E_opt), np.zeros(6*N)
tip_z, t_arr = [], []
for s in range(int(max(T_SETTLE*2, 5.0) / SOLVER.dt)):
    x_h, v_h, _, _ = sim_d.estimate_wire_state(x_h, v_h, ee_h, eq_h, np.zeros(6), f_d)
    tip_z.append(x_h[7*(N-1)+2])
    t_arr.append(s * SOLVER.dt)

fig = plt.figure(figsize=(14, 5))

ax1 = fig.add_subplot(131)
ax1.semilogx(E_grid/1e6, err_grid*1e3, 'o-', color='#C62828', lw=1.8, ms=5)
ax1.axvline(E_opt/1e6,     color='#C62828', ls='--', lw=1.5, label=f'E_opt={E_opt/1e6:.2f} MPa')
ax1.axvline(MATERIAL.E/1e6, color='#888',   ls=':',  lw=1.5, label=f'Current={MATERIAL.E/1e6:.2f} MPa')
ax1.set_xlabel('E [MPa]');  ax1.set_ylabel('Mean node error [mm]')
ax1.set_title('Loss landscape');  ax1.legend(fontsize=8);  ax1.grid(alpha=0.3)

ax2 = fig.add_subplot(132, projection='3d')
opt_pts = equilibrium_shape(E_opt) + ref[0]
ax2.plot(*ref.T, 'o-', color='#1565C0', lw=2, ms=4, label='TrackDLO mean')
ax2.plot(*opt_pts.T, 's--', color='#C62828', lw=2, ms=4, label=f'E={E_opt/1e6:.2f} MPa')
ax2.scatter(*ee_pos, c='k', s=50, zorder=5)
ax2.set_xlabel('X');  ax2.set_ylabel('Y');  ax2.set_zlabel('Z')
ax2.set_title('Equilibrium shape');  ax2.legend(fontsize=8)
try: ax2.set_box_aspect([1,1,1])
except: pass

ax3 = fig.add_subplot(133)
ax3.plot(t_arr, tip_z, color='#C62828', lw=1.5, label=f'Sim (ζ={zeta_opt:.3f})')
ax3.axvline(T_SETTLE, color='#1565C0', ls='--', lw=1.5, label=f'T_settle={T_SETTLE} s')
ax3.axvline(T_PERIOD, color='#888',    ls=':',  lw=1.5, label=f'T_period={T_PERIOD} s')
ax3.set_xlabel('Time [s]');  ax3.set_ylabel('Tip Z [m]')
ax3.set_title('Decay curve');  ax3.legend(fontsize=8);  ax3.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('param_estimation_results.png', dpi=150, bbox_inches='tight')
print("Saved param_estimation_results.png")
plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Stage 4 — live animation
# ══════════════════════════════════════════════════════════════════════════════
print("\nStarting animation (close window to exit) ...")

sim_a, p_a = make_sim(E_opt, zeta=zeta_opt)
f_a = np.zeros(6*N)
for i in range(N): f_a[i*6+2] = p_a.m_num[i]*(-9.81)

x_a, v_a = horizontal_init(E_opt), np.zeros(6*N)
t_a, T_anim = 0.0, max(T_SETTLE*2, 5.0)

fig_a = plt.figure(figsize=(8, 7))
ax_a  = fig_a.add_subplot(111, projection='3d')
plt.ion();  plt.show()

while t_a <= T_anim:
    x_a, v_a, _, _ = sim_a.estimate_wire_state(x_a, v_a, ee_h, eq_h, np.zeros(6), f_a)
    pts = x_a.reshape(N, 7)[:, :3]

    ax_a.cla()
    ax_a.plot(pts[:,0], pts[:,1], pts[:,2], 'o-', color='#C62828', lw=2, ms=5)
    ax_a.scatter(*pts[0], c='k', s=60, zorder=5)

    half = WIRE.total_length/2 + 0.02
    mid  = pts.mean(axis=0)
    ax_a.set_xlim(mid[0]-half, mid[0]+half)
    ax_a.set_ylim(mid[1]-half, mid[1]+half)
    ax_a.set_zlim(mid[2]-half, mid[2]+half)
    ax_a.set_xlabel('X [m]');  ax_a.set_ylabel('Y [m]');  ax_a.set_zlabel('Z [m]')
    ax_a.set_title(f'Horizontal release  —  t = {t_a:.2f} s\n'
                   f'E = {E_opt/1e6:.2f} MPa    ζ = {zeta_opt:.3f}', fontsize=10)
    try: ax_a.set_box_aspect([1,1,1])
    except: pass

    plt.pause(0.001)
    t_a += SOLVER.dt

plt.ioff();  plt.show()