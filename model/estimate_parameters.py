"""
estimate_parameters.py
======================
Identify wire material parameters by matching the simulated static
equilibrium shape to the TrackDLO reference shape.

Parameters that can be estimated from a static hang test
---------------------------------------------------------
  E   — Young's modulus [Pa]
        Controls bending stiffness EI. Higher E → less sag.

  rho — density [kg/m³]
        Controls weight per unit length ρA. Higher rho → more sag.

  d   — wire diameter [m]
        Affects both stiffness (I ∝ d⁴) and weight (A ∝ d²).
        Only identifiable if E and rho are reasonably well known.

Note: E and rho are not fully independent in a static hang — the
equilibrium shape depends primarily on EI / (ρAg), the ratio of
bending stiffness to weight per unit length.  The script estimates
them jointly but warns if the landscape is poorly conditioned.

Usage
-----
1. Set PARAMS_TO_FIT to select which parameters to estimate.
2. Set the search bounds in BOUNDS below.
3. Run:  python estimate_parameters.py

The script runs in two stages:
  1. Coarse grid scan — maps the loss landscape and locates the minimum.
  2. Nelder-Mead optimisation — refines the estimate.

All other parameters (geometry, solver settings) come from sim_setup.py.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import product
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wire_simulator import WireParams, WireSimulator
from sim_setup import WIRE, MATERIAL, SOLVER, gravity_forces

DATA = 'data_Viktor\ctrl_mads_lauge_anders_20260507_112904'
OUT  = 'data_Viktor\ctrl_mads_lauge_anders_20260507_112904'
os.makedirs(OUT, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Configuration — edit here
# ══════════════════════════════════════════════════════════════════════════════

# Which parameters to estimate (any combination of 'E', 'rho', 'd')
# rho and d are now derived from physical measurements (mass=26.6g, d=16mm)
# so only E remains free.
PARAMS_TO_FIT = ['E']

# Search bounds (log10 scale for E and rho, linear for d)
BOUNDS = {
    'E'   : (5e5,  50e6),    # Pa      — 0.5 MPa to 50 MPa
    'rho' : (200,  2000),    # kg/m³   — not used (fixed from measurements)
    'd'   : (0.008, 0.025),  # m       — not used (fixed from measurements)
}

# Grid points per parameter for the coarse scan
GRID_POINTS = {1: 12, 2: 7, 3: 5}[len(PARAMS_TO_FIT)]

# Solver settings for equilibrium runs
# More iterations = more accurate but slower. 30 = production quality.
SIM_ITER  = 30    # XPBD iterations per step
SIM_DT    = 0.01  # s — time step
SIM_STEPS = 150   # steps to run per evaluation  (~1.5 s sim time)
SIM_AVG   = 30    # average last N steps to suppress residual oscillation


# ══════════════════════════════════════════════════════════════════════════════
# 1.  Load TrackDLO data
# ══════════════════════════════════════════════════════════════════════════════
trk = pd.read_csv(f'{DATA}/trackdlo_nodes.csv')
N   = WIRE.n_nodes

trk_nodes = np.zeros((len(trk), N, 3))
for i in range(N):
    trk_nodes[:, i, 0] = trk[f'node_{i}_x'].values
    trk_nodes[:, i, 1] = trk[f'node_{i}_y'].values
    trk_nodes[:, i, 2] = trk[f'node_{i}_z'].values

# Mean shape — clean equilibrium reference
ref_shape  = trk_nodes.mean(axis=0)          # (N, 3)
noise_mm   = trk_nodes.std(axis=0).mean() * 1e3

# Reference expressed relative to node 0 — removes EE placement bias
# and lets us focus purely on the wire shape
ref_rel = ref_shape - ref_shape[0]           # (N, 3), ref_rel[0] = [0,0,0]

print(f"TrackDLO: {len(trk)} frames, {N} nodes")
print(f"Frame-to-frame noise: {noise_mm:.2f} mm  (resolution limit)")
print(f"Estimating: {PARAMS_TO_FIT}")
print(f"Grid points per parameter: {GRID_POINTS}")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Helpers
# ══════════════════════════════════════════════════════════════════════════════
def rotm_to_quat(R):
    tr = R[0,0] + R[1,1] + R[2,2]
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        return np.array([(R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s,
                         (R[1,0]-R[0,1])*s, 0.25/s])
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        return np.array([0.25*s, (R[0,1]+R[1,0])/s,
                         (R[0,2]+R[2,0])/s, (R[2,1]-R[1,2])/s])
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        return np.array([(R[0,1]+R[1,0])/s, 0.25*s,
                         (R[1,2]+R[2,1])/s, (R[0,2]-R[2,0])/s])
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        return np.array([(R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s,
                         0.25*s,             (R[1,0]-R[0,1])/s])


def state_from_positions(positions):
    xs = np.zeros(7 * len(positions))
    for i, p in enumerate(positions):
        xs[7*i:7*i+3] = p
        d3 = (positions[i+1] - p if i < len(positions)-1 else p - positions[i-1])
        d3 /= np.linalg.norm(d3)
        ref = np.array([0.,1.,0.]) if abs(d3[1]) < 0.9 else np.array([1.,0.,0.])
        d1 = np.cross(ref, d3);  d1 /= np.linalg.norm(d1)
        d2 = np.cross(d3, d1)
        xs[7*i+3:7*i+7] = rotm_to_quat(np.column_stack([d1, d2, d3]))
    return xs


# EE pin — derive orientation from the mean shape so it is noise-averaged
x_ref   = state_from_positions(ref_shape)
ee_pos  = ref_shape[0].copy()
ee_quat = x_ref[3:7].copy()

# Initial state — start sim from mean shape to reduce transient
x_init  = x_ref.copy()


def build_sim(E, rho, d):
    """Build a WireSimulator for the given parameters."""
    L  = WIRE.total_length / N
    r  = d / 2.0
    A  = np.pi * r**2
    I  = np.pi * r**4 / 4
    J  = np.pi * r**4 / 2
    G  = E / (2.0 * (1.0 + MATERIAL.nu))

    alpha_stretch = np.array([L/(G*A), L/(G*A), L/(E*A)])
    alpha_bend    = np.array([L/(E*I), L/(E*I), L/(G*J)])

    m_node = rho * A * L
    m_num  = np.ones(N) * m_node
    I_eff  = SOLVER.I_eff_scale * m_node * L**2
    I_num  = [np.eye(3) * I_eff for _ in range(N)]

    omega  = np.sqrt(E * I / (m_node * L**3))
    beta   = SOLVER.zeta_bend * (2.0 / omega)

    params = WireParams(
        N=N, dof=7, L=L,
        alpha_stretch=alpha_stretch, alpha_bend=alpha_bend,
        beta_stretch=np.zeros(3), beta_bend=np.full(3, beta),
        alpha_ee_pos=0., alpha_ee_orient=0.,
        beta_ee_pos=0.,  beta_ee_orient=0.,
        dt=SIM_DT, solver_iter=SIM_ITER,
        m_num=m_num, I_num=I_num,
    )
    return WireSimulator(params), params


def run_equilibrium(E, rho, d):
    """
    Run to equilibrium and return the mean shape of the last SIM_AVG steps.
    Shape is returned relative to node 0 to remove EE placement offsets.
    """
    sim, params = build_sim(E, rho, d)
    f_ext = np.zeros(6 * N)
    for i in range(N):
        f_ext[i*6 + 2] = params.m_num[i] * (-9.81)

    x, v = x_init.copy(), np.zeros(6 * N)
    tail  = []

    for step in range(SIM_STEPS):
        x, v, _, _ = sim.estimate_wire_state(
            x, v, ee_pos, ee_quat, np.zeros(6), f_ext
        )
        if step >= SIM_STEPS - SIM_AVG:
            pts = x.reshape(N, 7)[:, :3]
            tail.append((pts - pts[0]).copy())   # relative to node 0

    return np.mean(tail, axis=0)   # (N, 3)


def shape_error(E, rho, d):
    """Mean Euclidean node error [m] between sim equilibrium and reference."""
    sim_rel = run_equilibrium(E, rho, d)
    return float(np.linalg.norm(sim_rel - ref_rel, axis=1).mean())


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Pack / unpack parameter vector for the optimiser
# ══════════════════════════════════════════════════════════════════════════════
# Fixed (not being estimated) values come from sim_setup
FIXED = {
    'E'  : MATERIAL.E,
    'rho': MATERIAL.rho,
    'd'  : WIRE.diameter,
}

def pack(params_dict):
    """Dict → log10 vector (log scale for E and rho, linear for d)."""
    vec = []
    for p in PARAMS_TO_FIT:
        if p in ('E', 'rho'):
            vec.append(np.log10(params_dict[p]))
        else:
            vec.append(params_dict[p])
    return np.array(vec)


def unpack(vec):
    """log10 vector → (E, rho, d) tuple."""
    d = dict(FIXED)
    for i, p in enumerate(PARAMS_TO_FIT):
        d[p] = 10**vec[i] if p in ('E', 'rho') else vec[i]
    return d['E'], d['rho'], d['d']


def objective(vec):
    E, rho, d = unpack(vec)
    # Clip to bounds to avoid nonsensical evaluations
    E   = np.clip(E,   *BOUNDS['E'])
    rho = np.clip(rho, *BOUNDS['rho'])
    d   = np.clip(d,   *BOUNDS['d'])
    err = shape_error(E, rho, d)
    vals = {p: 10**vec[i] if p in ('E','rho') else vec[i]
            for i,p in enumerate(PARAMS_TO_FIT)}
    label = '  '.join(f"{p}={vals[p]/1e6:.3f}MPa" if p=='E'
                      else f"{p}={vals[p]:.1f}" for p in PARAMS_TO_FIT)
    print(f"  {label}  →  {err*1e3:.2f} mm")
    return err


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Coarse grid scan
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"Stage 1: Coarse grid  ({GRID_POINTS}^{len(PARAMS_TO_FIT)} = "
      f"{GRID_POINTS**len(PARAMS_TO_FIT)} evaluations)")
print(f"{'='*60}")

grid_axes = []
for p in PARAMS_TO_FIT:
    lo, hi = BOUNDS[p]
    if p in ('E', 'rho'):
        grid_axes.append(np.logspace(np.log10(lo), np.log10(hi), GRID_POINTS))
    else:
        grid_axes.append(np.linspace(lo, hi, GRID_POINTS))

grid_errors = np.zeros([GRID_POINTS] * len(PARAMS_TO_FIT))
grid_coords = list(product(*[range(GRID_POINTS)] * len(PARAMS_TO_FIT)))

best_err = np.inf
best_vec = None

for idx in grid_coords:
    vals = {p: grid_axes[k][idx[k]] for k, p in enumerate(PARAMS_TO_FIT)}
    vals.update({p: FIXED[p] for p in FIXED if p not in PARAMS_TO_FIT})
    err = shape_error(vals['E'], vals['rho'], vals['d'])
    grid_errors[idx] = err
    if err < best_err:
        best_err = err
        best_vec = pack(vals)
    label = '  '.join(f"{p}={vals[p]/1e6:.3f}MPa" if p=='E'
                      else f"{p}={vals[p]:.1f}" for p in PARAMS_TO_FIT)
    print(f"  {label}  →  {err*1e3:.2f} mm")

best_grid_vals = dict(zip(PARAMS_TO_FIT,
                          [10**best_vec[i] if p in ('E','rho') else best_vec[i]
                           for i,p in enumerate(PARAMS_TO_FIT)]))
print(f"\nGrid best: {best_err*1e3:.2f} mm  @ "
      + "  ".join(f"{p}={v/1e6:.3f}MPa" if p=='E' else f"{p}={v:.1f}"
                  for p,v in best_grid_vals.items()))


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Nelder-Mead refinement
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("Stage 2: Nelder-Mead refinement")
print(f"{'='*60}")

result = minimize(
    objective, best_vec,
    method='Nelder-Mead',
    options={'xatol': 0.02, 'fatol': 5e-4, 'maxiter': 60, 'disp': False},
)

E_opt, rho_opt, d_opt = unpack(result.x)
err_opt = result.fun

print(f"\n{'='*60}")
print("RESULTS")
print(f"{'='*60}")
for p in PARAMS_TO_FIT:
    v    = {'E': E_opt, 'rho': rho_opt, 'd': d_opt}[p]
    cur  = FIXED[p]
    unit = 'Pa' if p == 'E' else ('kg/m³' if p == 'rho' else 'm')
    print(f"  {p:3s}  =  {v:.4g} {unit}   (was {cur:.4g} {unit})")
print(f"\n  Mean node error  = {err_opt*1e3:.2f} mm")
print(f"  TrackDLO noise   = {noise_mm:.2f} mm")
print(f"\n  Update sim_setup.py:")
if 'E'   in PARAMS_TO_FIT: print(f"    E   : float = {E_opt:.4e}")
if 'rho' in PARAMS_TO_FIT: print(f"    rho : float = {rho_opt:.4f}")
if 'd'   in PARAMS_TO_FIT:
    print(f"    # diameter → update WIRE.diameter = {d_opt:.4f}")
print(f"{'='*60}")


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Plots
# ══════════════════════════════════════════════════════════════════════════════
C_REF = '#1565C0'
C_OPT = '#C62828'
C_OLD = '#888888'

# ── 6A. Loss landscape (1D or 2D heatmap) ───────────────────────────────────
if len(PARAMS_TO_FIT) == 1:
    p0 = PARAMS_TO_FIT[0]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.semilogx(grid_axes[0], grid_errors.flatten() * 1e3,
                'o-', color=C_OPT, lw=1.8, ms=5)
    ax.axvline(best_grid_vals[p0], color=C_OPT, ls='--', lw=1.5,
               label=f'Optimal {p0} = {best_grid_vals[p0]:.4g}')
    ax.axvline(FIXED[p0], color=C_OLD, ls=':', lw=1.5,
               label=f'Current {p0} = {FIXED[p0]:.4g}')
    ax.axhline(noise_mm, color=C_REF, ls='--', lw=1.2,
               label=f'TrackDLO noise  {noise_mm:.1f} mm')
    ax.set_xlabel(p0); ax.set_ylabel('Mean node error [mm]')
    ax.set_title(f'Loss landscape — {p0}', fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()

elif len(PARAMS_TO_FIT) == 2:
    p0, p1 = PARAMS_TO_FIT
    fig, ax = plt.subplots(figsize=(8, 6))
    X = grid_axes[0]
    Y = grid_axes[1]
    Z = grid_errors * 1e3   # mm

    # imshow with log-scale axes
    im = ax.pcolormesh(np.log10(X) if p0 in ('E','rho') else X,
                       np.log10(Y) if p1 in ('E','rho') else Y,
                       Z.T, cmap='viridis_r', shading='auto')
    plt.colorbar(im, ax=ax, label='Mean node error [mm]')

    opt_x = np.log10(best_grid_vals[p0]) if p0 in ('E','rho') else best_grid_vals[p0]
    opt_y = np.log10(best_grid_vals[p1]) if p1 in ('E','rho') else best_grid_vals[p1]
    ax.plot(opt_x, opt_y, 'r*', ms=14, label='Grid minimum')

    fix_x = np.log10(FIXED[p0]) if p0 in ('E','rho') else FIXED[p0]
    fix_y = np.log10(FIXED[p1]) if p1 in ('E','rho') else FIXED[p1]
    ax.plot(fix_x, fix_y, 'w^', ms=10, label='Current sim_setup')

    def fmt_ax(ax_obj, p, which):
        ticks = grid_axes[PARAMS_TO_FIT.index(p)]
        if p in ('E', 'rho'):
            ax_obj.set_ticks = None  # handled by log ticks
        lbl = f'{p} (log10)' if p in ('E','rho') else p
        if which == 'x': ax.set_xlabel(lbl)
        else: ax.set_ylabel(lbl)

    fmt_ax(ax, p0, 'x'); fmt_ax(ax, p1, 'y')
    ax.set_title(f'Loss landscape — {p0} vs {p1}', fontweight='bold')
    ax.legend(fontsize=9)
    plt.tight_layout()

p1_path = f'{OUT}/param_loss_landscape.png'
plt.savefig(p1_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved {p1_path}")

# ── 6B. Shape comparison ─────────────────────────────────────────────────────
sim_rel_opt = run_equilibrium(E_opt,        rho_opt,        d_opt)
sim_rel_cur = run_equilibrium(MATERIAL.E,   MATERIAL.rho,   WIRE.diameter)

# Convert back to world frame for plotting
sim_pts_opt = sim_rel_opt + ref_shape[0]
sim_pts_cur = sim_rel_cur + ref_shape[0]

fig = plt.figure(figsize=(10, 7))
ax3 = fig.add_subplot(111, projection='3d')

for fi in range(0, len(trk), max(1, len(trk)//15)):
    ax3.plot(trk_nodes[fi,:,0], trk_nodes[fi,:,1], trk_nodes[fi,:,2],
             color=C_REF, lw=0.4, alpha=0.15)

ax3.plot(ref_shape[:,0], ref_shape[:,1], ref_shape[:,2],
         'o-', color=C_REF, lw=2, ms=4, label='TrackDLO mean')

opt_lbl = '  '.join(f"{p}={best_grid_vals.get(p, FIXED[p]) / 1e6:.2f}MPa"
                    if p == 'E' else
                    f"{p}={best_grid_vals.get(p, FIXED[p]):.1f}"
                    for p in ['E','rho','d'] if p in PARAMS_TO_FIT)
ax3.plot(sim_pts_opt[:,0], sim_pts_opt[:,1], sim_pts_opt[:,2],
         's--', color=C_OPT, lw=2, ms=5, label=f'Optimal: {opt_lbl}')
ax3.plot(sim_pts_cur[:,0], sim_pts_cur[:,1], sim_pts_cur[:,2],
         '^:', color=C_OLD, lw=1.5, ms=5, label='Current sim_setup')
ax3.scatter(*ee_pos, c='k', s=60, zorder=5, label='EE pin')

all_pts = np.vstack([ref_shape, sim_pts_opt])
mid  = all_pts.mean(axis=0)
half = np.ptp(all_pts, axis=0).max() / 2 + 0.03
ax3.set_xlim(mid[0]-half, mid[0]+half)
ax3.set_ylim(mid[1]-half, mid[1]+half)
ax3.set_zlim(mid[2]-half, mid[2]+half)
ax3.set_xlabel('X [m]'); ax3.set_ylabel('Y [m]'); ax3.set_zlabel('Z [m]')
ax3.set_title('Equilibrium shape: TrackDLO vs simulation', fontweight='bold')
ax3.legend(fontsize=9)
try: ax3.set_box_aspect([1,1,1])
except AttributeError: pass

plt.tight_layout()
p2_path = f'{OUT}/param_shape_comparison.png'
plt.savefig(p2_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved {p2_path}")


# ── Helpers for Stage 3 visualisation ────────────────────────────────────────

def build_sim_damped(E, rho, d, zeta):
    """Build a WireSimulator with a specific damping ratio ζ."""
    L  = WIRE.total_length / N
    r  = d / 2.0
    A  = np.pi * r**2
    I  = np.pi * r**4 / 4
    J  = np.pi * r**4 / 2
    G  = E / (2.0 * (1.0 + MATERIAL.nu))
    alpha_stretch = np.array([L/(G*A), L/(G*A), L/(E*A)])
    alpha_bend    = np.array([L/(E*I), L/(E*I), L/(G*J)])
    m_node = rho * A * L
    m_num  = np.ones(N) * m_node
    I_eff  = SOLVER.I_eff_scale * m_node * L**2
    I_num  = [np.eye(3) * I_eff for _ in range(N)]
    omega  = np.sqrt(E * I / (m_node * L**3))
    beta   = zeta * (2.0 / omega)
    params = WireParams(
        N=N, dof=7, L=L,
        alpha_stretch=alpha_stretch, alpha_bend=alpha_bend,
        beta_stretch=np.zeros(3), beta_bend=np.full(3, beta),
        alpha_ee_pos=0., alpha_ee_orient=0.,
        beta_ee_pos=0.,  beta_ee_orient=0.,
        dt=SIM_DT, solver_iter=SIM_ITER,
        m_num=m_num, I_num=I_num,
    )
    return WireSimulator(params), params, beta


def horizontal_state(E, rho, d):
    """Initial state: wire pinned at origin, pointing horizontally along world X."""
    L  = WIRE.total_length / N
    r  = d / 2.0
    A  = np.pi * r**2
    I  = np.pi * r**4 / 4
    J  = np.pi * r**4 / 2
    G  = E / (2.0 * (1.0 + MATERIAL.nu))
    m_node = rho * A * L
    I_eff  = SOLVER.I_eff_scale * m_node * L**2
    tmp_params = WireParams(
        N=N, dof=7, L=L,
        alpha_stretch=np.array([L/(G*A), L/(G*A), L/(E*A)]),
        alpha_bend=np.array([L/(E*I), L/(E*I), L/(G*J)]),
        beta_stretch=np.zeros(3), beta_bend=np.zeros(3),
        alpha_ee_pos=0., alpha_ee_orient=0.,
        beta_ee_pos=0., beta_ee_orient=0.,
        dt=SIM_DT, solver_iter=SIM_ITER,
        m_num=np.ones(N) * m_node,
        I_num=[np.eye(3) * I_eff for _ in range(N)],
    )
    tmp_sim = WireSimulator(tmp_params)
    return tmp_sim.joint_to_world_init(
        np.array([0., 0., 0.]),
        np.array([0., np.pi/2, 0.]),   # z-axis → world X (horizontal)
        np.zeros((3, N - 1)),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3:  Estimate beta_bend from horizontal-release experiment
#
# You need two measurements from the experiment:
#   T_PERIOD          — time for one full oscillation (tip goes one way and
#                       comes back to the same side)
#   T_SETTLE_MEASURED — time until the wire visibly stopped moving
#
# From these, ζ follows directly with no simulation needed:
#   ω₁ = 2π / T_PERIOD
#   ζ  = 4.6 / (ω₁ · T_SETTLE)
#
# beta_bend then converts ζ to the XPBD Rayleigh coefficient using the
# single-segment natural frequency omega_seg (depends on E from Stage 2).
# ══════════════════════════════════════════════════════════════════════════════

T_PERIOD          = 6/13   # s  ← SET: time for one full oscillation
T_SETTLE_MEASURED = 6    # s  ← SET: time until visibly stopped

print(f"\n{'='*60}")
print(f"Stage 3: Damping from measured period + settling time")
print(f"{'='*60}")
print(f"  T_period  = {T_PERIOD} s")
print(f"  T_settle  = {T_SETTLE_MEASURED} s")

# ── Step 1: damping ratio directly from measurements ─────────────────────────
omega_1_measured = 2.0 * np.pi / T_PERIOD
zeta_opt         = 4.6 / (omega_1_measured * T_SETTLE_MEASURED)

print(f"\n  ω₁  = 2π / T_period = {omega_1_measured:.4f} rad/s")
print(f"  ζ   = 4.6 / (ω₁ · T_settle) = {zeta_opt:.4f}")

# ── Step 2: omega_seg from identified E ──────────────────────────────────────
# omega_seg is the natural bending frequency of one wire segment.
# It converts the dimensionless ζ into beta_bend [s] for the XPBD solver.
L_seg  = WIRE.total_length / N
r_opt  = d_opt / 2.0
A_opt  = np.pi * r_opt**2
I_opt  = np.pi * r_opt**4 / 4
m_node_opt  = rho_opt * A_opt * L_seg

omega_seg_opt = np.sqrt(E_opt * I_opt / (m_node_opt * L_seg**3))
beta_bend_opt = zeta_opt * (2.0 / omega_seg_opt)

print(f"\n  omega_seg = sqrt(E·I / (m·L³)) = {omega_seg_opt:.2f} rad/s")
print(f"  beta_bend = ζ · (2/omega_seg)   = {beta_bend_opt:.6f} s")

# ── Results ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  DAMPING RESULT")
print(f"{'='*60}")
print(f"  zeta_bend = {zeta_opt:.4f}  (from experiment)")
print(f"  beta_bend = {beta_bend_opt:.6f} s")
print(f"\n  Update sim_setup.py:")
print(f"    zeta_bend : float = {zeta_opt:.4f}")
print(f"{'='*60}")

# ── Plot: visualise what the identified damping looks like ────────────────────
# Run a short simulation with the identified damping to show the decay curve
print("\n  Running decay simulation for visualisation ...")

sim_decay, params_decay, _ = build_sim_damped(E_opt, rho_opt, d_opt, zeta_opt)

from quaternion_utils import euler_xyz_to_quat
ee_pos_h  = np.array([0., 0., 0.])
ee_quat_h = euler_xyz_to_quat(np.array([0., np.pi/2, 0.]))

f_ext_decay = np.zeros(6 * N)
for i in range(N):
    f_ext_decay[i*6 + 2] = params_decay.m_num[i] * (-9.81)

x_decay = horizontal_state(E_opt, rho_opt, d_opt)
v_decay = np.zeros(6 * N)

tip_z_history = []
t_history     = []
n_decay_steps = int(max(T_SETTLE_MEASURED * 2, 5.0) / SIM_DT)

for step in range(n_decay_steps):
    x_decay, v_decay, _, _ = sim_decay.estimate_wire_state(
        x_decay, v_decay, ee_pos_h, ee_quat_h, np.zeros(6), f_ext_decay
    )
    tip_z_history.append(x_decay[7*(N-1) + 2])   # tip Z position
    t_history.append(step * SIM_DT)

tip_z  = np.array(tip_z_history)
t_hist = np.array(t_history)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(t_hist, tip_z, color=C_OPT, lw=1.5, label=f'Sim tip Z  (ζ={zeta_opt:.3f})')
ax.axvline(T_SETTLE_MEASURED, color=C_REF, ls='--', lw=1.5,
           label=f'Measured settle  {T_SETTLE_MEASURED} s')
ax.axvline(T_PERIOD, color=C_OLD, ls=':', lw=1.5,
           label=f'Measured period  {T_PERIOD} s')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Tip Z position [m]')
ax.set_title('Horizontal release decay — identified damping', fontweight='bold')
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
p3_path = f'{OUT}/damping_decay_curve.png'
plt.savefig(p3_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved {p3_path}")