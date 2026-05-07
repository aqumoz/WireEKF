from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from quaternion_utils import quat_mul, quat_conj, quat_to_rotm, euler_xyz_to_quat
from constraints import (
    solve_stretch_shear,
    solve_bend_twist,
    solve_ee_position,
    solve_ee_orient,
    stretch_shear_violation,
    bend_twist_violation,
    ee_position_violation,
    ee_orient_violation,
)


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class WireParams:
    """
    Simulation parameters for the XPBD Cosserat-rod wire model.

    Elastic parameters
    ------------------
    alpha_stretch : (3,) compliance  [shear_x, shear_y, axial_z]   [m/N]
    alpha_bend    : (3,) compliance  [bend_x,  bend_y,  twist_z]   [1/(N·m)]

    Damping parameters  (Rayleigh constraint damping, XPBD Eq. 26)
    ------------------
    beta_stretch  : (3,) or float   damping coefficient  [s]
    beta_bend     : (3,) or float   damping coefficient  [s]

    gamma is computed internally as  γ = α · β / dt  per component.
    Larger β → stronger velocity damping along the constraint direction.

    EE constraints
    --------------
    alpha_ee_pos / alpha_ee_orient : compliance  (0 = rigid pin)
    beta_ee_pos / beta_ee_orient   : damping (only active when alpha > 0)
    """
    N: int
    dof: int                       = 7
    L: float                       = 0.1

    # Elastic compliance
    alpha_stretch: np.ndarray      = field(default_factory=lambda: np.ones(3))
    alpha_bend: np.ndarray         = field(default_factory=lambda: np.ones(3))

    # Rayleigh constraint damping coefficients  [s]
    beta_stretch: np.ndarray       = field(default_factory=lambda: np.zeros(3))
    beta_bend: np.ndarray          = field(default_factory=lambda: np.zeros(3))

    # EE pin
    alpha_ee_pos: float            = 0.0
    alpha_ee_orient: float         = 0.0
    beta_ee_pos: float             = 0.0
    beta_ee_orient: float          = 0.0

    # Solver
    dt: float                      = 0.05
    solver_iter: int               = 10

    # Mass / inertia
    m_num: np.ndarray              = field(default_factory=lambda: np.array([1.0]))
    I_num: List[np.ndarray]        = field(default_factory=list)


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class WireSimulator:
    """
    XPBD position-based dynamics simulator for a Cosserat rod.

    Usage
    -----
    sim    = WireSimulator(params)
    x_init = sim.joint_to_world_init(base_pos, base_euler_xyz, joint_angles)
    v_init = np.zeros(6 * params.N)

    for each time step:
        x, v, CW, ee_wrench = sim.estimate_wire_state(
            x, v, ee_pos, ee_quat, ft, f_ext)
    """

    def __init__(self, params: WireParams):
        self.p = params

        # Precompute γ = α·β / dt for each constraint type (done once)
        # These are (3,) arrays for segment constraints.
        p = params
        self._gamma_stretch   = p.alpha_stretch * p.beta_stretch / p.dt
        self._gamma_bend      = p.alpha_bend    * p.beta_bend    / p.dt
        self._gamma_ee_pos    = p.alpha_ee_pos    * p.beta_ee_pos    / p.dt
        self._gamma_ee_orient = p.alpha_ee_orient * p.beta_ee_orient / p.dt

    # ------------------------------------------------------------------
    # Initialisation helper
    # ------------------------------------------------------------------

    def joint_to_world_init(
        self,
        base_pos: np.ndarray,
        base_euler_xyz: np.ndarray,
        joint_angles_xyz: np.ndarray,
    ) -> np.ndarray:
        """
        Build a world-frame state vector from joint-space rotations.

        Parameters
        ----------
        base_pos         : (3,)      world position of node 0
        base_euler_xyz   : (3,)      intrinsic X→Y→Z Euler angles for node 0
        joint_angles_xyz : (3, N-1)  relative rotation per joint [rx, ry, rz]

        Returns
        -------
        x_init : (dof*N,)  flat state vector
        """
        p = self.p
        assert joint_angles_xyz.shape == (3, p.N - 1), \
            f"joint_angles_xyz must be (3, {p.N-1}), got {joint_angles_xyz.shape}."

        x_init    = np.zeros(p.dof * p.N)
        q_world   = euler_xyz_to_quat(base_euler_xyz)
        pos_world = base_pos.copy().astype(float)

        x_init[0:3] = pos_world
        x_init[3:7] = q_world

        for k in range(1, p.N):
            R  = quat_to_rotm(q_world)
            d3 = R[:, 2]
            pos_world = pos_world + p.L * d3

            q_rel   = euler_xyz_to_quat(joint_angles_xyz[:, k - 1])
            q_world = quat_mul(q_world, q_rel)
            q_world /= np.linalg.norm(q_world)

            idx = k * p.dof
            x_init[idx:idx + 3]     = pos_world
            x_init[idx + 3:idx + 7] = q_world

        return x_init

    # ------------------------------------------------------------------
    # Main simulation step
    # ------------------------------------------------------------------

    def estimate_wire_state(
        self,
        x_init: np.ndarray,
        v_init: np.ndarray,
        ee_pos: np.ndarray,
        ee_quat: np.ndarray,
        ft: np.ndarray,
        f_ext: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Advance the simulation by one XPBD time step.

        Constraint-iteration ordering per Gauss-Seidel sweep:
            1. stretch / shear
            2. bend / twist
            3. normalise all quaternions
            4. EE position   ← applied last so node 0 ends at ee target
            5. EE orientation

        The XPBD damping term (Eq. 26) uses C^n precomputed from x_init
        (state at the very start of the step, before prediction).  This
        gives physically meaningful Rayleigh-type dissipation along each
        constraint direction without the ad-hoc 0.99 velocity bleed.

        Parameters
        ----------
        x_init  : (dof*N,)  state at start of step
        v_init  : (6*N,)    velocity at start of step
        ee_pos  : (3,)      EE position target  (world frame)
        ee_quat : (4,)      EE orientation target  [x,y,z,w]
        ft      : (6,)      force/torque applied to node 0
        f_ext   : (6*N,)    external wrenches per node

        Returns
        -------
        x_num     : (dof*N,)  updated state
        v_num     : (6*N,)    updated velocity
        CW        : (6*N,)    internal contact wrench, local frames,
                              column-major  (matches MATLAB CW_local(:))
        ee_wrench : (6,)      EE constraint wrench  [fx,fy,fz, tx,ty,tz]
                              force  — world frame;  torque — EE frame
        """
        p = self.p

        x_num = x_init.copy()
        v_num = v_init.copy()

        rest_darboux = np.array([0.0, 0.0, 0.0, 1.0])
        

        # ----------------------------------------------------------------
        # 1.  Prediction step  (semi-implicit Euler)
        # ----------------------------------------------------------------
        for i in range(p.N):

            inv_mass_rot_i = 3.0 / np.trace(p.I_num[i])   # trace/3 = average inertia

            s_x = i * p.dof
            s_v = i * 6

            f                   = f_ext[s_v:s_v + 3]
            v_num[s_v:s_v + 3] += p.dt * f / p.m_num[i]
            x_num[s_x:s_x + 3] += p.dt * v_num[s_v:s_v + 3]

            tau   = f_ext[s_v + 3:s_v + 6]
            omega = v_num[s_v + 3:s_v + 6].copy()
            I     = p.I_num[i]
            omega = omega + p.dt * np.linalg.solve(I, tau - np.cross(omega, I @ omega))
            v_num[s_v + 3:s_v + 6] = omega

            q     = x_num[s_x + 3:s_x + 7].copy()
            q_dot = 0.5 * quat_mul(q, np.array([omega[0], omega[1], omega[2], 0.0]))
            q     = q + p.dt * q_dot
            x_num[s_x + 3:s_x + 7] = q / np.linalg.norm(q)

        # ----------------------------------------------------------------
        # 2.  Precompute C^n  —  constraint violations at x^n (= x_init),
        #     used as the damping reference inside the solver loop.
        #     Evaluated ONCE before iterating; does not change per iteration.
        # ----------------------------------------------------------------
        C_stretch_n = []
        C_bend_n    = []
        for i in range(1, p.N):
            s0, s1 = (i - 1) * p.dof, i * p.dof
            p0_n = x_init[s0:s0 + 3];  q0_n = x_init[s0 + 3:s0 + 7]
            p1_n = x_init[s1:s1 + 3];  q1_n = x_init[s1 + 3:s1 + 7]
            C_stretch_n.append(stretch_shear_violation(p0_n, p1_n, q0_n, p.L))
            C_bend_n.append(bend_twist_violation(q0_n, q1_n, rest_darboux))

        C_ee_pos_n    = ee_position_violation(x_init[0:3], ee_pos)
        C_ee_orient_n = ee_orient_violation(x_init[3:7], ee_quat)

        # ----------------------------------------------------------------
        # 3.  Lagrange multipliers
        # ----------------------------------------------------------------
        lambda_stretch   = np.zeros((3, p.N - 1))
        lambda_bend      = np.zeros((3, p.N - 1))
        lambda_ee_pos    = np.zeros(3)
        lambda_ee_orient = np.zeros(3)

        # ----------------------------------------------------------------
        # 4.  Constraint solver  (Gauss-Seidel XPBD with Rayleigh damping)
        # ----------------------------------------------------------------
        for _ in range(p.solver_iter):

            # ---- Stretch / Shear ----------------------------------------
            for i in range(1, p.N):
                s_p0, s_p1 = (i - 1) * p.dof, i * p.dof
                s_q0       = s_p0 + 3

                p0 = x_num[s_p0:s_p0 + 3].copy()
                p1 = x_num[s_p1:s_p1 + 3].copy()
                q0 = x_num[s_q0:s_q0 + 4].copy()

                c0, c1, cq0, dlam = solve_stretch_shear(
                    p0, 1.0 / p.m_num[i - 1],
                    p1, 1.0 / p.m_num[i],
                    q0, inv_mass_rot_i,
                    p.L,
                    lambda_stretch[:, i - 1],
                    p.alpha_stretch, p.dt,
                    C_loc_n = C_stretch_n[i - 1],
                    gamma   = self._gamma_stretch,
                )
                lambda_stretch[:, i - 1] += dlam

                x_num[s_p0:s_p0 + 3] += c0
                x_num[s_p1:s_p1 + 3] += c1
                x_num[s_q0:s_q0 + 4] += cq0
                x_num[s_q0:s_q0 + 4] /= np.linalg.norm(x_num[s_q0:s_q0 + 4])

            # ---- Bend / Twist -------------------------------------------
            for i in range(1, p.N):
                s_q0 = (i - 1) * p.dof + 3
                s_q1 =  i      * p.dof + 3

                q0 = x_num[s_q0:s_q0 + 4].copy()
                q1 = x_num[s_q1:s_q1 + 4].copy()

                cq0, cq1, dlam = solve_bend_twist(
                    q0, inv_mass_rot_i, q1, inv_mass_rot_i,
                    rest_darboux,
                    lambda_bend[:, i - 1],
                    p.alpha_bend, p.dt,
                    C_loc_n = C_bend_n[i - 1],
                    gamma   = self._gamma_bend,
                )
                lambda_bend[:, i - 1] += dlam

                x_num[s_q0:s_q0 + 4] += cq0
                x_num[s_q1:s_q1 + 4] += cq1
                x_num[s_q0:s_q0 + 4] /= np.linalg.norm(x_num[s_q0:s_q0 + 4])
                x_num[s_q1:s_q1 + 4] /= np.linalg.norm(x_num[s_q1:s_q1 + 4])

            # Normalise all quaternions
            for i in range(p.N):
                s_q = i * p.dof + 3
                x_num[s_q:s_q + 4] /= np.linalg.norm(x_num[s_q:s_q + 4])

            # ---- EE Position  (applied LAST) ----------------------------
            c_p, dlam = solve_ee_position(
                x_num[0:3].copy(), 1.0 / p.m_num[0], ee_pos,
                lambda_ee_pos, p.alpha_ee_pos, p.dt,
                C_n=C_ee_pos_n, gamma=self._gamma_ee_pos,
            )
            lambda_ee_pos += dlam
            x_num[0:3]    += c_p

            # ---- EE Orientation  (applied LAST) -------------------------
            c_q, dlam = solve_ee_orient(
                x_num[3:7].copy(), inv_mass_rot_i, ee_quat,
                lambda_ee_orient, p.alpha_ee_orient, p.dt,
                C_n=C_ee_orient_n, gamma=self._gamma_ee_orient,
            )
            lambda_ee_orient += dlam
            x_num[3:7]       += c_q
            x_num[3:7]       /= np.linalg.norm(x_num[3:7])

        # ----------------------------------------------------------------
        # 5.  Internal force and torque recovery  (segment constraints)
        # ----------------------------------------------------------------
        F_internal   = np.zeros((3, p.N))
        Tau_internal = np.zeros((3, p.N))

        for i in range(1, p.N):
            s_q0 = (i - 1) * p.dof + 3
            R0   = quat_to_rotm(x_num[s_q0:s_q0 + 4])
            f_seg = R0 @ lambda_stretch[:, i - 1] / p.dt**2
            F_internal[:, i]     += f_seg
            F_internal[:, i - 1] -= f_seg

        for i in range(1, p.N):
            tau_seg = lambda_bend[:, i - 1] / p.dt**2
            Tau_internal[:, i - 1] -= tau_seg
            Tau_internal[:, i]     += tau_seg

        F_local   = np.zeros((3, p.N))
        Tau_local = np.zeros((3, p.N))
        for j in range(p.N):
            s_q = j * p.dof + 3
            Rj  = quat_to_rotm(x_num[s_q:s_q + 4])
            F_local[:, j]   = Rj.T @ F_internal[:, j]
            Tau_local[:, j] = Rj.T @ Tau_internal[:, j]

        CW = np.vstack([F_local, Tau_local]).flatten(order='F')

        # ----------------------------------------------------------------
        # 6.  EE constraint wrench
        #     Force  (world frame) = lambda_ee_pos    / dt²
        #     Torque (EE frame)    = lambda_ee_orient / dt²
        # ----------------------------------------------------------------
        ee_wrench = np.concatenate([
            lambda_ee_pos    / p.dt**2,
            lambda_ee_orient / p.dt**2,
        ])

        # ----------------------------------------------------------------
        # 7.  Velocity update  (no ad-hoc velocity bleed — damping is
        #     handled physically by the beta parameters above)
        # ----------------------------------------------------------------
        for i in range(p.N):
            s_x = i * p.dof
            s_v = i * 6

            v_num[s_v:s_v + 3] = (
                (x_num[s_x:s_x + 3] - x_init[s_x:s_x + 3]) / p.dt
            )

            q_new = x_num[s_x + 3:s_x + 7].copy()
            q_old = x_init[s_x + 3:s_x + 7].copy()
            if np.dot(q_new, q_old) < 0:
                q_new = -q_new

            q_diff = quat_mul(q_new, quat_conj(q_old))
            v_num[s_v + 3:s_v + 6] = (2.0 / p.dt) * q_diff[:3]

        return x_num, v_num, CW, ee_wrench
