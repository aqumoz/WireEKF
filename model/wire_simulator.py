"""
wire_simulator.py
=================
XPBD Cosserat-rod wire simulator.

Automatically uses the compiled C++ backend (xpbd_core) for the hot loop
when available.  Falls back to pure Python if the extension hasn't been
built yet — behaviour is identical either way.

To build the C++ backend:
    pip install pybind11 numpy
    python setup.py build_ext --inplace
"""

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from quaternion_utils import quat_mul, quat_conj, quat_to_rotm, euler_xyz_to_quat
from constraints import (
    solve_stretch_shear, solve_bend_twist,
    solve_ee_position, solve_ee_orient,
    stretch_shear_violation, bend_twist_violation,
    ee_position_violation, ee_orient_violation,
)

# Try to import the compiled C++ backend
try:
    import xpbd_core as _cpp
    _CPP_AVAILABLE = True
except ImportError:
    _CPP_AVAILABLE = False


# ── Parameters ───────────────────────────────────────────────────────────────

@dataclass
class WireParams:
    N: int
    dof: int                       = 7
    L: float                       = 0.1
    alpha_stretch: np.ndarray      = field(default_factory=lambda: np.ones(3))
    alpha_bend: np.ndarray         = field(default_factory=lambda: np.ones(3))
    beta_stretch: np.ndarray       = field(default_factory=lambda: np.zeros(3))
    beta_bend: np.ndarray          = field(default_factory=lambda: np.zeros(3))
    alpha_ee_pos: float            = 0.0
    alpha_ee_orient: float         = 0.0
    beta_ee_pos: float             = 0.0
    beta_ee_orient: float          = 0.0
    dt: float                      = 0.05
    solver_iter: int               = 10
    m_num: np.ndarray              = field(default_factory=lambda: np.array([1.0]))
    I_num: List[np.ndarray]        = field(default_factory=list)


# ── Simulator ─────────────────────────────────────────────────────────────────

class WireSimulator:
    """
    XPBD position-based dynamics simulator for a Cosserat rod.

    Uses the compiled C++ backend automatically when available.
    """

    def __init__(self, params: WireParams):
        self.p = params
        p = params

        self._gamma_stretch   = p.alpha_stretch * p.beta_stretch / p.dt
        self._gamma_bend      = p.alpha_bend    * p.beta_bend    / p.dt
        self._gamma_ee_pos    = p.alpha_ee_pos    * p.beta_ee_pos    / p.dt
        self._gamma_ee_orient = p.alpha_ee_orient * p.beta_ee_orient / p.dt

        # Build C++ backend if available
        if _CPP_AVAILABLE:
            # Flatten I_num from list-of-3x3 to (N*9,) row-major
            I_flat = np.array([I.flatten() for I in p.I_num]).flatten()
            self._cpp = _cpp.WireSimulatorCPP(
                p.N, p.dof, p.L, p.dt, p.solver_iter,
                np.asarray(p.alpha_stretch, dtype=np.float64),
                np.asarray(p.alpha_bend,    dtype=np.float64),
                np.asarray(p.beta_stretch,  dtype=np.float64),
                np.asarray(p.beta_bend,     dtype=np.float64),
                float(p.alpha_ee_pos),    float(p.alpha_ee_orient),
                float(p.beta_ee_pos),     float(p.beta_ee_orient),
                np.asarray(p.m_num,         dtype=np.float64),
                np.asarray(I_flat,          dtype=np.float64),
            )
        else:
            self._cpp = None
            if not hasattr(WireSimulator, '_warned'):
                print("[wire_simulator] C++ backend not found — using pure Python.")
                print("  Build it with:  python setup.py build_ext --inplace")
                WireSimulator._warned = True

    # ── Initialisation helper ─────────────────────────────────────────────────

    def joint_to_world_init(
        self,
        base_pos: np.ndarray,
        base_euler_xyz: np.ndarray,
        joint_angles_xyz: np.ndarray,
    ) -> np.ndarray:
        p = self.p
        assert joint_angles_xyz.shape == (3, p.N - 1)

        x_init    = np.zeros(p.dof * p.N)
        q_world   = euler_xyz_to_quat(base_euler_xyz)
        pos_world = base_pos.copy().astype(float)

        x_init[0:3] = pos_world
        x_init[3:7] = q_world

        for k in range(1, p.N):
            R  = quat_to_rotm(q_world)
            pos_world = pos_world + p.L * R[:, 2]
            q_rel   = euler_xyz_to_quat(joint_angles_xyz[:, k - 1])
            q_world = quat_mul(q_world, q_rel)
            q_world /= np.linalg.norm(q_world)
            idx = k * p.dof
            x_init[idx:idx + 3]     = pos_world
            x_init[idx + 3:idx + 7] = q_world

        return x_init

    # ── Main step ─────────────────────────────────────────────────────────────

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
        Advance the simulation by one XPBD timestep.
        Returns (x_new, v_new, CW, ee_wrench).
        """
        if self._cpp is not None:
            return self._cpp.estimate_wire_state(
                np.ascontiguousarray(x_init, np.float64),
                np.ascontiguousarray(v_init, np.float64),
                np.ascontiguousarray(ee_pos,  np.float64),
                np.ascontiguousarray(ee_quat, np.float64),
                np.ascontiguousarray(f_ext,   np.float64),
            )
        return self._estimate_python(x_init, v_init, ee_pos, ee_quat, ft, f_ext)

    # ── Pure-Python fallback (original implementation) ────────────────────────

    def _estimate_python(self, x_init, v_init, ee_pos, ee_quat, ft, f_ext):
        p = self.p
        x_num = x_init.copy()
        v_num = v_init.copy()
        rest_darboux = np.array([0.0, 0.0, 0.0, 1.0])

        # 1. Prediction
        for i in range(p.N):
            inv_mass_rot_i = 3.0 / np.trace(p.I_num[i])
            sx, sv = i*p.dof, i*6
            v_num[sv:sv+3] += p.dt * f_ext[sv:sv+3] / p.m_num[i]
            x_num[sx:sx+3] += p.dt * v_num[sv:sv+3]
            tau = f_ext[sv+3:sv+6]; omega = v_num[sv+3:sv+6].copy()
            I = p.I_num[i]
            omega += p.dt * np.linalg.solve(I, tau - np.cross(omega, I @ omega))
            v_num[sv+3:sv+6] = omega
            q = x_num[sx+3:sx+7].copy()
            q += p.dt * 0.5 * quat_mul(q, np.array([omega[0],omega[1],omega[2],0.]))
            x_num[sx+3:sx+7] = q / np.linalg.norm(q)

        # 2. C^n
        C_str_n, C_ben_n = [], []
        for i in range(1, p.N):
            s0, s1 = (i-1)*p.dof, i*p.dof
            C_str_n.append(stretch_shear_violation(x_init[s0:s0+3],x_init[s1:s1+3],x_init[s0+3:s0+7],p.L))
            C_ben_n.append(bend_twist_violation(x_init[s0+3:s0+7],x_init[s1+3:s1+7],rest_darboux))
        C_ee_pos_n = ee_position_violation(x_init[0:3], ee_pos)
        C_ee_ori_n = ee_orient_violation(x_init[3:7], ee_quat)

        # 3. Lagrange multipliers
        lam_str = np.zeros((3, p.N-1)); lam_ben = np.zeros((3, p.N-1))
        lam_ee_pos = np.zeros(3); lam_ee_ori = np.zeros(3)
        inv_mass_rot_i = 3.0 / np.trace(p.I_num[0])

        # 4. GS loop
        for _ in range(p.solver_iter):
            for i in range(1, p.N):
                sp0,sp1,sq0 = (i-1)*p.dof, i*p.dof, (i-1)*p.dof+3
                c0,c1,cq0,dlam = solve_stretch_shear(
                    x_num[sp0:sp0+3], 1./p.m_num[i-1],
                    x_num[sp1:sp1+3], 1./p.m_num[i],
                    x_num[sq0:sq0+4], inv_mass_rot_i, p.L,
                    lam_str[:,i-1], p.alpha_stretch, p.dt,
                    C_loc_n=C_str_n[i-1], gamma=self._gamma_stretch)
                lam_str[:,i-1]+=dlam
                x_num[sp0:sp0+3]+=c0; x_num[sp1:sp1+3]+=c1; x_num[sq0:sq0+4]+=cq0
                x_num[sq0:sq0+4]/=np.linalg.norm(x_num[sq0:sq0+4])
            for i in range(1, p.N):
                sq0,sq1 = (i-1)*p.dof+3, i*p.dof+3
                cq0,cq1,dlam = solve_bend_twist(
                    x_num[sq0:sq0+4], inv_mass_rot_i, x_num[sq1:sq1+4], inv_mass_rot_i,
                    rest_darboux, lam_ben[:,i-1], p.alpha_bend, p.dt,
                    C_loc_n=C_ben_n[i-1], gamma=self._gamma_bend)
                lam_ben[:,i-1]+=dlam
                x_num[sq0:sq0+4]+=cq0; x_num[sq1:sq1+4]+=cq1
                x_num[sq0:sq0+4]/=np.linalg.norm(x_num[sq0:sq0+4])
                x_num[sq1:sq1+4]/=np.linalg.norm(x_num[sq1:sq1+4])
            for i in range(p.N):
                sq=i*p.dof+3; x_num[sq:sq+4]/=np.linalg.norm(x_num[sq:sq+4])
            c_p,dlam = solve_ee_position(x_num[0:3].copy(),1./p.m_num[0],ee_pos,lam_ee_pos,p.alpha_ee_pos,p.dt,C_n=C_ee_pos_n,gamma=self._gamma_ee_pos)
            lam_ee_pos+=dlam; x_num[0:3]+=c_p
            c_q,dlam = solve_ee_orient(x_num[3:7].copy(),inv_mass_rot_i,ee_quat,lam_ee_ori,p.alpha_ee_orient,p.dt,C_n=C_ee_ori_n,gamma=self._gamma_ee_orient)
            lam_ee_ori+=dlam; x_num[3:7]+=c_q; x_num[3:7]/=np.linalg.norm(x_num[3:7])

        # 5. Internal wrench
        F_int=np.zeros((3,p.N)); T_int=np.zeros((3,p.N))
        dt2=p.dt**2
        for i in range(1,p.N):
            R0=quat_to_rotm(x_num[(i-1)*p.dof+3:(i-1)*p.dof+7])
            fs=R0@lam_str[:,i-1]/dt2; F_int[:,i]+=fs; F_int[:,i-1]-=fs
        for i in range(1,p.N):
            ts=lam_ben[:,i-1]/dt2; T_int[:,i-1]-=ts; T_int[:,i]+=ts
        Fl=np.zeros((3,p.N)); Tl=np.zeros((3,p.N))
        for j in range(p.N):
            Rj=quat_to_rotm(x_num[j*p.dof+3:j*p.dof+7]); Fl[:,j]=Rj.T@F_int[:,j]; Tl[:,j]=Rj.T@T_int[:,j]
        CW=np.vstack([Fl,Tl]).flatten(order='F')

        ee_wrench=np.concatenate([lam_ee_pos/dt2, lam_ee_ori/dt2])

        # 6. Velocity update
        for i in range(p.N):
            sx,sv=i*p.dof,i*6
            v_num[sv:sv+3]=(x_num[sx:sx+3]-x_init[sx:sx+3])/p.dt
            qn=x_num[sx+3:sx+7].copy(); qo=x_init[sx+3:sx+7].copy()
            if np.dot(qn,qo)<0: qn=-qn
            v_num[sv+3:sv+6]=(2./p.dt)*quat_mul(qn,quat_conj(qo))[:3]

        return x_num, v_num, CW, ee_wrench
