from dataclasses import dataclass
import numpy as np

from wire_simulator import WireParams, WireSimulator

@dataclass(frozen=True)
class _Wire:
    """Physical constants for the cable under test."""

    # Overall
    total_length : float = 0.460       # m
    n_nodes      : int   = 20          # number of simulation nodes
    diameter     : float = 0.016       # m

    # Derived cross-section (solid circular)
    @property
    def radius(self):       return self.diameter / 2
    @property
    def area(self):         return np.pi * self.radius**2
    @property
    def I_bending(self):    return np.pi * self.radius**4 / 4   # 2nd moment [m^4]
    @property
    def J_torsion(self):    return np.pi * self.radius**4 / 2   # polar moment [m^4]
    @property
    def segment_length(self): return self.total_length / self.n_nodes

WIRE = _Wire()

@dataclass(frozen=True)
class _Material:
    """
    Elastic and inertial material properties.
    """
    E   : float = 5.0e6    # Pa   — Young's modulus  
    nu  : float = 0.3      # —    — Poisson's ratio  
    rho : float = 400    # kg/m³ — density

    @property
    def G(self): return self.E / (2.0 * (1.0 + self.nu))   # shear modulus

MATERIAL = _Material()


@dataclass(frozen=True)
class _Solver:
    dt          : float = 0.01   # s   — time step
    iterations  : int   = 30     # —   — iterations per step
    I_eff_scale : float = 2.0   # —   — rotational inertia scale factor %
    zeta_bend   : float = 1   # —   — bending damping ratio

SOLVER = _Solver()

@dataclass(frozen=True)
class _Gripper:
    """
    Geometric and inertial properties of the tool attached to the flange.
 
    These values are used by transform_ft_wrench() to gravity-compensate
    the raw FT sensor readings and shift the reference point to the wire
    attachment.
 
    Coordinate convention: all offsets are along the flange +z axis.
    The UR10 flange z-axis points away from the robot toward the tool.
    """
    mass          : float = 1.09    # kg  — total gripper mass  (identified)
    attach_offset : float = 0.140   # m   — flange origin → wire attachment
    com_offset    : float = 0.057   # m   — flange origin → gripper CoM
    #                                         (identified from torque channels)
 
GRIPPER = _Gripper()
 


def create_sim() -> tuple[WireSimulator, WireParams]:
    """
    Build and return a fully configured (WireSimulator, WireParams) pair.

    All physical and solver parameters come from the module-level constants
    WIRE, MATERIAL and SOLVER above.

    Returns
    -------
    sim    : WireSimulator  — ready to call .estimate_wire_state()
    params : WireParams     — the parameter struct used to build it
                              (useful for reading N, L, dt, m_num, etc.)

    Example
    -------
    >>> from sim_setup import create_sim
    >>> sim, params = create_sim()
    >>> x = np.zeros(7 * params.N)   # flat state vector
    >>> v = np.zeros(6 * params.N)
    >>> x, v, CW, ee_wrench = sim.estimate_wire_state(
    ...     x, v, ee_pos, ee_quat, np.zeros(6), f_ext
    ... )
    """
    w  = WIRE
    m  = MATERIAL
    s  = SOLVER
    N  = w.n_nodes
    L  = w.segment_length

    alpha_stretch = np.array([
        L / (m.G * w.area),       # shear X
        L / (m.G * w.area),       # shear Y
        L / (m.E * w.area),       # axial Z
    ])
    alpha_bend = np.array([
        L / (m.E * w.I_bending),  # bend X
        L / (m.E * w.I_bending),  # bend Y
        L / (m.G * w.J_torsion),  # twist Z
    ])

    # ── Mass and inertia per node ──────────────────────────────────────────
    m_node = m.rho * w.area * L
    m_num  = np.ones(N) * m_node

    I_eff  = s.I_eff_scale * m_node * L**2
    I_num  = [np.eye(3) * I_eff for _ in range(N)]

    # ── Rayleigh constraint damping ────────────────────────────────────────
    omega_seg = np.sqrt(m.E * w.I_bending / (m_node * L**3))
    beta_bend = s.zeta_bend * (2.0 / omega_seg)

    # ── Assemble WireParams ────────────────────────────────────────────────
    params = WireParams(
        N             = N,
        dof           = 7,
        L             = L,
        alpha_stretch = alpha_stretch,
        alpha_bend    = alpha_bend,
        beta_stretch  = np.zeros(3),
        beta_bend     = np.full(3, beta_bend),
        alpha_ee_pos    = 0.0,    
        alpha_ee_orient = 0.0,
        beta_ee_pos     = 0.0,
        beta_ee_orient  = 0.0,
        dt          = s.dt,
        solver_iter = s.iterations,
        m_num       = m_num,
        I_num       = I_num,
    )

    return WireSimulator(params), params



def gravity_forces(params: WireParams) -> np.ndarray:
    """
    Build the external force vector with gravity applied to all free nodes.

    Parameters
    ----------
    params   : WireParams  — from create_sim()
    pin_node : int         — index of the EE-pinned node (gravity skipped there
                             since the EE constraint already holds it)

    Returns
    -------
    f_ext : (6*N,)  external wrench vector  [fx,fy,fz, tx,ty,tz] per node
    """
    f_ext = np.zeros(6 * params.N)
    for i in range(params.N):
        f_ext[i * 6 + 2] = params.m_num[i] * (-9.81)   # -Z gravity
    return f_ext


if __name__ == '__main__':
    sim, p = create_sim()

    print("=== Wire ===")
    print(f"  N = {p.N} nodes,  L = {p.L*1e3:.1f} mm/segment")
    print(f"  total length = {WIRE.total_length*1e3:.0f} mm")
    print(f"  diameter     = {WIRE.diameter*1e3:.0f} mm")

    print("\n=== Material ===")
    print(f"  E   = {MATERIAL.E:.2e} Pa")
    print(f"  G   = {MATERIAL.G:.2e} Pa")
    print(f"  rho = {MATERIAL.rho:.1f} kg/m³")

    print("\n=== Mass / inertia ===")
    print(f"  m_node = {p.m_num[0]*1e3:.2f} g")
    print(f"  m_total = {p.m_num.sum()*1e3:.1f} g")

    print("\n=== Compliance ===")
    print(f"  alpha_bend[0]    = {p.alpha_bend[0]:.4f} rad/(N·m)  [bend]")
    print(f"  alpha_stretch[2] = {p.alpha_stretch[2]:.2e} m/N       [axial]")

    print("\n=== Solver ===")
    print(f"  dt = {p.dt} s,  iterations = {p.solver_iter}")

    print("\n=== One-step smoke test ===")
    x = np.zeros(7 * p.N)
    for i in range(p.N):
        x[7*i+6] = 1.0   # unit quaternion w=1
    v = np.zeros(6 * p.N)
    ee_pos  = np.zeros(3)
    ee_quat = np.array([0., 0., 0., 1.])
    f_ext   = gravity_forces(p)
    x2, v2, CW, ee_w = sim.estimate_wire_state(x, v, ee_pos, ee_quat, np.zeros(6), f_ext)
    print(f"  ee_wrench = {ee_w.round(4)}")
    print("  OK")