import numpy as np
from quaternion_utils import quat_mul, quat_conj, quat_to_rotm

def solve_stretch_shear(
    p0: np.ndarray,
    inv_mass0: float,
    p1: np.ndarray,
    inv_mass1: float,
    q0: np.ndarray,
    inv_mass_q0: float,
    rest_length: float,
    lambda_loc: np.ndarray,
    alpha_vec: np.ndarray,
    dt: float,
    C_loc_n: np.ndarray = None,
    gamma: np.ndarray    = None,
):
    """
    XPBD stretch / shear constraint between two consecutive nodes.

    Constraint (local frame):
        C_loc = Rᵀ · [ (p₁ − p₀) / L  −  d₃ ]  =  0

    Parameters
    ----------
    p0, p1       : (3,)   world positions of node i-1 and node i
    inv_mass0/1  : scalar  inverse translational masses
    q0           : (4,)   orientation quaternion of node i-1  [x,y,z,w]
    inv_mass_q0  : scalar  inverse rotational mass of node i-1
    rest_length  : float   segment rest length L
    lambda_loc   : (3,)   accumulated Lagrange multipliers (local frame)
    alpha_vec    : (3,)   compliance  [shear_x, shear_y, axial_z]
    dt           : float   time step
    C_loc_n      : (3,)   constraint violation at start of step  (for damping)
                          pass None or zeros(3) for no damping
    gamma        : (3,)   γ = α·β / dt  per component  (for damping)
                          pass None or zeros(3) for no damping

    Returns
    -------
    corr0, corr1 : (3,)  position corrections for p0 and p1
    corrq0       : (4,)  quaternion correction for q0
    dlambda_loc  : (3,)  Lagrange multiplier increment
    """
    if C_loc_n is None:
        C_loc_n = np.zeros(3)
    if gamma is None:
        gamma = np.zeros(3)

    qx, qy, qz, qw = q0

    # Material director d3 = local z-axis in world frame
    d3 = np.array([
        2*(qx*qz + qw*qy),
        2*(qy*qz - qw*qx),
        qw**2 - qx**2 - qy**2 + qz**2,
    ])

    R = quat_to_rotm(q0)

    C_glob = (p1 - p0) / rest_length - d3
    C_loc  = R.T @ C_glob                     # constraint in local material frame

    # Isotropic effective inverse mass
    w = (inv_mass0 + inv_mass1) / rest_length + inv_mass_q0 * 4.0 * rest_length

    # XPBD update — Eq. (26)
    alpha_tilde = alpha_vec / dt**2
    numerator   = -C_loc - alpha_tilde * lambda_loc - gamma * (C_loc - C_loc_n)
    denominator = (1.0 + gamma) * w + alpha_tilde + 1e-9
    dlambda_loc = numerator / denominator

    gamma_loc  = -dlambda_loc                 # position-correction direction (local)
    gamma_glob = R @ gamma_loc                # rotate back to world frame

    corr0 =  inv_mass0 * gamma_glob
    corr1 = -inv_mass1 * gamma_glob

    # Quaternion correction for q0
    q_e3_bar = np.array([-qy, qx, -qw, qz])
    g_quat   = np.array([gamma_glob[0], gamma_glob[1], gamma_glob[2], 0.0])
    corrq0   = quat_mul(g_quat, q_e3_bar) * (2.0 * inv_mass_q0 * rest_length)

    return corr0, corr1, corrq0, dlambda_loc


def solve_bend_twist(
    q0: np.ndarray,
    inv_mass_q0: float,
    q1: np.ndarray,
    inv_mass_q1: float,
    rest_darboux: np.ndarray,
    lambda_loc: np.ndarray,
    alpha_vec: np.ndarray,
    dt: float,
    C_loc_n: np.ndarray = None,
    gamma: np.ndarray    = None,
):
    """
    XPBD bend / twist constraint between two consecutive orientation frames.

    The constraint drives the Darboux vector (relative rotation between
    adjacent frames) to match the rest configuration.

    Constraint (local frame):
        C_loc = xyz_part( quat_conj(q0) * q1 ± rest_darboux )  =  0

    Parameters
    ----------
    q0, q1        : (4,)   orientation quaternions  [x,y,z,w]
    inv_mass_q0/1 : scalar  inverse rotational masses
    rest_darboux  : (4,)   rest Darboux vector as a quaternion
    lambda_loc    : (3,)   accumulated Lagrange multipliers
    alpha_vec     : (3,)   compliance  [bend_x, bend_y, twist_z]
    dt            : float   time step
    C_loc_n       : (3,)   constraint violation at start of step  (for damping)
    gamma         : (3,)   γ = α·β / dt  per component  (for damping)

    Returns
    -------
    corrq0, corrq1 : (4,)  quaternion corrections
    dlambda_loc    : (3,)  Lagrange multiplier increment
    """
    if C_loc_n is None:
        C_loc_n = np.zeros(3)
    if gamma is None:
        gamma = np.zeros(3)

    omega = quat_mul(quat_conj(q0), q1)

    # Shorter-arc selection
    omega_plus  = omega + rest_darboux
    omega_minus = omega - rest_darboux
    if np.dot(omega_minus, omega_minus) > np.dot(omega_plus, omega_plus):
        omega = omega_plus
    else:
        omega = omega_minus

    C_loc = omega[:3]

    # XPBD update — Eq. (26)
    alpha_tilde = alpha_vec / dt**2
    numerator   = -C_loc - alpha_tilde * lambda_loc - gamma * (C_loc - C_loc_n)
    denominator = (1.0 + gamma) * (inv_mass_q0 + inv_mass_q1) + alpha_tilde + 1e-9
    dlambda_loc = numerator / denominator

    correction  = -dlambda_loc
    omega_corr  = np.array([correction[0], correction[1], correction[2], 0.0])

    corrq0 =  inv_mass_q0 * quat_mul(q1, omega_corr)
    corrq1 = -inv_mass_q1 * quat_mul(q0, omega_corr)

    return corrq0, corrq1, dlambda_loc


def solve_ee_position(
    p0: np.ndarray,
    inv_mass0: float,
    ee_pos: np.ndarray,
    lambda_loc: np.ndarray,
    alpha_ee: float,
    dt: float,
    C_n: np.ndarray = None,
    gamma: float     = 0.0,
):
    """
    XPBD position constraint pinning a node to a fixed EE position.

    Constraint:
        C = p0 − ee_pos  =  0


    Parameters
    ----------
    p0        : (3,)   current world position of the node
    inv_mass0 : scalar  inverse translational mass of the node
    ee_pos    : (3,)   target EE position (world frame)
    lambda_loc: (3,)   accumulated Lagrange multiplier
    alpha_ee  : float   compliance (0 = rigid pin)
    dt        : float   time step
    C_n       : (3,)   C at start of step for damping  (None → no damping)
    gamma     : float   γ = α·β / dt  (0 → no damping)

    Returns
    -------
    corr_p0 : (3,)  position correction for the node
    dlambda : (3,)  Lagrange multiplier increment
    """
    if C_n is None:
        C_n = np.zeros(3)

    C           = p0 - ee_pos
    alpha_tilde = alpha_ee / dt**2
    numerator   = -C - alpha_tilde * lambda_loc - gamma * (C - C_n)
    denominator = (1.0 + gamma) * inv_mass0 + alpha_tilde + 1e-9
    dlambda     = numerator / denominator
    corr_p0     = inv_mass0 * dlambda
    return corr_p0, dlambda


def solve_ee_orient(
    q0: np.ndarray,
    inv_mass_q0: float,
    ee_quat: np.ndarray,
    lambda_loc: np.ndarray,
    alpha_ee: float,
    dt: float,
    C_n: np.ndarray = None,
    gamma: float     = 0.0,
):
    """
    XPBD orientation constraint pinning a node's frame to a fixed EE quaternion.

    Constraint:
        C = xyz_part( quat_conj(ee_quat) * q0 )  =  0

    Parameters
    ----------
    q0          : (4,)   current orientation quaternion  [x,y,z,w]
    inv_mass_q0 : scalar  inverse rotational mass of the node
    ee_quat     : (4,)   target EE orientation quaternion  [x,y,z,w]
    lambda_loc  : (3,)   accumulated Lagrange multiplier
    alpha_ee    : float   compliance (0 = rigid pin)
    dt          : float   time step
    C_n         : (3,)   C at start of step for damping  (None → no damping)
    gamma       : float   γ = α·β / dt  (0 → no damping)

    Returns
    -------
    corr_q0 : (4,)  quaternion correction for the node
    dlambda : (3,)  Lagrange multiplier increment
    """
    if C_n is None:
        C_n = np.zeros(3)

    # Relative rotation from ee_quat to q0
    omega = quat_mul(quat_conj(ee_quat), q0)

    # Shorter-arc selection
    rest        = np.array([0.0, 0.0, 0.0, 1.0])
    omega_plus  = omega + rest
    omega_minus = omega - rest
    if np.dot(omega_minus, omega_minus) > np.dot(omega_plus, omega_plus):
        omega = omega_plus
    else:
        omega = omega_minus

    C           = omega[:3]
    alpha_tilde = alpha_ee / dt**2
    numerator   = -C - alpha_tilde * lambda_loc - gamma * (C - C_n)
    denominator = (1.0 + gamma) * inv_mass_q0 + alpha_tilde + 1e-9
    dlambda     = numerator / denominator

    correction  = -dlambda
    omega_corr  = np.array([correction[0], correction[1], correction[2], 0.0])
    
    corr_q0     = -inv_mass_q0 * quat_mul(ee_quat, omega_corr)

    return corr_q0, dlambda


def stretch_shear_violation(p0, p1, q0, rest_length):
    """Return the local-frame stretch/shear constraint violation for a segment."""
    qx, qy, qz, qw = q0
    d3 = np.array([
        2*(qx*qz + qw*qy),
        2*(qy*qz - qw*qx),
        qw**2 - qx**2 - qy**2 + qz**2,
    ])
    R      = quat_to_rotm(q0)
    C_glob = (p1 - p0) / rest_length - d3
    return R.T @ C_glob


def bend_twist_violation(q0, q1, rest_darboux):
    """Return the local-frame bend/twist constraint violation for a segment."""
    omega       = quat_mul(quat_conj(q0), q1)
    omega_plus  = omega + rest_darboux
    omega_minus = omega - rest_darboux
    if np.dot(omega_minus, omega_minus) > np.dot(omega_plus, omega_plus):
        return omega_plus[:3].copy()
    return omega_minus[:3].copy()


def ee_position_violation(p0, ee_pos):
    """Return the EE position constraint violation."""
    return p0 - ee_pos


def ee_orient_violation(q0, ee_quat):
    """Return the EE orientation constraint violation."""
    rest        = np.array([0.0, 0.0, 0.0, 1.0])
    omega       = quat_mul(quat_conj(ee_quat), q0)
    omega_plus  = omega + rest
    omega_minus = omega - rest
    if np.dot(omega_minus, omega_minus) > np.dot(omega_plus, omega_plus):
        return omega_plus[:3].copy()
    return omega_minus[:3].copy()
