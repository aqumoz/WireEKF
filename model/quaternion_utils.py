import numpy as np


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------

def quat_conj(q: np.ndarray) -> np.ndarray:
    """
    Quaternion conjugate.

    Parameters
    ----------
    q : [x, y, z, w]

    Returns
    -------
    qc : [-x, -y, -z, w]
    """
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Quaternion product  q1 * q2.

    Parameters
    ----------
    q1, q2 : [x, y, z, w]

    Returns
    -------
    q_out : [x, y, z, w]
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ])


def quat_to_rotm(q: np.ndarray) -> np.ndarray:
    """
    3×3 rotation matrix from a unit quaternion.

    Parameters
    ----------
    q : [x, y, z, w]

    Returns
    -------
    R : (3, 3) ndarray
    """
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),      2*(x*z + w*y)      ],
        [2*(x*y + w*z),        1 - 2*(x*x + z*z),   2*(y*z - w*x)      ],
        [2*(x*z - w*y),        2*(y*z + w*x),       1 - 2*(x*x + y*y)  ],
    ])


# ---------------------------------------------------------------------------
# Euler → quaternion  (used in joint_to_world_init)
# ---------------------------------------------------------------------------

def euler_xyz_to_quat(euler_xyz: np.ndarray) -> np.ndarray:
    """
    Intrinsic X→Y→Z Euler angles (radians) to quaternion [x, y, z, w].

    Equivalent to the body-frame rotation  R_x(rx) · R_y(ry) · R_z(rz),
    which is the same as the extrinsic ZYX convention.

    Parameters
    ----------
    euler_xyz : [rx, ry, rz]  in radians

    Returns
    -------
    q : [x, y, z, w]
    """
    rx, ry, rz = euler_xyz
    cx, sx = np.cos(rx / 2), np.sin(rx / 2)
    cy, sy = np.cos(ry / 2), np.sin(ry / 2)
    cz, sz = np.cos(rz / 2), np.sin(rz / 2)

    w =  cx*cy*cz + sx*sy*sz
    x =  sx*cy*cz - cx*sy*sz
    y =  cx*sy*cz + sx*cy*sz
    z =  cx*cy*sz - sx*sy*cz

    return np.array([x, y, z, w])
