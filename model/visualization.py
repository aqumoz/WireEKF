import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)

from quaternion_utils import quat_to_rotm


def plot_rod_frames(
    ax: plt.Axes,
    x_num: np.ndarray,
    dof: int,
    N: int,
    time: float,
    frame_scale: float = 0.05,
) -> None:
    """
    Draw the Cosserat rod into a matplotlib 3-D axes object.

    Parameters
    ----------
    ax           : matplotlib 3D axes (must be created with projection='3d')
    x_num        : (dof*N,)  flat state vector
    dof          : degrees of freedom per node  (= 7)
    N            : number of nodes
    time         : current simulation time  [s]  (used for the title)
    frame_scale  : length of each drawn frame axis  [m]
    """
    ax.cla()

    # ----------------------------------------------------------------
    # Extract node positions and orientations
    # ----------------------------------------------------------------
    positions = np.zeros((N, 3))
    quats     = np.zeros((N, 4))
    for i in range(N):
        positions[i] = x_num[i * dof:i * dof + 3]
        quats[i]     = x_num[i * dof + 3:i * dof + 7]

    # ----------------------------------------------------------------
    # Draw wire backbone
    # ----------------------------------------------------------------
    ax.plot(
        positions[:, 0], positions[:, 1], positions[:, 2],
        'k-o', linewidth=2, markersize=4, label='Wire',
    )

    # ----------------------------------------------------------------
    # Draw coordinate frames at each node  (x=red, y=green, z=blue)
    # ----------------------------------------------------------------
    axis_colors = ['r', 'g', 'b']
    for i in range(N):
        p = positions[i]
        R = quat_to_rotm(quats[i])
        for j, color in enumerate(axis_colors):
            d = R[:, j] * frame_scale
            ax.quiver(
                p[0], p[1], p[2],
                d[0], d[1], d[2],
                color=color, linewidth=1.2, arrow_length_ratio=0.3,
            )

    # ----------------------------------------------------------------
    # Axis limits (auto-scaled around the wire)
    # ----------------------------------------------------------------
    center = positions.mean(axis=0)
    spread = np.ptp(positions, axis=0).max()
    half   = max(spread / 2 + frame_scale * 2, 0.15)

    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)

    # ----------------------------------------------------------------
    # Labels and title
    # ----------------------------------------------------------------
    ax.set_xlabel('X  [m]')
    ax.set_ylabel('Y  [m]')
    ax.set_zlabel('Z  [m]')
    ax.set_title(f'XPBD Wire Simulation  —  t = {time:.3f} s')

    try:
        ax.set_box_aspect([1, 1, 1])   # equal aspect ratio (matplotlib ≥ 3.3)
    except AttributeError:
        pass
