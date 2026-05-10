import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from sim_setup import create_sim, gravity_forces, WIRE, SOLVER
from visualization import plot_rod_frames


# ============================================================
# 1.  Build simulator from calibrated parameters
# ============================================================
sim, params = create_sim()


# ============================================================
# 2.  Initial conditions
# ============================================================
base_pos       = np.array([0.0, 0.0, 0.0])
base_euler_xyz = np.array([0.0, np.pi / 2, 0.0])   # cable z-axis → world X

# 1-degree Y-bend at every joint
joint_angles        = np.zeros((3, params.N - 1))
joint_angles[1, :]  = np.deg2rad(1.0)

x_init = sim.joint_to_world_init(base_pos, base_euler_xyz, joint_angles)
v_init = np.zeros(6 * params.N)


# ============================================================
# 3.  End-effector pose  (rigid pin at node 0)
# ============================================================
ee_pos  = np.array([0.0, 0.0, 0.0])
ee_quat = np.array([0.0, np.sin(np.pi / 4), 0.0, np.cos(np.pi / 4)])


# ============================================================
# 4.  External forces
# ============================================================
f_ext = gravity_forces(params)

# Tip load
tip_force = np.zeros(6)
f_ext[-6:] += tip_force

# ============================================================
# 5.  Simulation loop
# ============================================================
sim_time = 10.0
steps    = np.arange(0.0, sim_time + SOLVER.dt, SOLVER.dt)

fig = plt.figure(figsize=(9, 7))
ax  = fig.add_subplot(111, projection='3d')
plt.ion()
plt.show()

for time in steps:
    x_init, v_init, _cw, ee_wrench = sim.estimate_wire_state(
        x_init, v_init, ee_pos, ee_quat, np.zeros(6), f_ext,
    )

    plot_rod_frames(ax, x_init, params.dof, params.N, time, frame_scale=0.015)
    plt.pause(0.001)

plt.ioff()
plt.show()