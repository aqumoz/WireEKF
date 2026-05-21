import argparse
import numpy as np
import matplotlib

from DataSetLoader import DataSetLoader
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from WireEKF import WireEKF
from ModelWrapper import Model
from ColumnMap import Data, ColumnMap

class History:
    """
    Records position history for all nodes over time.
    For use when plotting steps.
    """
    
    def __init__(self, n_nodes: int):
        self.n_nodes = n_nodes
        self.reset()

    def reset(self):
        self.positions = []  # list of (n_nodes, 3) arrays
        self.t = []

    def log(self, state_vec: np.ndarray, t: float | None = None):
        """Append full node positions from state vector (3*n, 1)."""
        state_vec = np.asarray(state_vec).reshape((self.n_nodes, 3))
        self.positions.append(state_vec.copy())
        if t is not None:
            self.t.append(t)

    def get_node_trajectory(self, node_idx: int) -> tuple[list[float], list[float], list[float]]:
        """Get x, y, z lists for a specific node's trajectory."""
        x = [pos[node_idx, 0] for pos in self.positions]
        y = [pos[node_idx, 1] for pos in self.positions]
        z = [pos[node_idx, 2] for pos in self.positions]
        return x, y, z

    def get_all_positions_at_step(self, step: int) -> np.ndarray:
        """Get (n_nodes, 3) positions at a specific step."""
        return self.positions[step]


class InteractiveViewer:
    """Interactive EKF simulation driven by keyboard input."""
    
    def __init__(
        self,
        dataset_trackdlo: str,
        n_nodes: int = 10,
        print_table: bool = False,
        q_diag: float = 0.01,
        r_diag: float = 0.05,
    ):
        self.n_nodes = n_nodes
        self.print_table = print_table

        # Initialize simulator (loads datasets if provided)
        self.trackDLO = DataSetLoader(n_nodes, dataset_trackdlo)
        self.model = Model()

        # Initialize filter with simulator callbacks
        self.filter = WireEKF(
            initial_state = self.trackDLO.get_nodes(),
            n_nodes = n_nodes,
            get_model_state = self.model.predict,
            q_diag = q_diag,
            r_diag = r_diag
        )
        
        self.meas_h = History(n_nodes)
        self.ekf_h = History(n_nodes)
        self.mod_h = History(n_nodes)

        # Visibility toggles
        self.show_measurement = True
        self.show_model = True
        self.show_ekf = True

        # Animation state
        self.animation_mode = False
        self.animation = None
        self.current_frame = 0

        # Axis constraint
        self.constrain_axes = False

        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        self.reset_sim()
        print('--- Interactive Mode ---')
        print('1-6: Steps [1, 5, 10, 100, 500, 1000] | R: Reset Simulation')
        print('I: Toggle Measurement | O: Toggle Model | P: Toggle EKF')
        print('A: Start/Stop Animation | C: Constrain Axes (0-0.8)')

    def start_animation(self):
        """Start animation cycling through recorded frames."""
        if self.step_count == 0:
            print('Loading full dataset...')
            # Load all remaining data
            while True:
                try:
                    measurement = self.trackDLO.get_nodes()

                    self.step_count += 1
                    self.curr_t = measurement.time

                    self.filter.predict(self.curr_t)
                    self.filter.update(measurement)

                    model_nodes = self.filter.predicted_data.to_x().reshape((self.n_nodes, 3))
                    measurement_data = measurement.to_x().reshape((self.n_nodes, 3))
                    estimate = self.filter.current_state.to_x().reshape((self.n_nodes, 3))

                    self.mod_h.log(model_nodes)
                    self.meas_h.log(measurement_data)
                    self.ekf_h.log(estimate)
                except IndexError:
                    break

            print(f'Loaded {self.step_count} frames.')

        if self.step_count == 0:
            print('No data to animate.')
            return

        self.animation_mode = True
        self.current_frame = 0
        print(f'Animation started. Playing {self.step_count} frames...')
        self.animation = FuncAnimation(
            self.fig, self._animate_frame, frames=self.step_count,
            interval=50, repeat=True, blit=False
        )
        self.fig.canvas.draw_idle()

    def stop_animation(self):
        """Stop the animation."""
        if self.animation is not None:
            self.animation.event_source.stop()
            self.animation = None
        self.animation_mode = False
        print('Animation stopped.')
        self.update_plot()

    def _animate_frame(self, frame_idx):
        """Update plot for a single animation frame."""
        self.ax.cla()
        self.current_frame = frame_idx

        if frame_idx < len(self.ekf_h.positions):
            if self.show_ekf:
                pos = self.ekf_h.positions[frame_idx]
                self.ax.plot(pos[:, 0], pos[:, 1], pos[:, 2],
                            color='blue', lw=2, linestyle='-', label='EKF')

            if self.show_model:
                pos = self.mod_h.positions[frame_idx]
                self.ax.plot(pos[:, 0], pos[:, 1], pos[:, 2],
                            color='green', lw=2, linestyle='-', label='Model')

            if self.show_measurement:
                pos = self.meas_h.positions[frame_idx]
                self.ax.plot(pos[:, 0], pos[:, 1], pos[:, 2],
                            color='red', lw=2, linestyle='-', label='Measurement')

            self.ax.legend(loc='upper left')

        if self.constrain_axes:
            self.ax.set_xlim(0.0, 0.8)
            self.ax.set_ylim(0, -0.4)
            self.ax.set_zlim(-0.4, 0.4)

        self.ax.set_title(f'Animation | Frame: {frame_idx}/{self.step_count - 1}')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        return self.ax,

    def reset_sim(self):
        """Reset filter and simulator."""
        if self.animation_mode:
            self.stop_animation()
        self.curr_t = 0.0
        self.step_count = 0
        self.meas_h.reset()
        self.ekf_h.reset()
        self.mod_h.reset()
        # self.filter.reset()
        self.trackDLO.reset()
        self.update_plot()
        print('Simulation Reset.')

    def run_steps(self, num_steps: int):
        """Run num_steps of the filter."""
        for _ in range(num_steps):
            # dt = np.random.uniform(step_mean - 0.005, step_mean + 0.005)
            
            measurement = self.trackDLO.get_nodes()
            
            self.step_count += 1
            self.curr_t = measurement.time

            # Run filter steps
            self.filter.predict(self.curr_t)
            self.filter.update(measurement)


            model_nodes = self.filter.predicted_data.to_x().reshape((self.n_nodes, 3))
            measurement = measurement.to_x().reshape((self.n_nodes, 3))
            estimate = self.filter.current_state.to_x().reshape((self.n_nodes, 3))

            # Print step info
            if self.print_table:
                model_matrix = model_nodes.reshape((self.n_nodes, 3))
                measurement_matrix = measurement.reshape((self.n_nodes, 3))
                estimate_matrix = estimate.reshape((self.n_nodes, 3))

                print("\nnode | model                            | measurement                      | estimate")
                print("     |      x          y          z     |      x          y          z     |      x          y          z")
                print("-----+----------------------------------+----------------------------------+----------------------------------")
                for node_idx in range(self.n_nodes):
                    mx, my, mz = model_matrix[node_idx]
                    ox, oy, oz = measurement_matrix[node_idx]
                    ex, ey, ez = estimate_matrix[node_idx]
                    print(
                        f"{node_idx + 1:4d} | "
                        f"{mx:10.4f} {my:10.4f} {mz:10.4f} | "
                        f"{ox:10.4f} {oy:10.4f} {oz:10.4f} | "
                        f"{ex:10.4f} {ey:10.4f} {ez:10.4f}"
                    )

            # # Log state for plotting
            self.mod_h.log(model_nodes)
            self.meas_h.log(measurement)
            self.ekf_h.log(estimate)
        
        self.update_plot()

    def on_key(self, event):
        """Handle keyboard input."""
        key_map = {'1': 1, '2': 5, '3': 10, '4': 100, '5': 500, '6': 1000}
        if event.key in key_map:
            self.run_steps(key_map[event.key])
        elif event.key.lower() == 'r':
            self.reset_sim()
        elif event.key.lower() == 'i':
            self.show_measurement = not self.show_measurement
            print(f'Measurement: {"ON" if self.show_measurement else "OFF"}')
            self.update_plot()
        elif event.key.lower() == 'o':
            self.show_model = not self.show_model
            print(f'Model: {"ON" if self.show_model else "OFF"}')
            self.update_plot()
        elif event.key.lower() == 'p':
            self.show_ekf = not self.show_ekf
            print(f'EKF: {"ON" if self.show_ekf else "OFF"}')
            self.update_plot()
        elif event.key.lower() == 'a':
            if self.animation_mode:
                self.stop_animation()
            else:
                self.start_animation()
        elif event.key.lower() == 'c':
            self.constrain_axes = not self.constrain_axes
            print(f'Axis constraint: {"ON" if self.constrain_axes else "OFF"}')
            if self.constrain_axes:
                self.ax.view_init(elev=-60, azim=-60)
            self.update_plot()

    def update_plot(self):
        """Update 3D visualization with wires and trajectories."""
        elev, azim, roll = self.ax.elev, self.ax.azim, self.ax.roll
        self.ax.cla()

        if self.step_count > 0:
            # Plot for EKF
            if self.show_ekf:
                self._plot_history(self.ekf_h, wire_color='blue', wire_style='-', traj_color='blue', traj_style='-', label_prefix='EKF')

            # Plot for Model
            if self.show_model:
                self._plot_history(self.mod_h, wire_color='green', wire_style='-', traj_color='darkgreen', traj_style='-', label_prefix='Model')

            # Plot for Measurement
            if self.show_measurement:
                self._plot_history(self.meas_h, wire_color='red', wire_style='-', traj_color='darkred', traj_style='-', label_prefix='Measurement')

            self.ax.legend(loc='upper left')

        if self.constrain_axes:
            self.ax.set_xlim(0.0, 0.8)
            self.ax.set_ylim(0, -0.4)
            self.ax.set_zlim(-0.4, 0.4)

        self.ax.set_title(f'Wire | Total Steps: {self.step_count}')
        self.ax.view_init(elev=elev, azim=azim, roll=roll)
        self.fig.canvas.draw_idle()
    
    def _plot_history(self, history: History, wire_color: str, wire_style : str, traj_color: str, traj_style: str, label_prefix: str):
        """Plot wires at each step and trajectories for each node."""
        # Plot wires at each step (every 10th step to avoid clutter)
        for step in range(0, len(history.positions), max(1, len(history.positions) // 10)):
            pos = history.positions[step]  # (n_nodes, 3)
            self.ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], color=wire_color, alpha=0.3, lw=1, linestyle=wire_style)
        
        # Plot trajectories for each node
        skip_nodes = 2 # if set to e.g. 2, only the trajectory of every third node is drawn.
        for node_idx in range(0, self.n_nodes, 1+skip_nodes):
            x, y, z = history.get_node_trajectory(node_idx)
            self.ax.plot(x, y, z, color=traj_color, alpha=0.8, lw=1, linestyle=traj_style, label=f'{label_prefix}' if node_idx == 0 else "")

    # Destructor to ensure file is closed if viewer is garbage collected
    def __del__(self):
        self.trackDLO.close()


def main():
    parser = argparse.ArgumentParser(description='Run interactive EKF simulation.')
    parser.add_argument('--dataset-trackdlo', type=str, default=None, help='CSV path for TrackDLO dataset')
    parser.add_argument('--n-nodes', type=int, default=10, help='Number of wire nodes')
    parser.add_argument('--print-table', type=bool, default=False, help='Print estimation table at each step')
    parser.add_argument('--q-diag', type=float, default=0.01, help='Q-diagonal values')
    parser.add_argument('--r-diag', type=float, default=0.05, help='R-diagonal values')
    args = parser.parse_args()
    
    sim = InteractiveViewer(
        dataset_trackdlo=args.dataset_trackdlo,
        n_nodes=args.n_nodes,
        print_table=args.print_table,
        q_diag = args.q_diag,
        r_diag = args.r_diag
    )
    plt.show()


if __name__ == '__main__':
    main()
