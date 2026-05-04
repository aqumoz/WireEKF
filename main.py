import argparse
import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from WireEKF import WireEKF
from SimModelAndTrackDLO import SimModelAndTrackDLO


# --- Utilities ---

class History:
    """Records position history for all nodes over time."""
    
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


class InteractiveSimulation:
    """Interactive EKF simulation driven by keyboard input."""
    
    def __init__(
        self,
        n_nodes: int = 10,
        sensor_var: float = 0.001,
        dataset_model: str | None = None,
        dataset_trackdlo: str | None = None,
        print_table: bool = False,
    ):
        self.n_nodes = n_nodes
        self.sensor_var = sensor_var
        self.print_table = print_table

        # Initialize simulator (loads datasets if provided)
        self.sim = SimModelAndTrackDLO(n_nodes, dataset_model, dataset_trackdlo)
        
        # Initialize filter with simulator callbacks
        self.filter = WireEKF(
            n_nodes=n_nodes,
            initial_nodes=np.zeros((3 * n_nodes, 1)),
            get_model_nodes=self.sim.get_model_nodes if self.sim.model_dataset is not None else None,
            get_trackdlo_nodes=self.sim.get_trackdlo_nodes if self.sim.trackdlo_dataset is not None else None,
        )
        
        self.meas_h = History(n_nodes)
        self.ekf_h = History(n_nodes)
        self.mod_h = History(n_nodes)
        
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        self.reset_sim()
        print('--- Interactive Mode ---')
        print('1-6: Steps [1, 5, 10, 100, 500, 1000] | R: Reset Simulation')

    def reset_sim(self):
        """Reset filter and simulator."""
        self.curr_t = 0.0
        self.step_count = 0
        self.meas_h.reset()
        self.ekf_h.reset()
        self.mod_h.reset()
        self.filter.reset()
        self.sim.reset()
        self.update_plot()
        print('Simulation Reset.')

    def run_steps(self, num_steps: int):
        """Run num_steps of the filter."""
        period = 2 * np.pi
        step_mean = period / 500
        
        for _ in range(num_steps):
            # dt = np.random.uniform(step_mean - 0.005, step_mean + 0.005)
            self.curr_t = None  # Time is managed by filter callbacks, so we set it to None here
            self.step_count += 1
            
            if self.sim.model_dataset is not None:
                model_nodes = self.sim.get_model_nodes(self.curr_t)
            else:
                # TODO: When the model callback is done, insert the callback here!
                raise ValueError("No model callback was supplied for model nodes.")

            if self.sim.trackdlo_dataset is not None:
                measurement = self.sim.get_trackdlo_nodes(self.curr_t)
            else:
                # TODO: When TrackDLO have been setup, insert the callback here!
                raise ValueError("No TrackDLO callback was supplied for measurements.")
            
            # Run filter steps
            self.filter.predict(self.curr_t, model_nodes=model_nodes)
            estimate = self.filter.update(self.curr_t, trackdlo_nodes=measurement)


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

            # Log state for plotting
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

    def update_plot(self):
        """Update 3D visualization with wires and trajectories."""
        elev, azim = self.ax.elev, self.ax.azim
        self.ax.cla()
        
        if self.step_count > 0:
            # Plot for EKF
            self._plot_history(self.ekf_h, wire_color='blue', traj_color='blue', label_prefix='EKF')
            
            # Plot for Model
            self._plot_history(self.mod_h, wire_color='green', traj_color='darkgreen', label_prefix='Model')
            
            # Plot for Measurement
            self._plot_history(self.meas_h, wire_color='red', traj_color='darkred', label_prefix='Measurement')
            
            self.ax.legend(loc='upper left')
        
        self.ax.set_title(f'Interactive EKF | Total Steps: {self.step_count}')
        self.ax.view_init(elev=elev, azim=azim)
        self.fig.canvas.draw_idle()
    
    def _plot_history(self, history: History, wire_color: str, traj_color: str, label_prefix: str):
        """Plot wires at each step and trajectories for each node."""
        # Plot wires at each step (every 10th step to avoid clutter)
        for step in range(0, len(history.positions), max(1, len(history.positions) // 10)):
            pos = history.positions[step]  # (n_nodes, 3)
            self.ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], color=wire_color, alpha=0.3, lw=1)
        
        # Plot trajectories for each node
        for node_idx in range(0, self.n_nodes):
            x, y, z = history.get_node_trajectory(node_idx)
            self.ax.plot(x, y, z, color=traj_color, alpha=0.8, lw=1, label=f'{label_prefix}' if node_idx == 0 else "")


def main():
    parser = argparse.ArgumentParser(description='Run interactive EKF simulation.')
    parser.add_argument('--dataset-model', type=str, default=None, help='CSV path for model dataset')
    parser.add_argument('--dataset-trackdlo', type=str, default=None, help='CSV path for TrackDLO dataset')
    parser.add_argument('--sensor-var', type=float, default=0.001, help='Sensor noise variance')
    parser.add_argument('--n-nodes', type=int, default=10, help='Number of wire nodes')
    parser.add_argument('--print-table', type=bool, default=False, help='Print estimation table at each step')
    args = parser.parse_args()
    
    sim = InteractiveSimulation(
        n_nodes=args.n_nodes,
        sensor_var=args.sensor_var,
        dataset_model=args.dataset_model,
        dataset_trackdlo=args.dataset_trackdlo,
        print_table=args.print_table,
    )
    plt.show()


if __name__ == '__main__':
    main()
