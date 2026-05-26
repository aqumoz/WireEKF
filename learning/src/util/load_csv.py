import pandas as pd
import numpy as np
import torch
from pathlib import Path



def load_csv(file_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    # Load CSV
    df_ft = pd.read_csv(Path(file_path) / "robot_b_force_torque.csv")
    df_ft = df_ft.drop(columns=["time_ns"])
    df_tcp = pd.read_csv(Path(file_path) / "robot_b_tcp_pose.csv")
    df_tcp = df_tcp.drop(columns=["time_ns"])

    # df_q = pd.read_csv(Path(file_path) / "robot_b_q.csv")
    # df_q = df_q.drop(columns=["time_ns"])
    # df_qd = pd.read_csv(Path(file_path) / "robot_b_qd.csv")
    # df_qd = df_qd.drop(columns=["time_ns"])

    df_track_dlo = pd.read_csv(Path(file_path) / "trackdlo_nodes.csv")
    df_track_dlo = df_track_dlo.drop(columns=["time_ns"])


    df_model = pd.read_csv(Path(file_path) / "x_init_history.csv")

    df_ft["time_s"] = pd.to_numeric(df_ft["time_s"], errors="coerce")
    df_tcp["time_s"] = pd.to_numeric(df_tcp["time_s"], errors="coerce")
    df_track_dlo["time_s"] = pd.to_numeric(df_track_dlo["time_s"], errors="coerce")
    df_model["time_s"] = pd.to_numeric(df_model["time_s"], errors="coerce")

    df_merged = pd.merge_asof(
        df_track_dlo,
        df_ft,
        on="time_s",
        direction="nearest",
        tolerance=0.015  # type: ignore
    ).dropna()

    df_merged = pd.merge_asof(
        df_merged,
        df_tcp,
        on="time_s",
        direction="nearest",
        tolerance=0.015  # type: ignore
    ).dropna()

    # df_merged = pd.merge_asof(
    #     df_merged,
    #     df_q,
    #     on="time_s",
    #     direction="nearest",
    #     tolerance=0.015  # type: ignore
    # ).dropna()

    # df_merged = pd.merge_asof(
    #     df_merged,
    #     df_qd,
    #     on="time_s",
    #     direction="nearest",
    #     tolerance=0.015  # type: ignore
    # ).dropna()


    df_merged["time_s"] = df_merged["time_s"] - df_merged["time_s"].min()

    df_merged = pd.merge_asof(
        df_merged,
        df_model,
        on="time_s",
        direction="nearest",
        tolerance=0.015  # type: ignore
    ).dropna()


    cols = [c for df in [df_ft, df_tcp] for c in df.columns if c not in ("time_ns", "time_s")]
    x_inputs = df_merged[cols].values

    column_names = []

    num_nodes = 20

    for node in range(num_nodes):

        # Position names
        column_names.append(f"node_model_{node}_x")
        column_names.append(f"node_model_{node}_y")
        column_names.append(f"node_model_{node}_z")

        # Quaternion names
        # column_names.append(f"node_model_{node}_qx")
        # column_names.append(f"node_model_{node}_qy")
        # column_names.append(f"node_model_{node}_qz")
        # column_names.append(f"node_model_{node}_qw")


    y_model = df_merged[column_names].values

    column_names = []

    num_nodes = 20

    for node in range(num_nodes):

        # Position names
        column_names.append(f"node_{node}_x")
        column_names.append(f"node_{node}_y")
        column_names.append(f"node_{node}_z")

        # Quaternion names
        # column_names.append(f"node_{node}_qx")
        # column_names.append(f"node_{node}_qy")
        # column_names.append(f"node_{node}_qz")
        # column_names.append(f"node_{node}_qw")

    y_data = df_merged[column_names].values


    residuals = y_data - y_model

    x = torch.tensor(x_inputs, dtype=torch.float32)
    y = torch.tensor(residuals, dtype=torch.float32)
    return x, y

def trim_to_transient(df_xpbd, df_simulink, tip_node_idx, 
                      window=80, threshold=0.5e-1):
    """
    Trim both dataframes to only include the transient phase,
    detected by when the XPBD tip node velocity drops below threshold.
    
    Parameters
    ----------
    tip_node_idx : int   — index of the tip node (usually N-1)
    window       : int   — rolling window size for smoothing velocity
    threshold    : float — m/s below which we consider steady state
    """
    # Compute tip node displacement between timesteps
    tip_x = df_xpbd[f'node_{tip_node_idx}_x']
    tip_y = df_xpbd[f'node_{tip_node_idx}_y']
    tip_z = df_xpbd[f'node_{tip_node_idx}_z']

    dx = tip_x.diff().fillna(0)
    dy = tip_y.diff().fillna(0)
    dz = tip_z.diff().fillna(0)

    tip_speed = np.sqrt(dx**2 + dy**2 + dz**2)

    # Smooth it to avoid triggering on noise
    tip_speed_smooth = (tip_speed.rolling(window=window, center=True).mean().bfill().ffill())
    # print(f"  Tip speed stats:")
    # print(f"    max:    {tip_speed_smooth.max():.6f} m/step")
    # print(f"    min:    {tip_speed_smooth.min():.6f} m/step")
    # print(f"    median: {tip_speed_smooth.median():.6f} m/step")
    # print(f"    final 50 steps mean: {tip_speed_smooth.iloc[-50:].mean():.6f} m/step")
    # Find first index where it stays below threshold
    below = tip_speed_smooth < threshold
    # Require it to stay below for `window` consecutive steps
    steady_start = None
    for i in range(len(below) - window):
        if below.iloc[i:i+window].all():
            steady_start = i
            break

    if steady_start is None:
        print("  Warning: steady state never detected — keeping full trajectory")
        return df_xpbd, df_simulink

    print(f"  Steady state detected at index {steady_start}, "
          f"t = {df_xpbd['time'].iloc[steady_start]:.2f} s — trimming here")

    df_xpbd_trim     = df_xpbd.iloc[:steady_start].reset_index(drop=True)
    df_simulink_trim = df_simulink.iloc[:steady_start].reset_index(drop=True)

    return df_xpbd_trim, df_simulink_trim


def load_simulated_data(file_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    df_xpbd = pd.read_csv(Path(file_path) / "xpbd.csv")
    df_simulink = pd.read_csv(Path(file_path) / "simulink.csv")

    # df_xpbd, df_simulink = trim_to_transient(
    #     df_xpbd, df_simulink,
    #     tip_node_idx = 19,
    #     window    = 50,       # tune this — ~0.5s worth of steps at dt=0.01
    #     threshold = 1e-4      # tune this — depends on your wire's scale
    # )

    df_xpbd.columns = [col + "_xpbd" if col != "time" else col for col in df_xpbd.columns]


    df_merged = pd.merge_asof(
        df_xpbd,
        df_simulink,
        on="time",
        direction="nearest",
        tolerance=0.015  # type: ignore
    ).dropna()
    cols = ["time", "ee_fx_xpbd", "ee_fy_xpbd", "ee_fz_xpbd", "ee_tx_xpbd", "ee_ty_xpbd", "ee_tz_xpbd"]
    #cols = ["ft_x", "ft_y", "ft_z", "torque_x", "torque_y", "torque_z"]
    num_nodes = 20
    # for node in range(num_nodes):

    #     # Position names
    #     cols.append(f"node_{node}_vx_xpbd")
    #     cols.append(f"node_{node}_vy_xpbd")
    #     cols.append(f"node_{node}_vz_xpbd")

    x_inputs = df_merged[cols].values

    column_names = []

    for node in range(num_nodes):

        # Position names
        column_names.append(f"node_{node}_x")
        column_names.append(f"node_{node}_y")
        column_names.append(f"node_{node}_z")


    y_simulink = df_merged[column_names].values

    column_names = []

    for node in range(num_nodes):

        # Position names
        column_names.append(f"node_{node}_x_xpbd")
        column_names.append(f"node_{node}_y_xpbd")
        column_names.append(f"node_{node}_z_xpbd")

    y_xpbd = df_merged[column_names].values


    residuals =  y_xpbd - y_simulink

    x = torch.tensor(x_inputs, dtype=torch.float32)
    y = torch.tensor(residuals, dtype=torch.float32)

    return x, y
