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


def load_simulated_data(file_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    df_xpbd = pd.read_csv(Path(file_path) / "xpbd_simulation.csv")
    df_simulink = pd.read_csv(Path(file_path) / "simulink_simulation.csv")

    df_xpbd.columns = [col + "_xpbd" if col != "time" else col for col in df_xpbd.columns]


    df_merged = pd.merge_asof(
        df_xpbd,
        df_simulink,
        on="time",
        direction="nearest",
        tolerance=0.015  # type: ignore
    ).dropna()
    cols = ["ee_fx_xpbd", "ee_fy_xpbd", "ee_fz_xpbd", "ee_tx_xpbd", "ee_ty_xpbd", "ee_tz_xpbd"]
    #cols = ["ft_x", "ft_y", "ft_z", "torque_x", "torque_y", "torque_z"]
    x_inputs = df_merged[cols].values


    column_names = []

    num_nodes = 20

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
