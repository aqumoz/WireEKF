import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Load dataframe
# --------------------------------------------------
# Example:
# df = pd.read_csv("data.csv")

# Assume:
# df["time"] = timestamps
# df["signal"] = oscillating signal

df_simulink = pd.read_csv( "results_test/bx0.43_by2.28_bz0.24_fx-0.008_fy-0.001_fz0.006/simulink_simulation.csv")
df_xpbd = pd.read_csv( "results_test/bx0.43_by2.28_bz0.24_fx-0.008_fy-0.001_fz0.006/xpbd_simulation.csv")
df_xpbd.columns = [col + "_xpbd" if col != "time" else col for col in df_xpbd.columns]


df_merged = pd.merge_asof(
    df_xpbd,
    df_simulink,
    on="time",
    direction="nearest",
    tolerance=0.015  # type: ignore
).dropna()



signal_col = ["node_19_x", "node_19_y", "node_19_z", "node_19_x_xpbd", "node_19_y_xpbd", "node_19_z_xpbd"]

# --------------------------------------------------
# Parameters
# --------------------------------------------------
window_size = 200          # samples in rolling window
std_threshold = 0.01       # threshold for rolling std
amp_threshold = 0.05       # threshold for peak-to-peak amplitude
consecutive_windows = 5    # require stable behavior for several windows

# --------------------------------------------------
# Rolling statistics
# --------------------------------------------------
rolling_std = (
    df_merged[signal_col]
    .rolling(window_size)
    .std()
)

rolling_amp = (
    df_merged[signal_col]
    .rolling(window_size)
    .apply(lambda x: np.max(x) - np.min(x), raw=True)
)

# --------------------------------------------------
# Detect steady state
# --------------------------------------------------
steady_mask = (
    (rolling_std < std_threshold) &
    (rolling_amp < amp_threshold)
)

# Require several consecutive windows
steady_count = (
    steady_mask
    .rolling(consecutive_windows)
    .sum()
)

steady_indices = np.where(
    steady_count >= consecutive_windows
)[0]

if len(steady_indices) == 0:
    print("No steady state detected.")
    df_trimmed = df_merged.copy()

else:
    steady_start_idx = steady_indices[0]

    print(f"Steady state detected at index: {steady_start_idx}")

    # Keep only data BEFORE steady state
    df_trimmed = df_merged.iloc[:steady_start_idx]

# --------------------------------------------------
# Plot result
# --------------------------------------------------
plt.figure(figsize=(12,5))
#print(df_merged[signal_col].head())
#print(df_merged[signal_col].shape)
print(df_merged[signal_col[3:6]].values.shape)
print(df_merged[signal_col[0:3]].values.shape)
a = df_merged[signal_col[3:6]].values
b = df_merged[signal_col[0:3]].values
residuals = a - b
print(residuals.shape)
print(residuals[:, 0])
# plt.plot(residuals, label=["Residual x", "Residual y", "Residual z"])
plt.plot(residuals[:, 0], label="Residual x")
plt.plot(residuals[:, 1], label="Residual y")
plt.plot(residuals[:, 2], label="Residual z")


# plt.plot(df_merged[signal_col[0]], label="simulink Signal x")
# plt.plot(df_merged[signal_col[1]], label="simulink Signal y")
# plt.plot(df_merged[signal_col[2]], label="simulink Signal z")
# plt.plot(df_merged[signal_col[3]], label="XPBD Signal x")
# plt.plot(df_merged[signal_col[4]], label="XPBD Signal y")
# plt.plot(df_merged[signal_col[5]], label="XPBD Signal z")

if len(steady_indices) > 0:
    plt.axvline(
        steady_start_idx,
        linestyle="--",
        label="Steady State Detected"
    )

plt.legend()
plt.title("Steady State Detection")
plt.xlabel("Sample")
plt.ylabel(signal_col)
plt.show()

# --------------------------------------------------
# Result
# --------------------------------------------------
print(df_trimmed.head())