"""
Robot dataset plotter & inspector.
Usage:  python plot_datasets.py [file1.csv file2.csv ...]
        python plot_datasets.py          # plots all *.csv in current folder
"""

import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt

# ── helpers ──────────────────────────────────────────────────────────────────

def load(path):
    df = pd.read_csv(path)
    # use time_s as index if present, otherwise fall back to row number
    if "time_s" in df.columns:
        df = df.set_index("time_s")
        df.index -= df.index[0]          # start from t=0
        df.index.name = "time (s)"
    # drop columns we don't want to plot
    df = df.drop(columns=["time_ns", "msg_stamp_ns", "frame_id"], errors="ignore")
    return df


def group_columns(df):
    """Return a dict of {group_label: [col, ...]} based on column name prefixes."""
    groups = {}
    for col in df.columns:
        prefix = col.rsplit("_", 1)[0] if "_" in col else col
        groups.setdefault(prefix, []).append(col)
    return groups


def plot_file(path):
    df = load(path)
    title = path.split("/")[-1].replace(".csv", "")
    groups = group_columns(df)

    n = len(groups)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for ax, (label, cols) in zip(axes, groups.items()):
        for col in cols:
            ax.plot(df.index, df[col], label=col, linewidth=0.9)
        ax.set_ylabel(label, fontsize=9)
        ax.legend(fontsize=7, loc="upper right", ncol=min(len(cols), 4))
        ax.grid(True, linewidth=0.4, alpha=0.5)

    axes[-1].set_xlabel(df.index.name or "sample")
    fig.tight_layout()
    return fig


def print_summary(path, df):
    print(f"\n{'─'*60}")
    print(f"  {path}")
    print(f"  {len(df)} rows  ·  {len(df.columns)} signal columns")
    print(df.describe().round(4).to_string())

# ── main ─────────────────────────────────────────────────────────────────────

import os

raw_args = sys.argv[1:] or ["."]
files = []
for arg in raw_args:
    if os.path.isdir(arg):
        files.extend(sorted(glob.glob(os.path.join(arg, "*.csv"))))
    else:
        files.extend(sorted(glob.glob(arg)))  # also supports wildcards

if not files:
    print("No CSV files found.")
    sys.exit(1)

for path in files:
    try:
        df = load(path)
        print_summary(path, df)
        plot_file(path)
    except Exception as e:
        print(f"  [skip] {path}: {e}")

plt.show()
