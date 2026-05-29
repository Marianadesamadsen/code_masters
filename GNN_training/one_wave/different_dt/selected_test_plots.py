"""
Plot RMSE and relative energy error over rollout for the dt experiment.

The normal-dt and double-dt runs may have the same number of saved rollout columns,
but they do not correspond to the same physical time.

To compare equal physical time:
- normal_dt uses every second rollout: 2, 4, 6, ..., 20
- double_dt uses rollouts: 1, 2, 3, ..., 10

Both are plotted against the same equivalent normal-dt rollout axis:
2, 4, 6, ..., 20.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path("GNN_training/one_wave/different_dt")
RESULTS_DIR = BASE_DIR / "all_results_plot"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DT_RUNS = ["normal_dt", "double_dt"]

RUN_DIRS = {
    "normal_dt": {
        "label": r"$\Delta t$",
        "result_dir": "test_75_results_new",
        "columns": np.arange(0, 20),      # rollout 1,...,20
        "x_axis": np.arange(1, 21),       # physical normal-dt rollout index
    },
    "double_dt": {
        "label": r"$2\Delta t$",
        "result_dir": "test_75_results_double_dt",
        "columns": np.arange(0, 10),      # rollout 1,...,10
        "x_axis": np.arange(2, 21, 2),    # matches normal rollout 2,4,...,20
    },
}

METRIC_FILENAMES = {
    "rel_error": "test_energy_rel_error_per_sample.csv",
    "rmse": "test_rmse_per_sample.csv",
}


def save_figure(fig, filename):
    path = RESULTS_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def load_dt_results(base_dir, run_dirs):
    results = {}

    for run_key, cfg in run_dirs.items():
        result_dir = Path(base_dir) / cfg["result_dir"]

        results[run_key] = {
            "label": cfg["label"],
            "columns": cfg["columns"],
            "x_axis": cfg["x_axis"],
        }

        for metric_name, filename in METRIC_FILENAMES.items():
            path = result_dir / filename
            results[run_key][metric_name] = pd.read_csv(path)

    return results


def select_aligned_rollouts(df_metric, columns):
    rollout_cols = [
        col for col in df_metric.columns
        if col.startswith("rollout_")
    ]

    selected_cols = [rollout_cols[i] for i in columns if i < len(rollout_cols)]

    return df_metric[selected_cols]


def plot_rmse_and_relative_error_over_time(results):
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    for run_key in DT_RUNS:
        rmse = results[run_key]["rmse"]
        rel_error = results[run_key]["rel_error"]

        columns = results[run_key]["columns"]
        x_axis = results[run_key]["x_axis"]
        label = results[run_key]["label"]

        rmse_aligned = select_aligned_rollouts(rmse, columns)
        rel_error_aligned = select_aligned_rollouts(rel_error, columns)

        # Make x-axis match if fewer columns exist
        x_rmse = x_axis[: rmse_aligned.shape[1]]
        x_rel_error = x_axis[: rel_error_aligned.shape[1]]

        axes[0].loglog(
            x_rmse,
            rmse_aligned.mean(axis=0),
            label=label,
            linestyle="--",
            marker="o",
        )

        axes[1].loglog(
            x_rel_error,
            rel_error_aligned.mean(axis=0),
            label=label,
            linestyle="--",
            marker="o",
        )

    axes[0].set_ylabel("RMSE", fontsize=18)
    axes[0].legend(fontsize=14)
    axes[0].grid(True, which="both", alpha=0.4)

    axes[1].set_ylabel("Relative energy error", fontsize=18)
    axes[1].set_xlabel(
        r"Physical rollout horizon $K \Delta t$",
        fontsize=18,
    )
    axes[1].legend(fontsize=14)
    axes[1].grid(True, which="both", alpha=0.4)

    fig.suptitle(
        "RMSE and relative energy error at matched physical rollout times",
        fontsize=20,
    )

    fig.tight_layout(pad=3.0)

    save_figure(
        fig,
        "rmse_and_relative_error_matched_physical_time_dt_experiment.png",
    )


def main():
    results = load_dt_results(BASE_DIR, RUN_DIRS)
    plot_rmse_and_relative_error_over_time(results)


if __name__ == "__main__":
    main()