from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path("GNN_training/one_wave/different_mesh_size/final_results")
RESULTS_DIR = Path("GNN_training/one_wave/different_mesh_size/final_results_plots/IB/final")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DT_RUNS = ["1dt"]

RUN_DIRS = {
    "1dt": {
        "label": r"",
        "result_dir": "test_1dt_zero",
        "dt_scale": 1,
    },
    "80dt": {
        "label": r"$80\Delta t$",
        "result_dir": "test_80dt",
        "dt_scale": 80,
    },
}

METRIC_FILENAMES = {
    "energy_pred": "test_energy_pred_per_sample.csv",
    "energy_true": "test_energy_target_per_sample.csv",
    "rmse": "test_rmse_per_sample.csv",
}

ZERO_MEMBER = 50


def filter_member(df, member=ZERO_MEMBER):
    if "ensemble_member" not in df.columns:
        raise KeyError("CSV does not contain an 'ensemble_member' column.")

    df_member = df[df["ensemble_member"] == member]

    if len(df_member) == 0:
        raise ValueError(f"No rows found for ensemble_member={member}")

    return df_member

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
            "dt_scale": cfg["dt_scale"],
        }

        for metric_name, filename in METRIC_FILENAMES.items():
            path = result_dir / filename

            if not path.exists():
                raise FileNotFoundError(f"Could not find file: {path}")

            df = pd.read_csv(path)
            df = filter_member(df, ZERO_MEMBER)

            results[run_key][metric_name] = df

    return results

def get_rollout_cols(df_metric):
    rollout_cols = [
        col for col in df_metric.columns
        if col.startswith("rollout_")
    ]

    rollout_cols = sorted(
        rollout_cols,
        key=lambda x: int(x.split("_")[-1])
    )

    rollout_numbers = np.array([
        int(col.split("_")[-1]) for col in rollout_cols
    ])

    return rollout_cols, rollout_numbers


def get_mean_energy_curves(energy_pred_df, energy_true_df):
    cols_pred, rollouts_pred = get_rollout_cols(energy_pred_df)
    cols_true, rollouts_true = get_rollout_cols(energy_true_df)

    if not np.array_equal(rollouts_pred, rollouts_true):
        raise ValueError("Predicted and true energy files have different rollout columns.")

    energy_pred = energy_pred_df[cols_pred].values
    energy_true = energy_true_df[cols_true].values

    mean_energy_pred = energy_pred.mean(axis=0)
    mean_energy_true = energy_true.mean(axis=0)

    std_energy_pred = energy_pred.std(axis=0)
    std_energy_true = energy_true.std(axis=0)

    print("Mean true energy first/last:")
    print(mean_energy_true[0], mean_energy_true[-1])

    print("Mean predicted energy first/last:")
    print(mean_energy_pred[0], mean_energy_pred[-1])

    print("Signed relative energy error at final rollout:")
    print((mean_energy_pred[-1] - mean_energy_true[-1]) / (abs(mean_energy_true[-1]) + 1e-12))

    return mean_energy_pred, mean_energy_true, std_energy_pred, std_energy_true, rollouts_pred


def plot_rmse_and_energy(results):
    fig, axes = plt.subplots(2, 1, figsize=(16, 13), sharex=True)

    rmse_1 = results["1dt"]["rmse"]
    cols_rmse_1, rollouts_rmse_1 = get_rollout_cols(rmse_1)

    (
        mean_energy_pred_1,
        mean_energy_true_1,
        std_energy_pred_1,
        std_energy_true_1,
        rollouts_energy_1,
    ) = get_mean_energy_curves(
        results["1dt"]["energy_pred"],
        results["1dt"]["energy_true"],
    )

    axes[0].loglog(
        rollouts_rmse_1,
        rmse_1[cols_rmse_1].mean(axis=0),
        label="RMSE",#r"$1\Delta t$",
        linestyle="-",
        marker="o",
    )

    axes[1].loglog(
        rollouts_energy_1,
        mean_energy_true_1,
        label="True energy",
        linestyle="-",
        marker="o",
    )

    axes[1].loglog(
        rollouts_energy_1,
        mean_energy_pred_1,
        label="Predicted energy",
        linestyle="--",
        marker="o",
    )

    axes[0].set_ylabel("RMSE", fontsize=20)
    axes[0].legend(fontsize=18)
    axes[0].grid(True, which="both", alpha=0.4)

    axes[1].set_ylabel("Energy", fontsize=20)
    axes[1].set_xlabel("Physical time (s)", fontsize=20)
    axes[1].legend(fontsize=18)
    axes[1].grid(True, which="both", alpha=0.4)

    fig.suptitle(
        "RMSE and energy over rollouts - zero wave",
        fontsize=22,
    )

    fig.tight_layout(pad=2.0)

    save_figure(
        fig,
        "rmse_true_vs_pred_energy_1dt_zero.png",
    )


def main():
    results = load_dt_results(BASE_DIR, RUN_DIRS)
    plot_rmse_and_energy(results)


if __name__ == "__main__":
    main()