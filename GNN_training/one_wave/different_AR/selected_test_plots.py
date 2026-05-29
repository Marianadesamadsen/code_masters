"""
Plotting script for AR-rollout-in-training experiments.

Compares models trained with different autoregressive rollout lengths,
for example AR1 and AR5.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

from matplotlib.colors import LogNorm


BASE_DIR = Path("GNN_training/one_wave/different_AR")
RESULTS_DIR = BASE_DIR / "all_results_plot"
DATASET_PATH = Path(
    "GNN_training/one_wave/nc_files/wave_200_ts_600_g4_sigmamin_6.nc"
)

AR_TRAINING_RUNS = ["AR1", "AR5"]
MAIN_AR_TRAINING_RUNS = ["AR1", "AR5"]

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

AVAILABLE_AR_TRAINING_RUNS = list(dict.fromkeys(AR_TRAINING_RUNS))
AVAILABLE_MAIN_AR_TRAINING_RUNS = [
    n for n in list(dict.fromkeys(MAIN_AR_TRAINING_RUNS))
    if n in AVAILABLE_AR_TRAINING_RUNS
]

METRIC_FILENAMES = {
    "pred_energy": "test_energy_pred_per_sample.csv",
    "target_energy": "test_energy_target_per_sample.csv",
    "rel_error": "test_energy_rel_error_per_sample.csv",
    "abs_error": "test_energy_abs_error_per_sample.csv",
    "mse": "test_mse_per_sample.csv",
    "rmse": "test_rmse_per_sample.csv",
}


def save_figure(fig, filename):
    path = RESULTS_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def load_test_metadata(result_dir):
    path = Path(result_dir) / "test_metadata.csv"
    return pd.read_csv(path)


def load_ar_training_results(base_dir, ar_training_runs):
    results = {}

    for ar_run in list(dict.fromkeys(ar_training_runs)):
        result_dir = Path(base_dir) / f"test_75_results_{ar_run}_100"
        results[ar_run] = {}

        for metric_name, filename in METRIC_FILENAMES.items():
            path = result_dir / filename
            results[ar_run][metric_name] = pd.read_csv(path)

        results[ar_run]["metadata"] = load_test_metadata(result_dir)

    return results


def load_initial_condition_metadata(dataset_path):
    ds = xr.open_dataset(dataset_path)

    return {
        "centers": ds["center"].values,
        "sigmas": ds["sigma_deg"].values,
        "amplitudes": ds["A"].values,
    }


def get_rollout_columns(df_metric):
    return [col for col in df_metric.columns if col.startswith("rollout_")]


def attach_metadata(df_metric, df_meta):
    if len(df_metric) != len(df_meta):
        raise ValueError(
            f"Metric and metadata length mismatch: "
            f"{len(df_metric)=}, {len(df_meta)=}"
        )

    df = df_metric.copy()
    df["ensemble_member"] = df_meta["ensemble_member"].values
    df["sample_idx"] = df_meta["sample_idx"].values

    return df


def compute_per_wave_metric(df_metric, df_meta):
    df = attach_metadata(df_metric, df_meta)
    rollout_cols = get_rollout_columns(df)

    return df.groupby("ensemble_member")[rollout_cols].mean()


def get_wave_scores(df_metric, df_meta):
    per_wave = compute_per_wave_metric(df_metric, df_meta)
    return per_wave.mean(axis=1)


def get_wave_scores_at_rollout(df_metric, df_meta, rollout_idx):
    per_wave = compute_per_wave_metric(df_metric, df_meta)
    rollout_col = per_wave.columns[rollout_idx]

    return per_wave[rollout_col]


def build_wave_dataframe(score, metadata):
    ensemble_members = score.index.to_numpy(dtype=int)

    return pd.DataFrame({
        "ensemble_member": ensemble_members,
        "score": score.values,
        "sigma": metadata["sigmas"][ensemble_members],
        "A": metadata["amplitudes"][ensemble_members],
        "center_x": metadata["centers"][ensemble_members, 0],
        "center_y": metadata["centers"][ensemble_members, 1],
        "center_z": metadata["centers"][ensemble_members, 2],
    })


def plot_rmse_and_relative_error(results):
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    for ar_run in AVAILABLE_AR_TRAINING_RUNS:
        rmse = results[ar_run]["rmse"]
        rel_error = results[ar_run]["rel_error"]

        rollouts_rmse = np.arange(1, rmse.shape[1] + 1)
        rollouts_error = np.arange(1, rel_error.shape[1] + 1)

        mean_rmse = rmse.mean()
        mean_rel_error = rel_error.mean()

        axes[0].loglog(
            rollouts_rmse,
            mean_rmse,
            label=ar_run,
            linestyle="--",
            marker="o",
        )

        axes[1].loglog(
            rollouts_error,
            mean_rel_error,
            label=ar_run,
            linestyle="--",
            marker="o",
        )

    axes[0].set_ylabel("RMSE", fontsize=18)
    axes[0].legend(fontsize=14)
    axes[0].grid(True, which="both")

    axes[1].set_ylabel("Relative energy error", fontsize=18)
    axes[1].set_xlabel("Rollout", fontsize=18)
    axes[1].legend(fontsize=14)
    axes[1].grid(True, which="both")

    fig.suptitle(
        "RMSE and relative energy error over rollout for different AR training horizons",
        fontsize=20,
    )

    fig.tight_layout(pad=3.0)

    save_figure(fig, "rmse_and_relative_error_over_rollout_ar_training.png")


def plot_median_iqr_vs_ar_training(results, rollout_indices=(0, 9, 17)):
    metrics = {
        "RMSE": (
            "rmse",
            "rmse_median_iqr_vs_ar_training.png",
        ),
        "Relative energy error": (
            "rel_error",
            "relative_energy_error_median_iqr_vs_ar_training.png",
        ),
    }

    x = np.arange(len(AVAILABLE_AR_TRAINING_RUNS))
    x_labels = AVAILABLE_AR_TRAINING_RUNS

    for metric_label, (metric_key, filename) in metrics.items():
        fig, ax = plt.subplots(figsize=(8, 5))

        for rollout_idx in rollout_indices:
            medians = []
            q25s = []
            q75s = []

            for ar_run in AVAILABLE_AR_TRAINING_RUNS:
                scores = get_wave_scores_at_rollout(
                    results[ar_run][metric_key],
                    results[ar_run]["metadata"],
                    rollout_idx,
                ).values

                medians.append(np.median(scores))
                q25s.append(np.percentile(scores, 25))
                q75s.append(np.percentile(scores, 75))

            medians = np.asarray(medians)
            q25s = np.asarray(q25s)
            q75s = np.asarray(q75s)

            ax.plot(
                x,
                medians,
                marker="o",
                linewidth=2,
                label=f"Rollout {rollout_idx + 1}",
            )

            # Uncomment if you want IQR shading:
            # ax.fill_between(x, q25s, q75s, alpha=0.2)

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("AR rollout used during training")
        ax.set_ylabel(metric_label)
        ax.set_title(
            f"Per-wave {metric_label}: median across AR training horizons"
        )
        ax.set_yscale("log")
        ax.grid(True, which="both", axis="y")
        ax.legend()

        fig.tight_layout()

        save_figure(fig, filename)


def plot_median_iqr_vs_ar_training_two_metrics(
    results,
    rollout_indices=(0, 9, 17),
):
    metrics = [
        ("RMSE", "rmse"),
        ("Relative energy error", "rel_error"),
    ]

    x = np.arange(len(AVAILABLE_AR_TRAINING_RUNS))
    x_labels = AVAILABLE_AR_TRAINING_RUNS

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5),
        sharex=True,
    )

    for ax, (metric_label, metric_key) in zip(axes, metrics):
        for rollout_idx in rollout_indices:
            medians = []
            q25s = []
            q75s = []

            for ar_run in AVAILABLE_AR_TRAINING_RUNS:
                scores = get_wave_scores_at_rollout(
                    results[ar_run][metric_key],
                    results[ar_run]["metadata"],
                    rollout_idx,
                ).values

                medians.append(np.median(scores))
                q25s.append(np.percentile(scores, 25))
                q75s.append(np.percentile(scores, 75))

            medians = np.asarray(medians)
            q25s = np.asarray(q25s)
            q75s = np.asarray(q75s)

            ax.plot(
                x,
                medians,
                marker="o",
                linewidth=2,
                label=f"Rollout {rollout_idx + 1}",
            )

            # Uncomment if wanted:
            # ax.fill_between(x, q25s, q75s, alpha=0.2)

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("AR rollout used during training")
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label)
        ax.set_yscale("log")
        ax.grid(True, which="both", axis="y")
        ax.legend()

    fig.suptitle(
        "Per-wave median performance across AR training horizons",
        fontsize=16,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "rmse_and_relative_energy_error_median_vs_ar_training.png",
    )


def main():
    results = load_ar_training_results(
        BASE_DIR,
        AVAILABLE_AR_TRAINING_RUNS,
    )

    plot_rmse_and_relative_error(results)


if __name__ == "__main__":
    main()