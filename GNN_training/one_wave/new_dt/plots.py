"""
Cleaned plotting script for training-size experiments.

Updated version:
- Loads test_metadata.csv for each training size.
- Uses true ensemble_member from metadata.
- Does NOT infer wave_id from row order.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

from matplotlib.colors import LogNorm, SymLogNorm
from scipy.stats import pearsonr



BASE_DIR = Path("GNN_training/one_wave/new_dt")
RESULTS_DIR = BASE_DIR / Path("newsetup_mp3_results")
DATASET_PATH = Path("GNN_training/one_wave/nc_files/ wave_200_ts_100_Tmax6_g5.nc.nc")

TRAINING_SIZES = ["new_setup_mp3"]#5, 10, 25, 50]
MAIN_TRAINING_SIZES = ["new_setup_mp3"]#, 25, 50, 75]

RMSE_NORM = LogNorm(vmin=1e-4, vmax=1e-1)
REL_ENERGY_NORM = LogNorm(vmin=1e-4, vmax=1e-1)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Use a unique, ordered list internally. This prevents repeated keys/plots if
# TRAINING_SIZES is temporarily set to something like [50, 50, 50].
AVAILABLE_TRAINING_SIZES = list(dict.fromkeys(TRAINING_SIZES))
AVAILABLE_MAIN_TRAINING_SIZES = [
    n for n in list(dict.fromkeys(MAIN_TRAINING_SIZES))
    if n in AVAILABLE_TRAINING_SIZES
]



METRIC_FILENAMES = {
    "pred_energy": "test_energy_pred_per_sample.csv",
    "target_energy": "test_energy_target_per_sample.csv",
    "rel_error": "test_energy_rel_error_per_sample.csv",
    "abs_error": "test_energy_abs_error_per_sample.csv",
    "mse": "test_mse_per_sample.csv",
    "rmse": "test_rmse_per_sample.csv",
}

def add_initial_parameters_to_samples(df_metric, df_meta, metadata):
    df = attach_metadata(df_metric, df_meta)

    ensemble_members = df["ensemble_member"].to_numpy(dtype=int)

    df["sigma"] = metadata["sigmas"][ensemble_members]
    df["A"] = metadata["amplitudes"][ensemble_members]

    return df


def save_figure(fig, filename):
    path = RESULTS_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")



def load_test_metadata(result_dir):
    path = Path(result_dir) / "test_metadata.csv"
    return pd.read_csv(path)


def load_training_size_results(base_dir, training_sizes):
    results = {}

    for train_size in list(dict.fromkeys(training_sizes)):
        result_dir = Path(base_dir) / f"test_{train_size}"
        results[train_size] = {}

        for metric_name, filename in METRIC_FILENAMES.items():
            path = result_dir / filename
            results[train_size][metric_name] = pd.read_csv(path)

        results[train_size]["metadata"] = load_test_metadata(result_dir)

    return results


def load_initial_condition_metadata(dataset_path):
    ds = xr.open_dataset(dataset_path)

    metadata = {
        "centers": ds["center"].values,
        "sigmas": ds["sigma_deg"].values,
        "amplitudes": ds["A"].values,
    }

    return metadata


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

    for train_size in AVAILABLE_TRAINING_SIZES:
        rmse = results[train_size]["rmse"]
        rel_error = results[train_size]["rel_error"]

        rollouts_rmse = np.arange(1, rmse.shape[1] + 1)
        rollouts_error = np.arange(1, rel_error.shape[1] + 1)

        mean_rmse = rmse.mean()
        mean_rel_error = rel_error.mean()

        axes[0].loglog(
            rollouts_rmse,
            mean_rmse,
            label=rf"${train_size}$",
            linestyle="--",
            marker="o",
        )

        axes[1].loglog(
            rollouts_error,
            mean_rel_error,
            label=rf"${train_size}$",
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
        "RMSE and relative energy error over rollout",
        fontsize=20,
    )

    fig.tight_layout(pad=3.0)

    save_figure(fig, "rmse_and_relative_error_over_time.png")


def plot_median_iqr_vs_training_size(results, rollout_indices=(0, 9, 17)):
    metrics = {
        "RMSE": ("rmse", "rmse_median_iqr_vs_training_size.png"),
        "Relative energy error": ("rel_error", "relative_energy_error_median_iqr_vs_training_size.png"),
    }

    for metric_label, (metric_key, filename) in metrics.items():
        fig, ax = plt.subplots(figsize=(8, 5))

        for rollout_idx in rollout_indices:
            medians = []
            q25s = []
            q75s = []

            for train_size in AVAILABLE_TRAINING_SIZES:
                scores = get_wave_scores_at_rollout(
                    results[train_size][metric_key],
                    results[train_size]["metadata"],
                    rollout_idx,
                ).values

                medians.append(np.median(scores))
                q25s.append(np.percentile(scores, 25))
                q75s.append(np.percentile(scores, 75))

            medians = np.asarray(medians)
            q25s = np.asarray(q25s)
            q75s = np.asarray(q75s)

            ax.plot(
                AVAILABLE_TRAINING_SIZES,
                medians,
                marker="o",
                linewidth=2,
                label=f"Rollout {rollout_idx}",
            )

            #ax.fill_between(AVAILABLE_TRAINING_SIZES, q25s, q75s, alpha=0.2)

        ax.set_xlabel("Training size")
        ax.set_ylabel(metric_label)
        ax.set_title(f"Per-wave {metric_label}: median and interquartile range")
        ax.set_yscale("log")
        ax.grid(True, which="both", axis="y")
        ax.legend()

        fig.tight_layout()

        save_figure(fig, filename)



def main():
    results = load_training_size_results(BASE_DIR, AVAILABLE_TRAINING_SIZES)

    plot_rmse_and_relative_error(results)
    plot_median_iqr_vs_training_size(results, rollout_indices=(0, 9, 17))

if __name__ == "__main__":
    main()

