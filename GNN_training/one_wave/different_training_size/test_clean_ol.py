"""
Cleaned plotting script for training-size experiments.

This script loads per-sample CSV metrics for different training sizes and produces:
- sample-wise heatmaps
- energy drift / relative energy error curves
- RMSE curves
- one-step heatmaps
- per-wave heatmaps
- per-wave improvement heatmaps
- per-wave median + IQR plots
- per-wave parameter diagnostics

Assumptions:
- Test waves correspond to ensemble members 50:100 in the original NetCDF file.
- Training waves for training size N correspond to ensemble members 100:100+N.
- Each metric CSV has shape (n_test_samples, n_rollouts).
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

from matplotlib.colors import LogNorm, SymLogNorm
from scipy.stats import pearsonr


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

BASE_DIR = Path("GNN_training/one_wave/different_training_size")
RESULTS_DIR = BASE_DIR / "all_results_plot"
DATASET_PATH = Path("GNN_training/one_wave/nc_files/wave_200_ts_600_g4_sigmamin_15.nc")

TRAINING_SIZES = [50]# 25, 50, 75, 100]
MAIN_TRAINING_SIZES = [50]

N_WAVES_TEST = 50
TEST_MEMBER_SLICE = slice(50, 100)

RMSE_NORM = LogNorm(vmin=1e-4, vmax=1e-1)
REL_ENERGY_NORM = LogNorm(vmin=1e-4, vmax=1e-1)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

METRIC_FILENAMES = {
    "pred_energy": "test_energy_pred_per_sample.csv",
    "target_energy": "test_energy_target_per_sample.csv",
    "rel_error": "test_energy_rel_error_per_sample.csv",
    "abs_error": "test_energy_abs_error_per_sample.csv",
    "mse": "test_mse_per_sample.csv",
    "rmse": "test_rmse_per_sample.csv",
}


def load_training_size_results(base_dir, training_sizes):
    """Load all metric CSVs into a nested dictionary.

    Returns
    -------
    results[train_size][metric_name] = DataFrame
    """
    results = {}

    for train_size in training_sizes:
        result_dir = Path(base_dir) / f"test_{train_size}_results"
        results[train_size] = {}

        for metric_name, filename in METRIC_FILENAMES.items():
            path = result_dir / filename
            results[train_size][metric_name] = pd.read_csv(path)

    return results


def load_initial_condition_metadata(dataset_path):
    """Load center, sigma and amplitude metadata from the original NetCDF file."""
    ds = xr.open_dataset(dataset_path)

    metadata = {
        "centers": ds["center"].values,
        "sigmas": ds["sigma_deg"].values,
        "amplitudes": ds["A"].values,
    }

    return metadata


# ---------------------------------------------------------------------
# Per-wave utilities
# ---------------------------------------------------------------------

def get_rollout_columns(df_metric):
    """Return metric columns, excluding helper columns such as wave_id."""
    return [col for col in df_metric.columns if col != "wave_id"]


def add_wave_id(df_metric, n_waves=N_WAVES_TEST):
    """Add wave_id assuming samples are ordered by wave."""
    df = df_metric.copy()

    n_samples = df.shape[0]
    samples_per_wave = n_samples // n_waves

    if n_samples % n_waves != 0:
        print(
            f"Warning: {n_samples=} is not divisible by {n_waves=}. "
            f"Using samples_per_wave={samples_per_wave}."
        )

    df["wave_id"] = np.arange(n_samples) // samples_per_wave
    df = df[df["wave_id"] < n_waves]

    return df


def compute_per_wave_metric(df_metric, n_waves=N_WAVES_TEST):
    """Average metric over all samples belonging to the same wave.

    Returns
    -------
    DataFrame with shape (n_waves, n_rollouts)
    """
    df = add_wave_id(df_metric, n_waves=n_waves)
    rollout_cols = get_rollout_columns(df)

    return df.groupby("wave_id")[rollout_cols].mean()


def get_wave_scores(df_metric, n_waves=N_WAVES_TEST):
    """Mean over all rollouts for each wave."""
    per_wave = compute_per_wave_metric(df_metric, n_waves=n_waves)
    return per_wave.mean(axis=1)


def get_wave_scores_at_rollout(df_metric, rollout_idx, n_waves=N_WAVES_TEST):
    """Mean over samples for each wave at one rollout index."""
    per_wave = compute_per_wave_metric(df_metric, n_waves=n_waves)
    rollout_col = per_wave.columns[rollout_idx]
    return per_wave[rollout_col]


def build_wave_dataframe(score, metadata, test_member_slice=TEST_MEMBER_SLICE):
    """Combine wave scores with initial-condition metadata."""
    centers = metadata["centers"][test_member_slice]
    sigmas = metadata["sigmas"][test_member_slice]
    amplitudes = metadata["amplitudes"][test_member_slice]

    n_waves = len(score)

    return pd.DataFrame({
        "wave_id": np.arange(n_waves),
        "score": score.values,
        "sigma": sigmas[:n_waves],
        "A": amplitudes[:n_waves],
        "center_x": centers[:n_waves, 0],
        "center_y": centers[:n_waves, 1],
        "center_z": centers[:n_waves, 2],
    })


# ---------------------------------------------------------------------
# Generic plotting helpers
# ---------------------------------------------------------------------

def save_figure(fig, filename):
    path = RESULTS_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_stacked_heatmaps(
    matrices,
    labels,
    title,
    colorbar_label,
    norm,
    y_label="Test sample",
    filename=None,
):
    """Plot vertically stacked heatmaps with one shared colorbar."""
    fig, axes = plt.subplots(
        len(matrices),
        1,
        figsize=(10, 3 * len(matrices)),
        sharex=True,
    )

    if len(matrices) == 1:
        axes = [axes]

    im = None

    for ax, matrix, label in zip(axes, matrices, labels):
        im = ax.imshow(
            matrix,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            norm=norm,
        )

        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(f"Train size: {label}", fontsize=14, fontweight="bold")
        ax.set_xticks(np.arange(matrix.shape[1]))

    axes[-1].set_xlabel("Rollout", fontsize=12)

    fig.subplots_adjust(left=0.12)
    cbar_ax = fig.add_axes([1.01, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(colorbar_label, fontsize=12)

    fig.suptitle(title, fontsize=18, y=0.995)
    fig.tight_layout()

    if filename:
        save_figure(fig, filename)

    return fig


# ---------------------------------------------------------------------
# Main result plots
# ---------------------------------------------------------------------

def plot_sample_heatmaps(results):
    """Sample-wise RMSE and relative-energy heatmaps."""
    selected = MAIN_TRAINING_SIZES

    plot_stacked_heatmaps(
        matrices=[results[n]["rmse"].values for n in selected],
        labels=selected,
        title="RMSE heatmaps for different training sizes",
        colorbar_label="RMSE",
        norm=RMSE_NORM,
        y_label="Test sample",
        filename="rmse_heatmaps_all.png",
    )

    plot_stacked_heatmaps(
        matrices=[results[n]["rel_error"].values for n in selected],
        labels=selected,
        title="Relative energy error heatmaps for different training sizes",
        colorbar_label="Relative energy error",
        norm=REL_ENERGY_NORM,
        y_label="Test sample",
        filename="rel_error_heatmaps_all.png",
    )


def plot_energy_drift_and_error(results):
    """Plot mean relative energy drift and relative energy error over rollout."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    for train_size in TRAINING_SIZES:
        pred_energy = results[train_size]["pred_energy"]
        rel_error = results[train_size]["rel_error"]

        rollouts_energy = np.arange(1, pred_energy.shape[1] + 1)
        rollouts_error = np.arange(1, rel_error.shape[1] + 1)

        mean_energy = pred_energy.mean()
        drift = (mean_energy - mean_energy.iloc[0]) / mean_energy.iloc[0]

        axes[0].plot(
            rollouts_energy,
            drift,
            label=rf"${train_size}$",
            linestyle="--",
            marker="o",
        )

        axes[1].semilogy(
            rollouts_error,
            rel_error.mean(),
            label=rf"${train_size}$",
            linestyle="--",
            marker="o",
        )

    axes[0].axhline(0, color="black", linestyle="-", linewidth=1.5, label="Zero drift")
    axes[0].set_ylabel("Relative energy drift", fontsize=18)
    axes[0].legend(fontsize=14)
    axes[0].grid(True)

    axes[1].set_ylabel("Relative energy error", fontsize=18)
    axes[1].set_xlabel("Rollout", fontsize=18)
    axes[1].legend(fontsize=14)
    axes[1].grid(True)

    fig.suptitle("Mean relative energy drift and relative error over rollout time", fontsize=20)
    fig.tight_layout(pad=3.0)

    save_figure(fig, "energy_drift_over_time.png")


def plot_metric_over_rollout(results, metric_name, ylabel, filename, semilogy=True):
    """Plot mean metric over rollout for each training size."""
    fig, ax = plt.subplots(figsize=(16, 10))

    for train_size in TRAINING_SIZES:
        df = results[train_size][metric_name]
        rollouts = np.arange(1, df.shape[1] + 1)

        plot_func = ax.semilogy if semilogy else ax.plot
        ax.loglog(
            rollouts,
            df.mean(),
            label=rf"${train_size}$",
            linestyle="--",
            marker="o",
        )

    ax.set_xlabel("Rollout", fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.legend(fontsize=16)
    ax.grid(True)

    fig.suptitle(f"Mean {ylabel} over rollout time for different training sizes", fontsize=20)
    fig.tight_layout(pad=3.0)

    save_figure(fig, filename)


def plot_one_step_heatmap(results, metric_name, title, colorbar_label, filename, norm):
    """Plot first-rollout metric across training sizes and samples."""
    matrix = np.vstack([
        results[train_size][metric_name].iloc[:, 0].values
        for train_size in TRAINING_SIZES
    ])

    fig, ax = plt.subplots(figsize=(12, 8))

    im = ax.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        norm=norm,
    )

    ax.set_title(title, fontsize=18)
    ax.set_ylabel("Training size", fontsize=16)
    ax.set_yticks(np.arange(len(TRAINING_SIZES)))
    ax.set_yticklabels([rf"${n}$" for n in TRAINING_SIZES])

    for y in np.arange(0.5, len(TRAINING_SIZES), 1):
        ax.axhline(y, color="black", linewidth=1)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label, fontsize=14)

    fig.suptitle(title, fontsize=20)
    fig.tight_layout(pad=3.0)

    save_figure(fig, filename)


# ---------------------------------------------------------------------
# Per-wave diagnostic plots
# ---------------------------------------------------------------------

def plot_per_wave_heatmaps(results):
    """Per-wave RMSE heatmaps for selected training sizes."""
    selected = MAIN_TRAINING_SIZES
    per_wave_rmse = [
        compute_per_wave_metric(results[n]["rmse"]).values
        for n in selected
    ]

    plot_stacked_heatmaps(
        matrices=per_wave_rmse,
        labels=selected,
        title="Per-wave RMSE heatmaps for different training sizes",
        colorbar_label="RMSE",
        norm=RMSE_NORM,
        y_label="Wave ID",
        filename="rmse_heatmaps_per_wave.png",
    )


def plot_per_wave_score_boxplots(results, rollout_indices=(0, 9, 18)):
    """Boxplots of per-wave metric distributions across training sizes."""
    metrics = {
        "RMSE": ("rmse", "rmse_rollout_comparison.png"),
        "Relative energy error": ("rel_error", "relative_energy_error_rollout_comparison.png"),
    }

    for metric_label, (metric_key, filename) in metrics.items():
        fig, axes = plt.subplots(
            1,
            len(rollout_indices),
            figsize=(5.5 * len(rollout_indices), 5),
            sharey=True,
        )

        if len(rollout_indices) == 1:
            axes = [axes]

        for ax, rollout_idx in zip(axes, rollout_indices):
            plot_data = []

            for train_size in TRAINING_SIZES:
                score = get_wave_scores_at_rollout(
                    results[train_size][metric_key],
                    rollout_idx=rollout_idx,
                )
                plot_data.append(score.values)

            ax.boxplot(plot_data, labels=[str(n) for n in TRAINING_SIZES], showmeans=True)

            for i, values in enumerate(plot_data, start=1):
                jitter = np.random.normal(0, 0.04, size=len(values))
                ax.scatter(np.full(len(values), i) + jitter, values, alpha=0.45, s=20)

            ax.set_title(f"Rollout step {rollout_idx}")
            ax.set_xlabel("Training size")
            ax.set_yscale("log")
            ax.grid(True, axis="y")

        axes[0].set_ylabel(metric_label)

        fig.suptitle(f"Per-wave {metric_label} distribution vs training size", fontsize=16)
        fig.tight_layout()

        save_figure(fig, filename)


def plot_median_iqr_vs_training_size(results, rollout_indices=(0, 9, 17)):
    """Median and interquartile range across waves as a function of training size."""
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

            for train_size in TRAINING_SIZES:
                scores = get_wave_scores_at_rollout(
                    results[train_size][metric_key],
                    rollout_idx=rollout_idx,
                ).values

                medians.append(np.median(scores))
                q25s.append(np.percentile(scores, 25))
                q75s.append(np.percentile(scores, 75))

            medians = np.asarray(medians)
            q25s = np.asarray(q25s)
            q75s = np.asarray(q75s)

            ax.plot(
                TRAINING_SIZES,
                medians,
                marker="o",
                linewidth=2,
                label=f"Rollout {rollout_idx}",
            )

            ax.fill_between(TRAINING_SIZES, q25s, q75s, alpha=0.2)

        ax.set_xlabel("Training size")
        ax.set_ylabel(metric_label)
        ax.set_title(f"Per-wave {metric_label}: median and interquartile range")
        ax.set_yscale("log")
        ax.grid(True, which="both", axis="y")
        ax.legend()

        fig.tight_layout()

        save_figure(fig, filename)


def plot_improvement_heatmaps(results):
    """Per-wave RMSE improvement heatmaps.

    Positive values mean the larger training size has lower RMSE.
    """
    rmse_waves = {
        n: compute_per_wave_metric(results[n]["rmse"]).values
        for n in TRAINING_SIZES
    }

    diff_pairs = [
        (10, 25),
        (25, 50),
        (50, 75),
        (75, 100),
    ]

    diffs = [
        rmse_waves[a] - rmse_waves[b]
        for a, b in diff_pairs
    ]

    labels = [
        f"Train {a} - Train {b}"
        for a, b in diff_pairs
    ]

    max_abs = max(np.nanmax(np.abs(diff)) for diff in diffs)

    norm = SymLogNorm(
        linthresh=1e-4,
        linscale=1.0,
        vmin=-max_abs,
        vmax=max_abs,
        base=10,
    )

    fig, axes = plt.subplots(
        len(diffs),
        1,
        figsize=(10, 3 * len(diffs)),
        sharex=True,
    )

    if len(diffs) == 1:
        axes = [axes]

    im = None

    for ax, diff, label in zip(axes, diffs, labels):
        im = ax.imshow(
            diff,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap="coolwarm",
            norm=norm,
        )

        ax.set_ylabel("Wave ID")
        ax.set_title(label, fontsize=14, fontweight="bold")
        ax.set_xticks(np.arange(diff.shape[1]))

    axes[-1].set_xlabel("Rollout")

    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("RMSE improvement")

    fig.suptitle("Per-wave RMSE improvement with more training data", fontsize=18)
    fig.tight_layout()

    save_figure(fig, "rmse_difference_heatmaps_per_wave.png")


def plot_mean_improvement_over_rollout(results):
    """Mean RMSE improvement over rollout for adjacent training-size pairs."""
    rmse_waves = {
        n: compute_per_wave_metric(results[n]["rmse"]).values
        for n in TRAINING_SIZES
    }

    pairs = [
        (10, 25),
        (25, 50),
        (50, 75),
        (75, 100),
    ]

    fig, ax = plt.subplots(figsize=(7, 5))

    for a, b in pairs:
        diff = rmse_waves[a] - rmse_waves[b]
        mean_improvement = diff.mean(axis=0)
        ax.plot(mean_improvement, label=f"{a} → {b}", marker="o")

    ax.set_xlabel("Rollout")
    ax.set_ylabel("Mean RMSE improvement")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()

    save_figure(fig, "mean_RMSE_improvement.png")


# ---------------------------------------------------------------------
# Initial-condition diagnostics
# ---------------------------------------------------------------------

def print_best_worst_waves(df_wave, label, n=5):
    """Print best/worst waves with metadata."""
    print(f"\n{label}: worst {n} waves")
    for _, row in df_wave.nlargest(n, "score").iterrows():
        print(
            int(row["wave_id"]),
            "sigma =", row["sigma"],
            "A =", row["A"],
            "center =",
            np.array([row["center_x"], row["center_y"], row["center_z"]]),
            "score =", row["score"],
        )

    print(f"\n{label}: best {n} waves")
    for _, row in df_wave.nsmallest(n, "score").iterrows():
        print(
            int(row["wave_id"]),
            "sigma =", row["sigma"],
            "A =", row["A"],
            "center =",
            np.array([row["center_x"], row["center_y"], row["center_z"]]),
            "score =", row["score"],
        )


def plot_score_vs_initial_parameters(df_wave, score_label, filename_prefix):
    """Scatter score against sigma and amplitude, highlighting best/worst waves."""
    best_idx = df_wave.nsmallest(10, "score").index
    worst_idx = df_wave.nlargest(10, "score").index

    for x_col, x_label, filename in [
        ("sigma", r"$\sigma$ [deg]", f"{filename_prefix}_score_vs_sigma.png"),
        ("A", "Amplitude A", f"{filename_prefix}_score_vs_amplitude.png"),
    ]:
        fig, ax = plt.subplots(figsize=(7, 5))

        ax.scatter(df_wave[x_col], df_wave["score"], s=60, alpha=0.7)

        ax.scatter(
            df_wave.loc[best_idx, x_col],
            df_wave.loc[best_idx, "score"],
            s=120,
            color="green",
            edgecolor="black",
            label="Best 10",
        )

        ax.scatter(
            df_wave.loc[worst_idx, x_col],
            df_wave.loc[worst_idx, "score"],
            s=120,
            color="red",
            edgecolor="black",
            label="Worst 10",
        )

        ax.set_xlabel(x_label)
        ax.set_ylabel(score_label)
        ax.set_title(f"{score_label} vs {x_label}")
        ax.legend()
        ax.grid(True)

        fig.tight_layout()
        save_figure(fig, filename)


def plot_score_histogram(df_wave, score_label, filename):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(df_wave["score"], bins=15)
    ax.set_xlabel(score_label)
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of {score_label}")
    fig.tight_layout()
    save_figure(fig, filename)


def plot_centers_colored_by_score(df_wave, score_label, filename):
    """3D scatter of wave centers colored by score."""
    best_idx = df_wave.nsmallest(5, "score").index
    worst_idx = df_wave.nlargest(5, "score").index

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")

    p = ax.scatter(
        df_wave["center_x"],
        df_wave["center_y"],
        df_wave["center_z"],
        c=df_wave["score"],
        s=80,
        alpha=0.7,
    )

    ax.scatter(
        df_wave.loc[best_idx, "center_x"],
        df_wave.loc[best_idx, "center_y"],
        df_wave.loc[best_idx, "center_z"],
        color="green",
        edgecolor="black",
        s=180,
        label="Best 5",
    )

    ax.scatter(
        df_wave.loc[worst_idx, "center_x"],
        df_wave.loc[worst_idx, "center_y"],
        df_wave.loc[worst_idx, "center_z"],
        color="red",
        edgecolor="black",
        s=180,
        label="Worst 5",
    )

    for idx in list(best_idx) + list(worst_idx):
        ax.text(
            df_wave.loc[idx, "center_x"],
            df_wave.loc[idx, "center_y"],
            df_wave.loc[idx, "center_z"],
            str(int(df_wave.loc[idx, "wave_id"])),
            fontsize=8,
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"Wave centers colored by {score_label}")
    ax.set_box_aspect([1, 1, 1])
    ax.legend()

    cbar = fig.colorbar(p, ax=ax, shrink=0.7)
    cbar.set_label(score_label)

    fig.tight_layout()
    save_figure(fig, filename)


def analyze_initial_conditions(results, metadata):
    """Initial-condition analysis for relative energy and RMSE."""
    diagnostics = [
        (
            "Relative energy error",
            "rel_error",
            10,
            "rel_energy",
        ),
        (
            "Mean RMSE",
            "rmse",
            100,
            "rmse_train100",
        ),
    ]

    for score_label, metric_key, train_size, prefix in diagnostics:
        score = get_wave_scores(results[train_size][metric_key])
        df_wave = build_wave_dataframe(score, metadata)

        print_best_worst_waves(df_wave, label=f"{score_label}, train size {train_size}")

        corr_sigma, p_sigma = pearsonr(df_wave["sigma"], df_wave["score"])
        corr_A, p_A = pearsonr(df_wave["A"], df_wave["score"])

        print("\nCorrelation statistics:")
        print(f"{score_label} vs sigma: r = {corr_sigma:.4f}, p = {p_sigma:.4e}")
        print(f"{score_label} vs A:     r = {corr_A:.4f}, p = {p_A:.4e}")

        plot_score_vs_initial_parameters(
            df_wave,
            score_label=score_label,
            filename_prefix=prefix,
        )

        plot_score_histogram(
            df_wave,
            score_label=score_label,
            filename=f"{prefix}_score_histogram.png",
        )

        plot_centers_colored_by_score(
            df_wave,
            score_label=score_label,
            filename=f"{prefix}_center_score_3d_bestworst.png",
        )


def plot_best_worst_parameter_stripplot(results, metadata, training_sizes=(10, 50, 100), rollout_indices=(0, 9, 17)):
    """Strip plot of training parameters vs best/worst test waves for several rollouts."""
    sigmas = metadata["sigmas"]
    amplitudes = metadata["amplitudes"]

    rollout_colors = {
        rollout_indices[0]: "tab:blue",
        rollout_indices[1]: "tab:orange",
        rollout_indices[2]: "tab:green",
    }

    fig, axes = plt.subplots(
        len(training_sizes),
        2,
        figsize=(17, 3.6 * len(training_sizes)),
        sharex=False,
    )

    rng = np.random.default_rng(42)

    for i, train_size in enumerate(training_sizes):
        train_slice = slice(100, 100 + train_size)
        train_sigma = sigmas[train_slice]
        train_A = amplitudes[train_slice]

        for j, param_name in enumerate(["sigma", "A"]):
            ax = axes[i, j]

            vals = train_sigma if param_name == "sigma" else train_A
            jitter = rng.normal(0, 0.05, size=len(vals))

            ax.scatter(
                np.full(len(vals), 0) + jitter,
                vals,
                s=35,
                alpha=0.45,
                color="lightgray",
                edgecolor="black",
                linewidth=0.2,
                label="Training" if i == 0 and j == 0 else None,
            )

            group_names = [
                "Training",
                "Best RMSE",
                "Worst RMSE",
                "Best Rel. energy",
                "Worst Rel. energy",
            ]

            for rollout_idx in rollout_indices:
                color = rollout_colors[rollout_idx]

                rmse_score = get_wave_scores_at_rollout(results[train_size]["rmse"], rollout_idx)
                rel_score = get_wave_scores_at_rollout(results[train_size]["rel_error"], rollout_idx)

                groups = {
                    1: rmse_score.nsmallest(10).index.values,
                    2: rmse_score.nlargest(10).index.values,
                    3: rel_score.nsmallest(10).index.values,
                    4: rel_score.nlargest(10).index.values,
                }

                for x_pos, ids in groups.items():
                    if param_name == "sigma":
                        group_vals = sigmas[TEST_MEMBER_SLICE][ids]
                    else:
                        group_vals = amplitudes[TEST_MEMBER_SLICE][ids]

                    jitter = rng.normal(0, 0.035, size=len(group_vals))

                    ax.scatter(
                        np.full(len(group_vals), x_pos) + jitter,
                        group_vals,
                        s=45,
                        alpha=0.75,
                        color=color,
                        edgecolor="black",
                        linewidth=0.25,
                        label=(
                            f"Rollout {rollout_idx}"
                            if i == 0 and j == 0 and x_pos == 1
                            else None
                        ),
                    )

            ax.set_xticks(range(len(group_names)))
            ax.set_xticklabels(group_names, rotation=25, ha="right")
            ax.grid(True, axis="y")

            if param_name == "sigma":
                ax.set_ylabel(r"$\sigma$ [deg]")
                ax.set_title(f"Training size {train_size}: Gaussian width")
            else:
                ax.set_ylabel("Amplitude A")
                ax.set_title(f"Training size {train_size}: amplitude")

    axes[0, 0].legend(fontsize=9)

    fig.suptitle(
        "Initial-condition parameters for best/worst test waves across rollout horizons",
        fontsize=16,
    )

    fig.tight_layout()

    save_figure(fig, "training_vs_best_worst_multiple_rollouts_stripplots.png")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    results = load_training_size_results(BASE_DIR, TRAINING_SIZES)
    metadata = load_initial_condition_metadata(DATASET_PATH)

    plot_sample_heatmaps(results)

    plot_energy_drift_and_error(results)

    plot_metric_over_rollout(
        results,
        metric_name="rmse",
        ylabel="RMSE",
        filename="rmse_over_time.png",
        semilogy=False,
    )

    plot_one_step_heatmap(
        results,
        metric_name="rel_error",
        title="One-step relative energy error for different training sizes",
        colorbar_label="Relative energy error",
        filename="one_step_heatmaps_rel_energy.png",
        norm=REL_ENERGY_NORM,
    )

    plot_one_step_heatmap(
        results,
        metric_name="rmse",
        title="One-step RMSE for different training sizes",
        colorbar_label="RMSE",
        filename="one_step_heatmaps_rmse.png",
        norm=RMSE_NORM,
    )

    plot_per_wave_heatmaps(results)

    plot_per_wave_score_boxplots(results, rollout_indices=(0, 9, 17))

    plot_median_iqr_vs_training_size(results, rollout_indices=(0, 9, 17))

    plot_improvement_heatmaps(results)

    plot_mean_improvement_over_rollout(results)

    analyze_initial_conditions(results, metadata)

    plot_best_worst_parameter_stripplot(
        results,
        metadata,
        training_sizes=(10, 50, 100),
        rollout_indices=(0, 9, 17),
    )


if __name__ == "__main__":
    main()
