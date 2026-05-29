"""
Plotting script for message-passing-depth experiments.

This version:
- Compares different message-passing steps instead of training sizes.
- Loads test_metadata.csv for each MP-depth result folder.
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


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

BASE_DIR = Path("GNN_training/one_wave/different_mp")
RESULTS_DIR = Path("GNN_training/one_wave/different_mp/results")
DATASET_PATH = Path("GNN_training/one_wave/nc_files/wave_200_ts_600_g4_sigmamin_6.nc")

# Message-passing step: result folder name
MP_RUNS = {
    1: "test_mp1_results_new",
    2: "test_mp2_results_new",
    3: "test_mp3_results_new",
}

MP_STEPS = list(MP_RUNS.keys())
MAIN_MP_STEPS = MP_STEPS

RMSE_NORM = LogNorm(vmin=1e-4, vmax=1e-1)
REL_ENERGY_NORM = LogNorm(vmin=1e-4, vmax=1e-1)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


METRIC_FILENAMES = {
    "pred_energy": "test_energy_pred_per_sample.csv",
    "target_energy": "test_energy_target_per_sample.csv",
    "rel_error": "test_energy_rel_error_per_sample.csv",
    "abs_error": "test_energy_abs_error_per_sample.csv",
    "mse": "test_mse_per_sample.csv",
    "rmse": "test_rmse_per_sample.csv",
}


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_test_metadata(result_dir):
    path = Path(result_dir) / "test_metadata.csv"
    return pd.read_csv(path)


def load_mp_results(base_dir, mp_runs):
    results = {}

    for mp_step, folder_name in mp_runs.items():
        result_dir = Path(base_dir) / folder_name

        if not result_dir.exists():
            print(f"Skipping MP step {mp_step}: folder does not exist: {result_dir}")
            continue

        results[mp_step] = {}

        for metric_name, filename in METRIC_FILENAMES.items():
            path = result_dir / filename
            results[mp_step][metric_name] = pd.read_csv(path)

        results[mp_step]["metadata"] = load_test_metadata(result_dir)

    return results


def load_initial_condition_metadata(dataset_path):
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


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def save_figure(fig, filename):
    path = RESULTS_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def mp_label(mp_step):
    return f"MP {mp_step}"


# ---------------------------------------------------------------------
# Heatmaps and rollout plots
# ---------------------------------------------------------------------

def plot_stacked_heatmaps(
    matrices,
    labels,
    title,
    colorbar_label,
    norm,
    y_label="Test sample",
    filename=None,
):
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
        ax.set_title(f"Message-passing steps: {label}", fontsize=14, fontweight="bold")
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

def plot_best_worst_parameter_stripplot(
    results,
    metadata,
    mp_steps=None,
    rollout_indices=(0, 9, 17),
):
    if mp_steps is None:
        mp_steps = list(results.keys())

    mp_steps = [mp for mp in mp_steps if mp in results]

    if len(mp_steps) == 0:
        print("Skipping stripplot: no requested MP steps are available.")
        return

    sigmas = metadata["sigmas"]
    amplitudes = metadata["amplitudes"]

    rollout_colors = {
        rollout_indices[0]: "tab:blue",
        rollout_indices[1]: "tab:orange",
        rollout_indices[2]: "tab:green",
    }

    fig, axes = plt.subplots(
        len(mp_steps),
        2,
        figsize=(17, 3.6 * len(mp_steps)),
        sharex=False,
    )

    if len(mp_steps) == 1:
        axes = np.asarray([axes])

    rng = np.random.default_rng(42)

    for i, mp_step in enumerate(mp_steps):
        for j, param_name in enumerate(["sigma", "A"]):
            ax = axes[i, j]

            group_names = [
                "Best RMSE",
                "Worst RMSE",
                "Best Rel. energy",
                "Worst Rel. energy",
            ]

            for rollout_idx in rollout_indices:
                color = rollout_colors[rollout_idx]

                rmse_score = get_wave_scores_at_rollout(
                    results[mp_step]["rmse"],
                    results[mp_step]["metadata"],
                    rollout_idx,
                )

                rel_score = get_wave_scores_at_rollout(
                    results[mp_step]["rel_error"],
                    results[mp_step]["metadata"],
                    rollout_idx,
                )

                groups = {
                    0: rmse_score.nsmallest(10).index.to_numpy(dtype=int),
                    1: rmse_score.nlargest(10).index.to_numpy(dtype=int),
                    2: rel_score.nsmallest(10).index.to_numpy(dtype=int),
                    3: rel_score.nlargest(10).index.to_numpy(dtype=int),
                }

                for x_pos, ids in groups.items():
                    if param_name == "sigma":
                        group_vals = sigmas[ids]
                    else:
                        group_vals = amplitudes[ids]

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
                            if i == 0 and j == 0 and x_pos == 0
                            else None
                        ),
                    )

            ax.set_xticks(range(len(group_names)))
            ax.set_xticklabels(group_names, rotation=25, ha="right")
            ax.grid(True, axis="y")

            if param_name == "sigma":
                ax.set_ylabel(r"$\sigma$ [deg]")
                ax.set_title(f"MP {mp_step}: Gaussian width")
            else:
                ax.set_ylabel("Amplitude A")
                ax.set_title(f"MP {mp_step}: amplitude")

    axes[0, 0].legend(fontsize=9)

    fig.suptitle(
        "Initial-condition parameters for best/worst test waves across rollout horizons",
        fontsize=16,
    )

    fig.tight_layout()

    save_figure(fig, "best_worst_multiple_rollouts_stripplots_mp.png")


def plot_sample_heatmaps(results):
    selected = [mp for mp in MAIN_MP_STEPS if mp in results]

    plot_stacked_heatmaps(
        matrices=[results[mp]["rmse"].values for mp in selected],
        labels=selected,
        title="RMSE heatmaps for different message-passing steps",
        colorbar_label="RMSE",
        norm=RMSE_NORM,
        y_label="Test sample",
        filename="rmse_heatmaps_all_mp.png",
    )

    plot_stacked_heatmaps(
        matrices=[results[mp]["rel_error"].values for mp in selected],
        labels=selected,
        title="Relative energy error heatmaps for different message-passing steps",
        colorbar_label="Relative energy error",
        norm=REL_ENERGY_NORM,
        y_label="Test sample",
        filename="rel_error_heatmaps_all_mp.png",
    )


def plot_energy_drift_and_error(results):
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    for mp_step in results:
        pred_energy = results[mp_step]["pred_energy"]
        rel_error = results[mp_step]["rel_error"]

        rollouts_energy = np.arange(1, pred_energy.shape[1] + 1)
        rollouts_error = np.arange(1, rel_error.shape[1] + 1)

        mean_energy = pred_energy.mean()
        drift = (mean_energy - mean_energy.iloc[0]) / mean_energy.iloc[0]

        axes[0].plot(
            rollouts_energy,
            drift,
            label=mp_label(mp_step),
            linestyle="--",
            marker="o",
        )

        axes[1].loglog(
            rollouts_error,
            rel_error.mean(),
            label=mp_label(mp_step),
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
    axes[1].grid(True, which="both")

    fig.suptitle(
        "Mean relative energy drift and relative error over rollout time",
        fontsize=20,
    )
    fig.tight_layout(pad=3.0)

    save_figure(fig, "energy_drift_over_time_mp.png")


def plot_metric_over_rollout(results, metric_name, ylabel, filename, semilogy=True):
    fig, ax = plt.subplots(figsize=(16, 10))

    for mp_step in results:
        df = results[mp_step][metric_name]
        rollouts = np.arange(1, df.shape[1] + 1)

        if semilogy:
            ax.semilogy(
                rollouts,
                df.mean(),
                label=mp_label(mp_step),
                linestyle="--",
                marker="o",
            )
        else:
            ax.loglog(
                rollouts,
                df.mean(),
                label=mp_label(mp_step),
                linestyle="--",
                marker="o",
            )

    ax.set_xlabel("Rollout", fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.legend(fontsize=16)
    ax.grid(True, which="both")

    fig.suptitle(
        f"Mean {ylabel} over rollout time for different message-passing steps",
        fontsize=20,
    )
    fig.tight_layout(pad=3.0)

    save_figure(fig, filename)


def plot_one_step_heatmap(results, metric_name, title, colorbar_label, filename, norm):
    mp_steps = list(results.keys())

    matrix = np.vstack([
        results[mp_step][metric_name].iloc[:, 0].values
        for mp_step in mp_steps
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
    ax.set_ylabel("Message-passing steps", fontsize=16)
    ax.set_yticks(np.arange(len(mp_steps)))
    ax.set_yticklabels([str(mp) for mp in mp_steps])

    for y in np.arange(0.5, len(mp_steps), 1):
        ax.axhline(y, color="black", linewidth=1)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label, fontsize=14)

    fig.tight_layout(pad=3.0)

    save_figure(fig, filename)


# ---------------------------------------------------------------------
# Per-wave diagnostics
# ---------------------------------------------------------------------

def plot_per_wave_heatmaps(results):
    selected = [mp for mp in MAIN_MP_STEPS if mp in results]

    per_wave_rmse = [
        compute_per_wave_metric(
            results[mp]["rmse"],
            results[mp]["metadata"],
        ).values
        for mp in selected
    ]

    plot_stacked_heatmaps(
        matrices=per_wave_rmse,
        labels=selected,
        title="Per-wave RMSE heatmaps for different message-passing steps",
        colorbar_label="RMSE",
        norm=RMSE_NORM,
        y_label="Ensemble member",
        filename="rmse_heatmaps_per_wave_mp.png",
    )


def plot_per_wave_score_boxplots(results, rollout_indices=(0, 9, 17)):
    metrics = {
        "RMSE": ("rmse", "rmse_rollout_comparison_mp.png"),
        "Relative energy error": (
            "rel_error",
            "relative_energy_error_rollout_comparison_mp.png",
        ),
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

            for mp_step in results:
                score = get_wave_scores_at_rollout(
                    results[mp_step][metric_key],
                    results[mp_step]["metadata"],
                    rollout_idx,
                )
                plot_data.append(score.values)

            ax.boxplot(
                plot_data,
                labels=[str(mp) for mp in results.keys()],
                showmeans=True,
            )

            for i, values in enumerate(plot_data, start=1):
                jitter = np.random.normal(0, 0.04, size=len(values))
                ax.scatter(np.full(len(values), i) + jitter, values, alpha=0.45, s=20)

            ax.set_title(f"Rollout step {rollout_idx}")
            ax.set_xlabel("Message-passing steps")
            ax.set_yscale("log")
            ax.grid(True, axis="y", which="both")

        axes[0].set_ylabel(metric_label)

        fig.suptitle(
            f"Per-wave {metric_label} distribution vs message-passing steps",
            fontsize=16,
        )
        fig.tight_layout()

        save_figure(fig, filename)


def plot_median_iqr_vs_mp_steps(results, rollout_indices=(0, 9, 17)):
    metrics = {
        "RMSE": ("rmse", "rmse_median_iqr_vs_mp_steps.png"),
        "Relative energy error": (
            "rel_error",
            "relative_energy_error_median_iqr_vs_mp_steps.png",
        ),
    }

    mp_steps = list(results.keys())

    for metric_label, (metric_key, filename) in metrics.items():
        fig, ax = plt.subplots(figsize=(8, 5))

        for rollout_idx in rollout_indices:
            medians = []
            q25s = []
            q75s = []

            for mp_step in mp_steps:
                scores = get_wave_scores_at_rollout(
                    results[mp_step][metric_key],
                    results[mp_step]["metadata"],
                    rollout_idx,
                ).values

                medians.append(np.median(scores))
                q25s.append(np.percentile(scores, 25))
                q75s.append(np.percentile(scores, 75))

            medians = np.asarray(medians)
            q25s = np.asarray(q25s)
            q75s = np.asarray(q75s)

            ax.plot(
                mp_steps,
                medians,
                marker="o",
                linewidth=2,
                label=f"Rollout {rollout_idx}",
            )

            ax.fill_between(mp_steps, q25s, q75s, alpha=0.2)

        ax.set_xlabel("Message-passing steps")
        ax.set_ylabel(metric_label)
        ax.set_title(f"Per-wave {metric_label}: median and interquartile range")
        ax.set_yscale("log")
        ax.grid(True, which="both", axis="y")
        ax.legend()

        fig.tight_layout()

        save_figure(fig, filename)


def plot_improvement_heatmaps(results):
    rmse_waves = {}

    for mp_step in results:
        rmse_waves[mp_step] = compute_per_wave_metric(
            results[mp_step]["rmse"],
            results[mp_step]["metadata"],
        )

    mp_steps = list(results.keys())

    diff_pairs = [
        (mp_steps[i], mp_steps[i + 1])
        for i in range(len(mp_steps) - 1)
    ]

    if len(diff_pairs) == 0:
        print("Skipping RMSE improvement heatmaps: need at least two MP settings.")
        return

    diffs = []
    labels = []

    for a, b in diff_pairs:
        common_members = rmse_waves[a].index.intersection(rmse_waves[b].index)

        if len(common_members) == 0:
            print(f"Skipping pair MP {a}->MP {b}: no common ensemble members.")
            continue

        diff = (
            rmse_waves[a].loc[common_members].values
            - rmse_waves[b].loc[common_members].values
        )

        diffs.append(diff)
        labels.append(f"MP {a} - MP {b}")

    if len(diffs) == 0:
        print("Skipping RMSE improvement heatmaps: no valid differences.")
        return

    max_abs = max(np.nanmax(np.abs(diff)) for diff in diffs)

    if max_abs == 0 or not np.isfinite(max_abs):
        max_abs = 1e-12

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

        ax.set_ylabel("Ensemble member")
        ax.set_title(label, fontsize=14, fontweight="bold")
        ax.set_xticks(np.arange(diff.shape[1]))

    axes[-1].set_xlabel("Rollout")

    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("RMSE improvement")

    fig.suptitle("Per-wave RMSE improvement with more message-passing steps", fontsize=18)
    fig.tight_layout()

    save_figure(fig, "rmse_difference_heatmaps_per_wave_mp.png")


def plot_mean_improvement_over_rollout(results):
    rmse_waves = {
        mp: compute_per_wave_metric(
            results[mp]["rmse"],
            results[mp]["metadata"],
        )
        for mp in results
    }

    mp_steps = list(results.keys())

    pairs = [
        (mp_steps[i], mp_steps[i + 1])
        for i in range(len(mp_steps) - 1)
    ]
    pairs.append((mp_steps[0],mp_steps[2]))

    if len(pairs) == 0:
        print("Skipping mean RMSE improvement plot: need at least two MP settings.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    for a, b in pairs:
        common_members = rmse_waves[a].index.intersection(rmse_waves[b].index)

        if len(common_members) == 0:
            print(f"Skipping pair MP {a}->MP {b}: no common ensemble members.")
            continue

        diff = (
            rmse_waves[a].loc[common_members].values
            - rmse_waves[b].loc[common_members].values
        )

        mean_improvement = diff.mean(axis=0)
        ax.plot(mean_improvement, label=f"MP {a} → MP {b}", marker="o")

    ax.set_xlabel("Rollout")
    ax.set_ylabel("Mean RMSE improvement")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()

    save_figure(fig, "mean_rmse_improvement_mp.png")


# ---------------------------------------------------------------------
# Initial-condition diagnostics
# ---------------------------------------------------------------------

def print_best_worst_waves(df_wave, label, n=5):
    print(f"\n{label}: worst {n} waves")
    for _, row in df_wave.nlargest(n, "score").iterrows():
        print(
            int(row["ensemble_member"]),
            "sigma =", row["sigma"],
            "A =", row["A"],
            "center =",
            np.array([row["center_x"], row["center_y"], row["center_z"]]),
            "score =", row["score"],
        )

    print(f"\n{label}: best {n} waves")
    for _, row in df_wave.nsmallest(n, "score").iterrows():
        print(
            int(row["ensemble_member"]),
            "sigma =", row["sigma"],
            "A =", row["A"],
            "center =",
            np.array([row["center_x"], row["center_y"], row["center_z"]]),
            "score =", row["score"],
        )


def plot_score_vs_initial_parameters(df_wave, score_label, filename_prefix):
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

def plot_score_vs_initial_parameters_rollout_mp(
    results,
    metadata,
    mp_runs=(1, 2, 3),
    rollout_indices=(0, 18),
    metric_key="rmse",
):
    metric_label = {
        "rmse": "RMSE",
        "rel_error": "Relative energy error",
    }.get(metric_key, metric_key)

    for rollout_idx in rollout_indices:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for ax, (x_col, x_label) in zip(
            axes,
            [
                ("sigma", r"$\sigma$ [deg]"),
                ("A", "Amplitude A"),
            ],
        ):
            for mp_step in mp_runs:
                if mp_step not in results:
                    continue

                score = get_wave_scores_at_rollout(
                    results[mp_step][metric_key],
                    results[mp_step]["metadata"],
                    rollout_idx,
                )

                df_wave = build_wave_dataframe(score, metadata)

                ax.scatter(
                    df_wave[x_col],
                    df_wave["score"],
                    s=55,
                    alpha=0.65,
                    label=f"MP {mp_step}",
                )

            ax.set_xlabel(x_label)
            ax.set_ylabel(metric_label)
            ax.set_title(
                f"{metric_label} vs {x_label}, rollout {rollout_idx + 1}"
            )
            ax.grid(True)

        axes[0].legend()
        fig.tight_layout()

        save_figure(
            fig,
            f"{metric_key}_vs_parameters_rollout{rollout_idx + 1}_mp_compare.png",
        )

def plot_score_histogram(df_wave, score_label, filename):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(df_wave["score"], bins=15)
    ax.set_xlabel(score_label)
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of {score_label}")
    fig.tight_layout()
    save_figure(fig, filename)


def plot_centers_colored_by_score(df_wave, score_label, filename):
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
            str(int(df_wave.loc[idx, "ensemble_member"])),
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
    first_mp = list(results.keys())[0]

    diagnostics = [
        (
            "Relative energy error",
            "rel_error",
            first_mp,
            f"rel_energy_mp{first_mp}",
        ),
        (
            "Mean RMSE",
            "rmse",
            first_mp,
            f"rmse_mp{first_mp}",
        ),
    ]

    for score_label, metric_key, mp_step, prefix in diagnostics:
        if mp_step not in results:
            print(f"Skipping initial-condition diagnostic for MP {mp_step}.")
            continue

        score = get_wave_scores(
            results[mp_step][metric_key],
            results[mp_step]["metadata"],
        )

        df_wave = build_wave_dataframe(score, metadata)

        print_best_worst_waves(df_wave, label=f"{score_label}, MP {mp_step}")

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


def add_initial_parameters_to_samples(df_metric, df_meta, metadata):
    df = attach_metadata(df_metric, df_meta)

    ensemble_members = df["ensemble_member"].to_numpy(dtype=int)

    df["sigma"] = metadata["sigmas"][ensemble_members]
    df["A"] = metadata["amplitudes"][ensemble_members]

    return df


def plot_metric_over_rollout_by_bins(
    results,
    metadata,
    mp_step,
    metric_key,
    parameter,
    bins,
    ylabel,
    filename,
    use_log=True,
):
    df = add_initial_parameters_to_samples(
        results[mp_step][metric_key],
        results[mp_step]["metadata"],
        metadata,
    )

    rollout_cols = get_rollout_columns(df)

    df["bin"] = pd.cut(df[parameter], bins=bins, include_lowest=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    print(f"\n{metric_key.upper()} grouped by {parameter}, MP {mp_step}")

    for bin_name, group in df.groupby("bin", observed=True):
        if len(group) == 0:
            continue

        mean_over_time = group[rollout_cols].mean(axis=0)
        rollouts = np.arange(1, len(rollout_cols) + 1)

        if use_log:
            ax.semilogy(
                rollouts,
                mean_over_time,
                marker="o",
                linestyle="--",
                label=f"{bin_name}  n={len(group)}",
            )
        else:
            ax.plot(
                rollouts,
                mean_over_time,
                marker="o",
                linestyle="--",
                label=f"{bin_name}  n={len(group)}",
            )

        print(f"\nBin {bin_name}:")
        print("  number of samples:", len(group))
        print("  ensemble members:", sorted(group["ensemble_member"].unique()))

    ax.set_xlabel("Rollout")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} over rollout grouped by {parameter}, MP {mp_step}")
    ax.grid(True, which="both")
    ax.legend()

    fig.tight_layout()
    save_figure(fig, filename)


def print_best_worst_by_metric(results, metadata, mp_step, metric_key, n=10):
    score = get_wave_scores(
        results[mp_step][metric_key],
        results[mp_step]["metadata"],
    )

    df_wave = build_wave_dataframe(score, metadata)

    print_best_worst_waves(
        df_wave,
        label=f"{metric_key}, MP {mp_step}",
        n=n,
    )


def analyze_sigma_amplitude_ranges(results, metadata, mp_step):
    sigmas = metadata["sigmas"]
    amplitudes = metadata["amplitudes"]

    sigma_bins = np.linspace(np.nanmin(sigmas), np.nanmax(sigmas), 5)
    amplitude_bins = np.linspace(np.nanmin(amplitudes), np.nanmax(amplitudes), 5)

    plot_metric_over_rollout_by_bins(
        results,
        metadata,
        mp_step=mp_step,
        metric_key="rmse",
        parameter="sigma",
        bins=sigma_bins,
        ylabel="RMSE",
        filename=f"rmse_over_time_by_sigma_mp{mp_step}.png",
        use_log=True,
    )

    plot_metric_over_rollout_by_bins(
        results,
        metadata,
        mp_step=mp_step,
        metric_key="rmse",
        parameter="A",
        bins=amplitude_bins,
        ylabel="RMSE",
        filename=f"rmse_over_time_by_amplitude_mp{mp_step}.png",
        use_log=True,
    )

    plot_metric_over_rollout_by_bins(
        results,
        metadata,
        mp_step=mp_step,
        metric_key="rel_error",
        parameter="sigma",
        bins=sigma_bins,
        ylabel="Relative energy error",
        filename=f"rel_energy_error_over_time_by_sigma_mp{mp_step}.png",
        use_log=True,
    )

    plot_metric_over_rollout_by_bins(
        results,
        metadata,
        mp_step=mp_step,
        metric_key="rel_error",
        parameter="A",
        bins=amplitude_bins,
        ylabel="Relative energy error",
        filename=f"rel_energy_error_over_time_by_amplitude_mp{mp_step}.png",
        use_log=True,
    )

    print_best_worst_by_metric(results, metadata, mp_step, "rmse", n=10)
    print_best_worst_by_metric(results, metadata, mp_step, "rel_error", n=10)


# ---------------------------------------------------------------------
# Selected wave plots
# ---------------------------------------------------------------------

def plot_relative_energy_for_selected_waves(
    results,
    mp_step,
    ensemble_members,
    filename=None,
):
    df = attach_metadata(
        results[mp_step]["rel_error"],
        results[mp_step]["metadata"],
    )

    rollout_cols = get_rollout_columns(df)

    fig, ax = plt.subplots(figsize=(10, 6))

    for member in ensemble_members:
        group = df[df["ensemble_member"] == member]

        if len(group) == 0:
            print(f"No samples found for ensemble member {member}")
            continue

        mean_curve = group[rollout_cols].mean(axis=0)

        ax.loglog(
            np.arange(1, len(rollout_cols) + 1),
            mean_curve,
            marker="o",
            linestyle="--",
            label=f"wave {member}",
        )

    ax.set_xlabel("Rollout")
    ax.set_ylabel("Relative energy error")
    ax.set_title(f"Mean relative energy error per selected wave, MP {mp_step}")
    ax.grid(True, which="both")
    ax.legend()

    fig.tight_layout()

    if filename is None:
        filename = f"selected_waves_relative_energy_mp{mp_step}.png"

    save_figure(fig, filename)


def plot_rmse_for_selected_waves(
    results,
    mp_step,
    ensemble_members,
    filename=None,
):
    df = attach_metadata(
        results[mp_step]["rmse"],
        results[mp_step]["metadata"],
    )

    rollout_cols = get_rollout_columns(df)

    fig, ax = plt.subplots(figsize=(10, 6))

    for member in ensemble_members:
        group = df[df["ensemble_member"] == member]

        if len(group) == 0:
            print(f"No samples found for ensemble member {member}")
            continue

        mean_curve = group[rollout_cols].mean(axis=0)

        ax.loglog(
            np.arange(1, len(rollout_cols) + 1),
            mean_curve,
            marker="o",
            linestyle="--",
            label=f"wave {member}",
        )

    ax.set_xlabel("Rollout")
    ax.set_ylabel("RMSE")
    ax.set_title(f"Mean RMSE per selected wave, MP {mp_step}")
    ax.grid(True, which="both")
    ax.legend()

    fig.tight_layout()

    if filename is None:
        filename = f"selected_waves_rmse_mp{mp_step}.png"

    save_figure(fig, filename)


def compare_best_worst_groups_over_rollout(
    results,
    best_mp_step,
    worst_mp_step,
    metric_key="rmse",
    n=5,
    filename=None,
):
    best_scores = get_wave_scores(
        results[best_mp_step][metric_key],
        results[best_mp_step]["metadata"],
    )

    worst_scores = get_wave_scores(
        results[worst_mp_step][metric_key],
        results[worst_mp_step]["metadata"],
    )

    best_members = best_scores.nsmallest(n).index.to_numpy(dtype=int)
    worst_members = worst_scores.nlargest(n).index.to_numpy(dtype=int)

    print(f"\nBest {n} waves from MP {best_mp_step}:")
    print(best_members)

    print(f"\nWorst {n} waves from MP {worst_mp_step}:")
    print(worst_members)

    groups = {
        f"Best {n} from MP {best_mp_step}": (best_mp_step, best_members),
        f"Worst {n} from MP {worst_mp_step}": (worst_mp_step, worst_members),
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for label, (mp_step, members) in groups.items():
        df = attach_metadata(
            results[mp_step][metric_key],
            results[mp_step]["metadata"],
        )

        rollout_cols = get_rollout_columns(df)
        group = df[df["ensemble_member"].isin(members)]
        mean_curve = group[rollout_cols].mean(axis=0)

        ax.semilogy(
            np.arange(1, len(rollout_cols) + 1),
            mean_curve,
            marker="o",
            linestyle="--",
            label=label,
        )

    metric_label = {
        "rmse": "RMSE",
        "rel_error": "Relative energy error",
    }.get(metric_key, metric_key)

    ax.set_xlabel("Rollout")
    ax.set_ylabel(metric_label)
    ax.set_title(
        f"{metric_label}: best {n} from MP {best_mp_step} "
        f"vs worst {n} from MP {worst_mp_step}"
    )
    ax.grid(True, which="both")
    ax.legend()

    fig.tight_layout()

    if filename is None:
        filename = (
            f"{metric_key}_best{n}_mp{best_mp_step}"
            f"_vs_worst{n}_mp{worst_mp_step}.png"
        )

    save_figure(fig, filename)


def plot_best_from_one_worst_from_another_over_rollout(
    results,
    best_mp_step,
    worst_mp_step,
    plot_mp_steps,
    metric_key="rmse",
    n=5,
    filename=None,
):
    best_scores = get_wave_scores(
        results[best_mp_step][metric_key],
        results[best_mp_step]["metadata"],
    )

    best_members = best_scores.nsmallest(n).index.to_numpy(dtype=int)

    worst_scores = get_wave_scores(
        results[worst_mp_step][metric_key],
        results[worst_mp_step]["metadata"],
    )

    worst_members = worst_scores.nlargest(n).index.to_numpy(dtype=int)

    print(f"\nBest {n} waves from MP {best_mp_step}:")
    print(best_members)

    print(f"\nWorst {n} waves from MP {worst_mp_step}:")
    print(worst_members)

    selected_groups = {
        f"Best {n} from MP {best_mp_step}": best_members,
        f"Worst {n} from MP {worst_mp_step}": worst_members,
    }

    fig, axes = plt.subplots(
        1,
        len(selected_groups),
        figsize=(7 * len(selected_groups), 5),
        sharey=True,
    )

    if len(selected_groups) == 1:
        axes = [axes]

    for ax, (group_label, ensemble_members) in zip(axes, selected_groups.items()):

        for mp_step in plot_mp_steps:
            if mp_step not in results:
                print(f"Skipping MP {mp_step}: not available.")
                continue

            df = attach_metadata(
                results[mp_step][metric_key],
                results[mp_step]["metadata"],
            )

            rollout_cols = get_rollout_columns(df)

            group = df[df["ensemble_member"].isin(ensemble_members)]

            if len(group) == 0:
                print(
                    f"No matching samples for {group_label} "
                    f"in MP {mp_step}"
                )
                continue

            mean_curve = group[rollout_cols].mean(axis=0)

            ax.loglog(
                np.arange(1, len(rollout_cols) + 1),
                mean_curve,
                marker="o",
                linestyle="--",
                label=f"MP {mp_step}",
            )

        ax.set_title(group_label)
        ax.set_xlabel("Rollout")
        ax.grid(True, which="both")
        ax.legend()

    metric_label = {
        "rmse": "RMSE",
        "rel_error": "Relative energy error",
    }.get(metric_key, metric_key)

    axes[0].set_ylabel(metric_label)

    fig.suptitle(
        f"{metric_label} over rollout for selected best/worst waves",
        fontsize=16,
    )

    fig.tight_layout()

    if filename is None:
        filename = (
            f"{metric_key}_best_mp{best_mp_step}"
            f"_worst_mp{worst_mp_step}.png"
        )

    save_figure(fig, filename)

def plot_mean_relative_energy_improvement_over_rollout(results):
    rel_waves = {
        mp: compute_per_wave_metric(
            results[mp]["rel_error"],
            results[mp]["metadata"],
        )
        for mp in results
    }

    mp_steps = list(results.keys())

    pairs = [
        (mp_steps[i], mp_steps[i + 1])
        for i in range(len(mp_steps) - 1)
    ]

    pairs.append((mp_steps[0],mp_steps[2]))

    if len(pairs) == 0:
        print("Skipping mean relative energy improvement plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    for a, b in pairs:
        common_members = rel_waves[a].index.intersection(rel_waves[b].index)

        if len(common_members) == 0:
            continue

        diff = (
            rel_waves[a].loc[common_members].values
            - rel_waves[b].loc[common_members].values
        )

        mean_improvement = diff.mean(axis=0)

        ax.plot(
            mean_improvement,
            label=f"MP {a} → MP {b}",
            marker="o",
        )

    ax.set_xlabel("Rollout")
    ax.set_ylabel("Mean relative energy error improvement")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()

    save_figure(fig, "mean_relative_energy_improvement_mp.png")

def plot_directional_slices_compare_mp(
    results,
    ds_data,
    mp_runs,
    base_dir,
    sample_idx=0,
    rollout_indices=(0, 9, 19),
    slice_width=0.04,
    filename=None,
):
    mp_steps = [mp for mp in mp_runs.keys() if mp in results]

    if len(mp_steps) == 0:
        print("Skipping directional slice plot: no MP results available.")
        return

    reference_mp = mp_steps[0]

    ensemble_member = int(
        results[reference_mp]["metadata"].loc[sample_idx, "ensemble_member"]
    )

    xyz = np.stack(
        [
            ds_data["x_static"].values,
            ds_data["y_static"].values,
            ds_data["z_static"].values,
        ],
        axis=1,
    )

    center = ds_data["center"].isel(
        ensemble_member=ensemble_member
    ).values
    center = center / np.linalg.norm(center)

    reference = np.array([0.0, 0.0, 1.0])
    if np.abs(np.dot(center, reference)) > 0.95:
        reference = np.array([0.0, 1.0, 0.0])

    direction = reference - np.dot(reference, center) * center
    direction = direction / np.linalg.norm(direction)

    normal = np.cross(center, direction)
    normal = normal / np.linalg.norm(normal)

    distance_to_plane = np.abs(xyz @ normal)
    mask = distance_to_plane < slice_width

    if np.sum(mask) < 10:
        print(
            f"Warning: only {np.sum(mask)} nodes in slice. "
            "Consider increasing slice_width."
        )

    xyz_slice = xyz[mask]

    signed_angle = np.arctan2(
        xyz_slice @ direction,
        xyz_slice @ center,
    )

    order = np.argsort(signed_angle)
    signed_angle = signed_angle[order]
    slice_node_indices = np.where(mask)[0][order]

    fig, axes = plt.subplots(
        len(rollout_indices),
        len(mp_steps),
        figsize=(5.2 * len(mp_steps), 3.6 * len(rollout_indices)),
        sharex=True,
        sharey=True,
    )

    if len(rollout_indices) == 1:
        axes = np.asarray([axes])

    if len(mp_steps) == 1:
        axes = axes[:, None]

    for col, mp_step in enumerate(mp_steps):
        pred_path = Path(base_dir) /  f"test_mp{mp_step}_results_new.zarr"

        ds_pred = xr.open_zarr(pred_path)

        sample_ensemble_member = int(
            results[mp_step]["metadata"].loc[sample_idx, "ensemble_member"]
        )

        if sample_ensemble_member != ensemble_member:
            print(
                f"Warning: sample {sample_idx} has ensemble member "
                f"{sample_ensemble_member} for MP {mp_step}, but "
                f"{ensemble_member} for MP {reference_mp}."
            )

        for row, rollout_idx in enumerate(rollout_indices):
            ax = axes[row, col]

            target = ds_pred["target"].isel(
                sample=sample_idx,
                rollout_step=rollout_idx,
                state_feature=0,
            ).values

            pred = ds_pred["prediction"].isel(
                sample=sample_idx,
                rollout_step=rollout_idx,
                state_feature=0,
            ).values

            target_slice = target[slice_node_indices]
            pred_slice = pred[slice_node_indices]

            ax.plot(
                signed_angle,
                target_slice,
                linewidth=2.0,
                label="Target",
            )

            ax.plot(
                signed_angle,
                pred_slice,
                linewidth=2.0,
                linestyle="--",
                label="Prediction",
            )

            ax.axvline(0.0, color="black", linewidth=1, alpha=0.5)
            ax.grid(True)

            if row == 0:
                ax.set_title(f"MP {mp_step}")

            if col == 0:
                ax.set_ylabel(f"Rollout {rollout_idx + 1}\n" + r"$u$")

            if row == len(rollout_indices) - 1:
                ax.set_xlabel("Signed angle [rad]")

    axes[0, 0].legend()

    fig.suptitle(
        f"Directional target/prediction slices across message-passing depth "
        f"(wave {ensemble_member})",
        fontsize=16,
    )

    fig.tight_layout()

    if filename is None:
        filename = (
            f"directional_slices_compare_mp_sample{sample_idx}.png"
        )

    save_figure(fig, filename)

def plot_best_worst_rmse_directional_slices_compare_mp(
    results,
    ds_data,
    mp_runs,
    base_dir,
    reference_mp=None,
    rollout_indices=(0, 9, 19),
    slice_width=0.04,
    filename=None,
):
    mp_steps = [mp for mp in mp_runs.keys() if mp in results]

    if len(mp_steps) == 0:
        print("Skipping best/worst directional slice plot: no MP results available.")
        return

    if reference_mp is None:
        reference_mp = mp_steps[-1]

    if reference_mp not in results:
        raise ValueError(f"reference_mp={reference_mp} is not available.")

    rmse_scores = get_wave_scores(
        results[reference_mp]["rmse"],
        results[reference_mp]["metadata"],
    )

    best_member = int(rmse_scores.idxmin())
    worst_member = int(rmse_scores.idxmax())

    print(f"\nBest RMSE wave from MP {reference_mp}: {best_member}")
    print(f"Worst RMSE wave from MP {reference_mp}: {worst_member}")

    selected_waves = {
        "Best RMSE": best_member,
        "Worst RMSE": worst_member,
    }

    ds_preds = {}
    for mp_step in mp_steps:
        pred_path = Path(base_dir) / f"test_mp{mp_step}_results_new.zarr"

        if not pred_path.exists():
            print(f"Skipping MP {mp_step}: prediction Zarr not found: {pred_path}")
            continue

        ds_preds[mp_step] = xr.open_zarr(pred_path)

    available_mp_steps = list(ds_preds.keys())

    if len(available_mp_steps) == 0:
        print("No prediction Zarr files found.")
        return

    xyz = np.stack(
        [
            ds_data["x_static"].values,
            ds_data["y_static"].values,
            ds_data["z_static"].values,
        ],
        axis=1,
    )

    n_rows = len(selected_waves) * len(rollout_indices)
    n_cols = len(available_mp_steps)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.2 * n_cols, 3.3 * n_rows),
        sharex=True,
        sharey=True,
    )

    if n_cols == 1:
        axes = axes[:, None]

    row_idx = 0

    for wave_label, ensemble_member in selected_waves.items():

        center = ds_data["center"].isel(
            ensemble_member=ensemble_member
        ).values
        center = center / np.linalg.norm(center)

        reference = np.array([0.0, 0.0, 1.0])
        if np.abs(np.dot(center, reference)) > 0.95:
            reference = np.array([0.0, 1.0, 0.0])

        direction = reference - np.dot(reference, center) * center
        direction = direction / np.linalg.norm(direction)

        normal = np.cross(center, direction)
        normal = normal / np.linalg.norm(normal)

        distance_to_plane = np.abs(xyz @ normal)
        mask = distance_to_plane < slice_width

        if np.sum(mask) < 10:
            print(
                f"Warning: only {np.sum(mask)} nodes in slice for wave {ensemble_member}. "
                "Consider increasing slice_width."
            )

        xyz_slice = xyz[mask]

        signed_angle = np.arctan2(
            xyz_slice @ direction,
            xyz_slice @ center,
        )

        order = np.argsort(signed_angle)
        signed_angle = signed_angle[order]
        slice_node_indices = np.where(mask)[0][order]

        for rollout_idx in rollout_indices:
            for col, mp_step in enumerate(available_mp_steps):
                ax = axes[row_idx, col]

                df_meta = results[mp_step]["metadata"]

                matching_samples = df_meta.index[
                    df_meta["ensemble_member"] == ensemble_member
                ].to_numpy()

                if len(matching_samples) == 0:
                    ax.set_title(f"MP {mp_step}: missing wave")
                    ax.axis("off")
                    continue

                sample_idx = int(matching_samples[0])

                ds_pred = ds_preds[mp_step]

                target = ds_pred["target"].isel(
                    sample=sample_idx,
                    rollout_step=rollout_idx,
                    state_feature=0,
                ).values

                pred = ds_pred["prediction"].isel(
                    sample=sample_idx,
                    rollout_step=rollout_idx,
                    state_feature=0,
                ).values

                target_slice = target[slice_node_indices]
                pred_slice = pred[slice_node_indices]

                ax.plot(
                    signed_angle,
                    target_slice,
                    linewidth=2.0,
                    label="Target",
                )

                ax.plot(
                    signed_angle,
                    pred_slice,
                    linewidth=2.0,
                    linestyle="--",
                    label="Prediction",
                )

                ax.axvline(0.0, color="black", linewidth=1, alpha=0.5)
                ax.grid(True)

                if row_idx == 0:
                    ax.set_title(f"MP {mp_step}")

                if col == 0:
                    ax.set_ylabel(
                        f"{wave_label}\nwave {ensemble_member}\n"
                        f"rollout {rollout_idx + 1}\n" + r"$u$"
                    )

                if row_idx == n_rows - 1:
                    ax.set_xlabel("Signed angle [rad]")

            row_idx += 1

    axes[0, 0].legend()

    fig.suptitle(
        f"Best and worst RMSE wave slices across message-passing depth "
        f"(selected using MP {reference_mp})",
        fontsize=16,
    )

    fig.tight_layout()

    if filename is None:
        filename = (
            f"best_worst_rmse_directional_slices_compare_mp"
            f"_refmp{reference_mp}.png"
        )

    save_figure(fig, filename)


def plot_best_worst_metric_directional_slices_compare_mp(
    results,
    ds_data,
    mp_runs,
    base_dir,
    metric_key="rmse",
    reference_mp=None,
    rollout_indices=(0, 9, 19),
    slice_width=0.04,
    filename=None,
    sample_position_within_wave=0,
):
    mp_steps = [mp for mp in mp_runs.keys() if mp in results]

    if len(mp_steps) == 0:
        print("Skipping best/worst directional slice plot: no MP results available.")
        return

    if reference_mp is None:
        reference_mp = mp_steps[-1]

    if reference_mp not in results:
        raise ValueError(f"reference_mp={reference_mp} is not available.")

    metric_scores = get_wave_scores(
        results[reference_mp][metric_key],
        results[reference_mp]["metadata"],
    )

    best_member = int(metric_scores.idxmin())
    worst_member = int(metric_scores.idxmax())

    metric_label = {
        "rmse": "RMSE",
        "rel_error": "relative energy error",
    }.get(metric_key, metric_key)


    selected_waves = {
        f"Best {metric_label}": best_member,
        f"Worst {metric_label}": worst_member,
    }

    ds_preds = {}
    for mp_step in mp_steps:
        pred_path = Path(base_dir) / f"test_mp{mp_step}_results_new.zarr"

        if not pred_path.exists():
            print(f"Skipping MP {mp_step}: prediction Zarr not found: {pred_path}")
            continue

        ds_preds[mp_step] = xr.open_zarr(pred_path)

    available_mp_steps = list(ds_preds.keys())

    if len(available_mp_steps) == 0:
        print("No prediction Zarr files found.")
        return

    xyz = np.stack(
        [
            ds_data["x_static"].values,
            ds_data["y_static"].values,
            ds_data["z_static"].values,
        ],
        axis=1,
    )

    n_rows = len(selected_waves) * len(rollout_indices)
    n_cols = len(available_mp_steps)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.2 * n_cols, 3.3 * n_rows),
        sharex=True,
        sharey=True,
    )

    if n_cols == 1:
        axes = axes[:, None]

    row_idx = 0

    for wave_label, ensemble_member in selected_waves.items():

        center = ds_data["center"].isel(
            ensemble_member=ensemble_member
        ).values
        center = center / np.linalg.norm(center)

        reference = np.array([0.0, 0.0, 1.0])
        if np.abs(np.dot(center, reference)) > 0.95:
            reference = np.array([0.0, 1.0, 0.0])

        direction = reference - np.dot(reference, center) * center
        direction = direction / np.linalg.norm(direction)

        normal = np.cross(center, direction)
        normal = normal / np.linalg.norm(normal)

        distance_to_plane = np.abs(xyz @ normal)
        mask = distance_to_plane < slice_width

        if np.sum(mask) < 10:
            print(
                f"Warning: only {np.sum(mask)} nodes in slice for wave {ensemble_member}. "
                "Consider increasing slice_width."
            )

        xyz_slice = xyz[mask]

        signed_angle = np.arctan2(
            xyz_slice @ direction,
            xyz_slice @ center,
        )

        order = np.argsort(signed_angle)
        signed_angle = signed_angle[order]
        slice_node_indices = np.where(mask)[0][order]

        for rollout_idx in rollout_indices:
            for col, mp_step in enumerate(available_mp_steps):
                ax = axes[row_idx, col]

                df_meta = results[mp_step]["metadata"]

                matching_samples = np.where(
                    df_meta["ensemble_member"].to_numpy() == ensemble_member
                )[0]

                if len(matching_samples) == 0:
                    ax.set_title(f"MP {mp_step}: missing wave")
                    ax.axis("off")
                    continue

                if sample_position_within_wave >= len(matching_samples):
                    sample_idx = int(matching_samples[-1])
                else:
                    sample_idx = int(matching_samples[sample_position_within_wave])
                ds_pred = ds_preds[mp_step]

                target = ds_pred["target"].isel(
                    sample=sample_idx,
                    rollout_step=rollout_idx,
                    state_feature=0,
                ).values

                pred = ds_pred["prediction"].isel(
                    sample=sample_idx,
                    rollout_step=rollout_idx,
                    state_feature=0,
                ).values

                target_slice = target[slice_node_indices]
                pred_slice = pred[slice_node_indices]

                ax.plot(
                    signed_angle,
                    target_slice,
                    linewidth=2.0,
                    label="$u_t$",
                )

                ax.plot(
                    signed_angle,
                    pred_slice,
                    linewidth=2.0,
                    linestyle="--",
                    label="$\hat u_t$",
                )

                ax.axvline(0.0, color="black", linewidth=1, alpha=0.5)
                ax.grid(True)

                if row_idx == 0:
                    ax.set_title(f"MP {mp_step}",fontsize=20)

                if col == 0:
                    ax.set_ylabel(
                        f"{wave_label}\n"
                        f"rollout {rollout_idx + 1}\n" + r"$u$",fontsize=20
                    )

                if row_idx == n_rows - 1:
                    ax.set_xlabel("Signed angle [rad]",fontsize=20)

            row_idx += 1

    axes[0, 0].legend(fontsize=20)

    fig.suptitle(
        f"Best and worst {metric_label} wave slices across message-passing depth"
        f" (based on MP {reference_mp})",
        fontsize=22,
    )

    fig.tight_layout(pad=2.0)

    if filename is None:
        filename = (
            f"best_worst_{metric_key}_directional_slices_compare_mp"
            f"_refmp{reference_mp}_rolloutstart{sample_position_within_wave}.png"
        )

    save_figure(fig, filename)

def plot_error_improvement_ratio_vs_parameter(
    results,
    metadata,
    metric_key="rmse",
    mp_base=1,
    mp_compare=3,
    rollout_indices=(0, 17),
    parameter="sigma",
):
    x_label = {
        "sigma": r"$\sigma$ [deg]",
        "A": "Amplitude A",
    }[parameter]

    fig, axes = plt.subplots(
        1,
        len(rollout_indices),
        figsize=(7 * len(rollout_indices), 5),
        squeeze=False,
    )

    for ax, rollout_idx in zip(axes[0], rollout_indices):

        base_score = get_wave_scores_at_rollout(
            results[mp_base][metric_key],
            results[mp_base]["metadata"],
            rollout_idx,
        )

        compare_score = get_wave_scores_at_rollout(
            results[mp_compare][metric_key],
            results[mp_compare]["metadata"],
            rollout_idx,
        )

        common_members = (
            base_score.index.intersection(compare_score.index)
        )

        ratio = base_score.loc[common_members] / (
            compare_score.loc[common_members] + 1e-12
        )

        df = pd.DataFrame({
            "ensemble_member": common_members.to_numpy(dtype=int),
            "ratio": ratio.values,
            "sigma": metadata["sigmas"][common_members],
            "A": metadata["amplitudes"][common_members],
        })

        ax.scatter(
            df[parameter],
            df["ratio"],
            s=70,
            alpha=0.75,
            edgecolor="black",
            linewidth=0.3,
        )

        ax.axhline(
            1.0,
            color="black",
            linestyle="--",
            linewidth=1.5,
        )

        ax.set_xlabel(x_label)
        ax.set_ylabel(
            f"{metric_key.upper()} MP{mp_base} / MP{mp_compare}"
        )

        ax.set_title(
            f"Rollout {rollout_idx + 1}"
        )

        ax.grid(True)

    fig.suptitle(
        f"Improvement ratio MP{mp_base} vs MP{mp_compare}"
    )

    fig.tight_layout()

    save_figure(
        fig,
        f"{metric_key}_improvement_ratio_mp{mp_base}_mp{mp_compare}"
        f"_vs_{parameter}.png",
    )

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    results = load_mp_results(BASE_DIR, MP_RUNS)

    if len(results) == 0:
        raise RuntimeError("No MP result folders were loaded. Check BASE_DIR and MP_RUNS.")

    metadata = load_initial_condition_metadata(DATASET_PATH)

    first_mp = list(results.keys())[0]
    last_mp = list(results.keys())[-1]

    # plot_sample_heatmaps(results)

    # plot_energy_drift_and_error(results)

    # plot_metric_over_rollout(
    #     results,
    #     metric_name="rmse",
    #     ylabel="RMSE",
    #     filename="rmse_over_time_mp.png",
    #     semilogy=False,
    # )

    # plot_one_step_heatmap(
    #     results,
    #     metric_name="rel_error",
    #     title="One-step relative energy error for different message-passing steps",
    #     colorbar_label="Relative energy error",
    #     filename="one_step_heatmaps_rel_energy_mp.png",
    #     norm=REL_ENERGY_NORM,
    # )

    # plot_one_step_heatmap(
    #     results,
    #     metric_name="rmse",
    #     title="One-step RMSE for different message-passing steps",
    #     colorbar_label="RMSE",
    #     filename="one_step_heatmaps_rmse_mp.png",
    #     norm=RMSE_NORM,
    # )

    # plot_per_wave_heatmaps(results)

    # plot_per_wave_score_boxplots(results, rollout_indices=(0, 9, 17))

    # plot_median_iqr_vs_mp_steps(results, rollout_indices=(0, 9, 17))

    # plot_improvement_heatmaps(results)

    # plot_mean_improvement_over_rollout(results)
    
    # plot_mean_relative_energy_improvement_over_rollout(results)

    # analyze_initial_conditions(results, metadata)

    # analyze_sigma_amplitude_ranges(
    #     results,
    #     metadata,
    #     mp_step=first_mp,
    # )

    # plot_best_worst_parameter_stripplot(
    #     results,
    #     metadata,
    #     mp_steps=list(results.keys()),
    #     rollout_indices=(0, 9, 17),
    # )   
    
    # plot_relative_energy_for_selected_waves(
    #     results,
    #     mp_step=first_mp,
    #     ensemble_members=[0],
    # )

    # plot_rmse_for_selected_waves(
    #     results,
    #     mp_step=first_mp,
    #     ensemble_members=[0],
    # )

    # plot_best_from_one_worst_from_another_over_rollout(
    #     results,
    #     best_mp_step=last_mp,
    #     worst_mp_step=first_mp,
    #     plot_mp_steps=list(results.keys()),
    #     metric_key="rmse",
    #     n=5,
    # )

    # plot_best_from_one_worst_from_another_over_rollout(
    #     results,
    #     best_mp_step=last_mp,
    #     worst_mp_step=first_mp,
    #     plot_mp_steps=list(results.keys()),
    #     metric_key="rel_error",
    #     n=5,
    # )

    # compare_best_worst_groups_over_rollout(
    #     results,
    #     best_mp_step=last_mp,
    #     worst_mp_step=first_mp,
    #     metric_key="rmse",
    #     n=5,
    # )

    # compare_best_worst_groups_over_rollout(
    #     results,
    #     best_mp_step=last_mp,
    #     worst_mp_step=first_mp,
    #     metric_key="rel_error",
    #     n=5,
    # )

    ds_data = xr.open_dataset(DATASET_PATH)

    # plot_directional_slices_compare_mp(
    #     results=results,
    #     ds_data=ds_data,
    #     mp_runs=MP_RUNS,
    #     base_dir=BASE_DIR,
    #     sample_idx=0,
    #     rollout_indices=(0, 9, 19),
    #     slice_width=0.04,
    # )

    plot_best_worst_metric_directional_slices_compare_mp(
        results=results,
        ds_data=ds_data,
        mp_runs=MP_RUNS,
        base_dir=BASE_DIR,
        metric_key="rel_error",
        reference_mp=last_mp,
        rollout_indices=(0, 9, 19),
        slice_width=0.04,
        sample_position_within_wave=0
    )

    plot_best_worst_metric_directional_slices_compare_mp(
        results=results,
        ds_data=ds_data,
        mp_runs=MP_RUNS,
        base_dir=BASE_DIR,
        metric_key="rmse",
        reference_mp=last_mp,
        rollout_indices=(0, 19),
        slice_width=0.04,
        sample_position_within_wave=0
    )


if __name__ == "__main__":
    main()
