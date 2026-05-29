import os
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


BASE_DIR = "GNN_training/one_wave/different_training_size"
DATASET_PATH = "GNN_training/one_wave/nc_files/wave_200_ts_600_g4_sigmamin_6.nc"

TRAINING_SIZES = [1, 25, 50, 75]
METRIC_KEY = "energy_rel_error"

import numpy as np
from matplotlib.colors import LogNorm


from scipy.interpolate import griddata
from matplotlib.colors import LogNorm
import numpy as np


def load_results(base_dir, training_sizes):
    results = {}

    for train_size in training_sizes:
        result_dir = os.path.join(
            base_dir,
            f"test_{train_size}_results_new",
        )

        metric_path = os.path.join(
            result_dir,
            f"test_{METRIC_KEY}_per_sample.csv",
        )

        metadata_path = os.path.join(
            result_dir,
            "test_metadata.csv",
        )

        results[train_size] = {
            METRIC_KEY: pd.read_csv(metric_path),
            "metadata": pd.read_csv(metadata_path),
        }

    return results


def load_initial_condition_metadata(dataset_path):
    ds = xr.open_dataset(dataset_path)

    return {
        "sigmas": ds["sigma_deg"].values,
        "amplitudes": ds["A"].values,
    }


def compute_per_wave_scores(
    results,
    metadata,
    training_sizes,
    metric_key="rmse",
):
    all_rows = []

    for train_size in training_sizes:
        metric_df = results[train_size][metric_key]
        meta_df = results[train_size]["metadata"]

        df = metric_df.copy()
        df["ensemble_member"] = meta_df["ensemble_member"].values

        rollout_cols = [
            c for c in metric_df.columns
            if c.startswith("rollout_")
        ]

        per_wave_scores = (
            df.groupby("ensemble_member")[rollout_cols]
            .mean()
            .mean(axis=1)
        )

        for ensemble_member, score in per_wave_scores.items():
            ensemble_member = int(ensemble_member)

            all_rows.append({
                "train_size": train_size,
                "ensemble_member": ensemble_member,
                "sigma_deg": metadata["sigmas"][ensemble_member],
                "amplitude_A": metadata["amplitudes"][ensemble_member],
                f"mean_{metric_key}": score,
            })

    return pd.DataFrame(all_rows)


def plot_sigma_amplitude_rmse_scatter_clean(df_scores, metric_key="rmse"):
    from matplotlib.colors import LogNorm

    train_sizes = sorted(df_scores["train_size"].unique())
    value_col = f"mean_{metric_key}"

    vmin = df_scores[value_col].min()
    vmax = df_scores[value_col].max()

    fig, axes = plt.subplots(
        1,
        len(train_sizes),
        figsize=(6 * len(train_sizes), 5),
        sharex=True,
        sharey=True,
    )

    if len(train_sizes) == 1:
        axes = [axes]

    sc = None

    for ax, train_size in zip(axes, train_sizes):
        df_sub = df_scores[df_scores["train_size"] == train_size]

        sc = ax.scatter(
            df_sub["sigma_deg"],
            df_sub["amplitude_A"],
            c=df_sub[value_col],
            cmap="viridis",
            norm=LogNorm(vmin=vmin, vmax=vmax),
            s=110,
            edgecolor="black",
            linewidth=0.6,
            alpha=0.9,
        )

        worst = df_sub.nlargest(5, value_col)
        best = df_sub.nsmallest(5, value_col)

        ax.scatter(
            worst["sigma_deg"],
            worst["amplitude_A"],
            s=220,
            facecolors="none",
            edgecolors="red",
            linewidth=2.0,
            label="Worst 5",
        )

        ax.scatter(
            best["sigma_deg"],
            best["amplitude_A"],
            s=220,
            facecolors="none",
            edgecolors="lime",
            linewidth=2.0,
            label="Best 5",
        )

        ax.set_title(f"Train size {train_size}")
        ax.set_xlabel(r"$\sigma$ [deg]")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Amplitude A")
    axes[0].legend()

    cbar = fig.colorbar(sc, ax=axes, shrink=0.9, pad=0.02)
    cbar.set_label(f"Mean {metric_key.upper()}")

    fig.suptitle(
        f"Mean {metric_key.upper()} over initial-condition parameter space",
        fontsize=16,
    )

    fig.tight_layout()

    out_path = os.path.join(
        BASE_DIR,
        f"{metric_key}_sigma_amplitude_scatter_clean.png",
    )

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved figure to: {out_path}")

def plot_training_size_improvement_parameter_space(
    df_scores,
    metric_key="rmse",
    reference_train_size=1,
):
    train_sizes = sorted(df_scores["train_size"].unique())
    value_col = f"mean_{metric_key}"

    ref_df = (
        df_scores[df_scores["train_size"] == reference_train_size]
        .set_index("ensemble_member")
    )

    rows = []

    for train_size in train_sizes:
        if train_size == reference_train_size:
            continue

        cur_df = (
            df_scores[df_scores["train_size"] == train_size]
            .set_index("ensemble_member")
        )

        common_members = ref_df.index.intersection(cur_df.index)

        for ensemble_member in common_members:
            ref_val = ref_df.loc[ensemble_member, value_col]
            cur_val = cur_df.loc[ensemble_member, value_col]

            rows.append({
                "train_size": train_size,
                "ensemble_member": ensemble_member,
                "sigma_deg": cur_df.loc[ensemble_member, "sigma_deg"],
                "amplitude_A": cur_df.loc[ensemble_member, "amplitude_A"],
                "log_improvement": np.log10(ref_val / (cur_val + 1e-12)),
                "improved": cur_val < ref_val,
            })

    df_imp = pd.DataFrame(rows)

    abs_max = np.nanmax(np.abs(df_imp["log_improvement"].values))

    norm = plt.Normalize(vmin=-abs_max, vmax=abs_max)

    fig, axes = plt.subplots(
        1,
        len(train_sizes) - 1,
        figsize=(6 * (len(train_sizes) - 1), 5),
        sharex=True,
        sharey=True,
    )

    if len(train_sizes) - 1 == 1:
        axes = [axes]

    sc = None

    for ax, train_size in zip(axes, [s for s in train_sizes if s != reference_train_size]):
        df_sub = df_imp[df_imp["train_size"] == train_size]

        sc = ax.scatter(
            df_sub["sigma_deg"],
            df_sub["amplitude_A"],
            c=df_sub["log_improvement"],
            cmap="RdBu_r",
            norm=norm,
            s=110,
            edgecolor="black",
            linewidth=0.6,
            alpha=0.9,
        )

        improved = df_sub["improved"]

        ax.scatter(
            df_sub.loc[improved, "sigma_deg"],
            df_sub.loc[improved, "amplitude_A"],
            s=180,
            facecolors="none",
            edgecolors="red",
            linewidth=1.5,
            label="Improved",
        )

        ax.set_title(
            f"Train {reference_train_size} → {train_size}",
            fontsize=14,
        )
        ax.set_xlabel(r"$\sigma$ [deg]")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Amplitude A")
    axes[0].legend()

    cbar = fig.colorbar(
        sc,
        ax=axes,
        shrink=0.9,
        pad=0.02,
    )
    cbar.set_label(
        rf"$\log_{{10}}(\mathrm{{{metric_key.upper()}}}_{{train {reference_train_size}}}"
        rf"/\mathrm{{{metric_key.upper()}}}_{{train size}})$"
    )

    fig.suptitle(
        f"Relative improvement in mean {metric_key.upper()} over initial-condition parameter space",
        fontsize=16,
    )

    fig.tight_layout()

    out_path = os.path.join(
        BASE_DIR,
        f"{metric_key}_relative_improvement_parameter_space_train{reference_train_size}.png",
    )

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved figure to: {out_path}")

if __name__ == "__main__":
    results = load_results(BASE_DIR, TRAINING_SIZES)

    metadata = load_initial_condition_metadata(
        DATASET_PATH
    )

    df_scores = compute_per_wave_scores(
        results=results,
        metadata=metadata,
        training_sizes=TRAINING_SIZES,
        metric_key=METRIC_KEY,
    )

    print(df_scores)


    # plot_sigma_amplitude_rmse_scatter_clean(
    #     df_scores=df_scores,
    #     metric_key="rmse",
    # )

    plot_training_size_improvement_parameter_space(
        df_scores=df_scores,
        metric_key="energy_rel_error",
        reference_train_size=1,
    )

    