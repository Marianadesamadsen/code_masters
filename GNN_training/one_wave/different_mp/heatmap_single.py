import os
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


BASE_DIR = "GNN_training/one_wave/different_mp"
DATASET_PATH = "GNN_training/one_wave/nc_files/wave_200_ts_600_g4_sigmamin_6.nc"

MP_STEP = 1

METRIC_KEYS = ["rmse", "energy_rel_error"]


def load_metric_result(base_dir, mp_step, metric_key):
    result_dir = os.path.join(
        base_dir,
        f"test_mp{mp_step}_results_new",
    )

    metric_path = os.path.join(
        result_dir,
        f"test_{metric_key}_per_sample.csv",
    )

    metadata_path = os.path.join(
        result_dir,
        "test_metadata.csv",
    )

    return {
        metric_key: pd.read_csv(metric_path),
        "metadata": pd.read_csv(metadata_path),
    }


def load_initial_condition_metadata(dataset_path):
    ds = xr.open_dataset(dataset_path)

    return {
        "sigmas": ds["sigma_deg"].values,
        "amplitudes": ds["A"].values,
    }


def compute_per_wave_scores(result, metadata, mp_step, metric_key):
    metric_df = result[metric_key]
    meta_df = result["metadata"]

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

    rows = []

    for ensemble_member, score in per_wave_scores.items():
        ensemble_member = int(ensemble_member)

        rows.append({
            "mp_step": mp_step,
            "ensemble_member": ensemble_member,
            "sigma_deg": metadata["sigmas"][ensemble_member],
            "amplitude_A": metadata["amplitudes"][ensemble_member],
            f"mean_{metric_key}": score,
        })

    return pd.DataFrame(rows)

def plot_mp1_metrics(df_rmse, df_rel_error):

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6),
        sharex=True,
        sharey=True,
    )

    metrics = [
        ("RMSE", df_rmse, axes[0]),
        ("Relative Energy Error", df_rel_error, axes[1]),
    ]

    for title, df, ax in metrics:

        value_col = [c for c in df.columns if c.startswith("mean_")][0]

        sc = ax.scatter(
            df["sigma_deg"],
            df["amplitude_A"],
            c=df[value_col],
            cmap="viridis",
            norm=LogNorm(
                vmin=df[value_col].min(),
                vmax=df[value_col].max(),
            ),
            s=110,
            edgecolor="black",
            linewidth=0.6,
        )

        worst = df.nlargest(10, value_col)
        best = df.nsmallest(10, value_col)

        ax.scatter(
            worst["sigma_deg"],
            worst["amplitude_A"],
            s=220,
            facecolors="none",
            edgecolors="red",
            linewidth=2,
            label="Worst 10",
        )

        ax.scatter(
            best["sigma_deg"],
            best["amplitude_A"],
            s=220,
            facecolors="none",
            edgecolors="lime",
            linewidth=2,
            label="Best 10",
        )

        ax.set_title(title, fontsize=16)
        ax.set_xlabel(r"$\sigma$ [deg]")
        ax.grid(alpha=0.3)

        cbar = fig.colorbar(
            sc,
            ax=ax,
            fraction=0.046,
            pad=0.04,
        )
        cbar.set_label(title)

    axes[0].set_ylabel("Amplitude A")
    axes[0].legend()

    plt.suptitle(
        "Performance across initial conditions",
        fontsize=18,
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            BASE_DIR,
            "mp1_rmse_relerror_sigma_amplitude.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

if __name__ == "__main__":
    metadata = load_initial_condition_metadata(DATASET_PATH)

    rmse_result = load_metric_result(
        BASE_DIR,
        MP_STEP,
        "rmse",
    )

    rel_result = load_metric_result(
        BASE_DIR,
        MP_STEP,
        "energy_rel_error",
    )

    df_rmse = compute_per_wave_scores(
        rmse_result,
        metadata,
        MP_STEP,
        "rmse",
    )

    df_rel_error = compute_per_wave_scores(
        rel_result,
        metadata,
        MP_STEP,
        "energy_rel_error",
    )

    plot_mp1_metrics(
        df_rmse,
        df_rel_error,
    )