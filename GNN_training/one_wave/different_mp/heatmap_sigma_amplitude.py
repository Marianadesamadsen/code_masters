import os
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


BASE_DIR = "GNN_training/one_wave/different_mp"
DATASET_PATH = "GNN_training/one_wave/nc_files/wave_200_ts_600_g4_sigmamin_6.nc"

MP_STEPS = [1, 2, 3]
METRIC_KEY = "rel_error"


def load_results(base_dir, mp_steps):
    results = {}

    for mp_step in mp_steps:
        result_dir = os.path.join(
            base_dir,
            f"test_mp{mp_step}_results_new",
        )

        metric_path = os.path.join(
            result_dir,
            f"test_{METRIC_KEY}_per_sample.csv",
        )

        metadata_path = os.path.join(
            result_dir,
            "test_metadata.csv",
        )

        results[mp_step] = {
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
    mp_steps,
    metric_key="rmse",
):
    all_rows = []

    for mp_step in mp_steps:
        metric_df = results[mp_step][metric_key]
        meta_df = results[mp_step]["metadata"]

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
                "mp_step": mp_step,
                "ensemble_member": ensemble_member,
                "sigma_deg": metadata["sigmas"][ensemble_member],
                "amplitude_A": metadata["amplitudes"][ensemble_member],
                f"mean_{metric_key}": score,
            })

    return pd.DataFrame(all_rows)


def plot_sigma_amplitude_metric_scatter_mp(df_scores, metric_key="rmse"):
    mp_steps = sorted(df_scores["mp_step"].unique())
    value_col = f"mean_{metric_key}"

    vmin = df_scores[value_col].min()
    vmax = df_scores[value_col].max()

    fig, axes = plt.subplots(
        1,
        len(mp_steps),
        figsize=(6 * len(mp_steps), 5),
        sharex=True,
        sharey=True,
    )

    if len(mp_steps) == 1:
        axes = [axes]

    sc = None

    for ax, mp_step in zip(axes, mp_steps):
        df_sub = df_scores[df_scores["mp_step"] == mp_step]

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

        ax.set_title(f"MP {mp_step}")
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
        f"{metric_key}_sigma_amplitude_scatter_clean_mp.png",
    )

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved figure to: {out_path}")


if __name__ == "__main__":
    results = load_results(BASE_DIR, MP_STEPS)

    metadata = load_initial_condition_metadata(DATASET_PATH)

    df_scores = compute_per_wave_scores(
        results=results,
        metadata=metadata,
        mp_steps=MP_STEPS,
        metric_key=METRIC_KEY,
    )

    print(df_scores)

    plot_sigma_amplitude_metric_scatter_mp(
        df_scores=df_scores,
        metric_key=METRIC_KEY,
    )