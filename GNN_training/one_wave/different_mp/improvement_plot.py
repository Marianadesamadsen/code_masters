import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm


BASE_DIR = "GNN_training/one_wave/different_mp"
DATASET_PATH = "GNN_training/one_wave/nc_files/wave_200_ts_600_g4_sigmamin_6.nc"
PLOT_DIR = "GNN_training/one_wave/different_mp/results"

MP_BASE = 1
MP_COMPARE = 2
METRIC_KEY = "rmse"
ROLLOUT_INDICES = (0, 9, 17)


def load_initial_condition_metadata(dataset_path):
    ds = xr.open_dataset(dataset_path)
    return {
        "sigmas": ds["sigma_deg"].values,
        "amplitudes": ds["A"].values,
    }


def load_metric_for_mp(mp_step, metric_key):
    result_dir = os.path.join(BASE_DIR, f"test_mp{mp_step}_results_new")

    metric_df = pd.read_csv(
        os.path.join(result_dir, f"test_{metric_key}_per_sample.csv")
    )

    metadata_df = pd.read_csv(
        os.path.join(result_dir, "test_metadata.csv")
    )

    return metric_df, metadata_df


def per_wave_score_at_rollout(metric_df, metadata_df, rollout_idx):
    df = metric_df.copy()
    df["ensemble_member"] = metadata_df["ensemble_member"].values

    rollout_cols = [c for c in metric_df.columns if c.startswith("rollout_")]
    rollout_col = rollout_cols[rollout_idx]

    return df.groupby("ensemble_member")[rollout_col].mean()


def make_improvement_dataframe(
    metadata,
    mp_base=1,
    mp_compare=3,
    metric_key="rmse",
    rollout_indices=(0, 9, 19),
):
    metric_base, meta_base = load_metric_for_mp(mp_base, metric_key)
    metric_compare, meta_compare = load_metric_for_mp(mp_compare, metric_key)

    rows = []

    for rollout_idx in rollout_indices:
        score_base = per_wave_score_at_rollout(
            metric_base,
            meta_base,
            rollout_idx,
        )

        score_compare = per_wave_score_at_rollout(
            metric_compare,
            meta_compare,
            rollout_idx,
        )

        common_members = score_base.index.intersection(score_compare.index)

        for ensemble_member in common_members:
            ensemble_member = int(ensemble_member)

            base_val = float(score_base.loc[ensemble_member])
            compare_val = float(score_compare.loc[ensemble_member])

            ratio = np.log10(base_val / (compare_val + 1e-12))
            diff = base_val - compare_val

            rows.append({
                "rollout_idx": rollout_idx,
                "rollout": rollout_idx + 1,
                "ensemble_member": ensemble_member,
                f"mp{mp_base}_{metric_key}": base_val,
                f"mp{mp_compare}_{metric_key}": compare_val,
                "improvement_ratio": ratio,
                "improvement_difference": diff,
                "sigma_deg": metadata["sigmas"][ensemble_member],
                "amplitude_A": metadata["amplitudes"][ensemble_member],
            })

    return pd.DataFrame(rows)

def plot_mp_improvement_thesis_figure(
    df,
    mp_base=1,
    mp_compare=3,
    metric_key="rmse",
    rollout_indices=(0, 9, 19),
):
    os.makedirs(PLOT_DIR, exist_ok=True)

    base_col = f"mp{mp_base}_{metric_key}"
    compare_col = f"mp{mp_compare}_{metric_key}"

    ratio_norm = TwoSlopeNorm(
        vmin=-0.2,
        vcenter=0,
        vmax=0.2
    )

    fig, axes = plt.subplots(
        1,
        len(rollout_indices),
        figsize=(5.8 * len(rollout_indices), 4.2),
        sharex=True,
        sharey=True,
    )

    if len(rollout_indices) == 1:
        axes = [axes]

    last_ratio_im = None

    for col, rollout_idx in enumerate(rollout_indices):
        ax = axes[col]
        df_r = df[df["rollout_idx"] == rollout_idx]

        last_ratio_im = ax.scatter(
            df_r["sigma_deg"],
            df_r["amplitude_A"],
            c=df_r["improvement_ratio"],
            cmap="RdBu_r",
            norm=ratio_norm,
            s=110,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.9,
        )

        ax.set_title(f"Rollout {rollout_idx + 1}",fontsize=18)
        ax.set_xlabel(r"$\sigma$ [deg]",fontsize=18)
        ax.grid(True, alpha=0.3)

        if col == 0:
            ax.set_ylabel("Amplitude A",fontsize=18)

        # Circle the cases where MP_compare improves over MP_base
        improved = df_r[compare_col] < df_r[base_col]

        improvement_amount = (
            df_r.loc[improved, base_col]
            - df_r.loc[improved, compare_col]
        )

        if len(improvement_amount) > 0 and improvement_amount.max() > 0:

            ax.scatter(
                df_r.loc[improved, "sigma_deg"],
                df_r.loc[improved, "amplitude_A"],
                facecolors="none",
                edgecolors="red",
                s=70,
                linewidth=.8,
                zorder=5,
            )

    cbar_ratio = fig.colorbar(
        last_ratio_im,
        ax=axes[2],
        shrink=0.9,
        pad=0.02,
    )

    metric_dict = {"energy_rel_error":"REE", "rmse":"RMSE" }
    metric_dict2 =  {"energy_rel_error":"relative energy error (REE)", "rmse":"root mean squared error (RMSE)" }

    cbar_ratio.set_label(
         rf"$\log_{{10}}({metric_dict[metric_key]}_{{MP{mp_base}}}"
        rf"/{metric_dict[metric_key]}_{{MP{mp_compare}}}) - $ "
    )

    fig.suptitle(
        rf"Effect of message-passing depth over initial-condition parameter space" "\n"
        rf"from {mp_base} to {mp_compare} message-passing layers - {metric_dict2[metric_key]}",
        fontsize=22,
    )

    fig.tight_layout()

    out_path = os.path.join(
        PLOT_DIR,
        f"{metric_key}_mp{mp_base}_mp{mp_compare}_improvement_parameter_space_bottom_row_only.png",
    )

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure to: {out_path}")

if __name__ == "__main__":
    metadata = load_initial_condition_metadata(DATASET_PATH)

    df = make_improvement_dataframe(
        metadata=metadata,
        mp_base=MP_BASE,
        mp_compare=MP_COMPARE,
        metric_key=METRIC_KEY,
        rollout_indices=ROLLOUT_INDICES,
    )


    plot_mp_improvement_thesis_figure(
        df=df,
        mp_base=MP_BASE,
        mp_compare=MP_COMPARE,
        metric_key=METRIC_KEY,
        rollout_indices=ROLLOUT_INDICES,
    )
