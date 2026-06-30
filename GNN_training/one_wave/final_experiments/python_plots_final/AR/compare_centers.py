from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


# -----------------------------
# Settings
# -----------------------------
BASE_DIR = Path("GNN_training/one_wave/different_mesh_size/final_results")
RESULTS_DIR = Path("GNN_training/one_wave/different_mesh_size/final_results_plots")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NC_FILE = Path(
    "GNN_training/one_wave/nc_files/"
    "wave_200_dtsmall_min10_g4_sigamin6_sigmax12_correctT_plus160dt.nc"
)

RUNS = {
    "1dt": {"label": r"$1\Delta t$", "result_dir": "test_1dt", "dt_scale": 1},
    "10dt": {"label": r"$10\Delta t$", "result_dir": "test_10dt", "dt_scale": 10},
    "20dt": {"label": r"$20\Delta t$", "result_dir": "test_20dt", "dt_scale": 20},
    "40dt": {"label": r"$40\Delta t$", "result_dir": "test_40dt", "dt_scale": 40},
    "80dt": {"label": r"$80\Delta t$", "result_dir": "test_80dt", "dt_scale": 80},
}

RUN_KEY = "80dt"
SAMPLE_IDXS_TO_PLOT = [0, 8]

FEATURE_IDX = 0


def get_rollout_cols(df):
    cols = [c for c in df.columns if c.startswith("rollout_")]
    cols = sorted(cols, key=lambda c: int(c.split("_")[-1]))
    rollouts = np.array([int(c.split("_")[-1]) for c in cols])
    return cols, rollouts


def load_metric(run_key, filename):
    cfg = RUNS[run_key]
    csv_path = BASE_DIR / cfg["result_dir"] / filename

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    return pd.read_csv(csv_path)


def find_matching_nc_member(ds_nc, ds_zarr, sample_idx, feature_idx=0):
    target0 = ds_zarr["target"].isel(
        sample=sample_idx,
        rollout_step=0,
        state_feature=feature_idx,
    ).values

    valid_time = float(
        ds_zarr["valid_time"].isel(
            sample=sample_idx,
            rollout_step=0,
        ).values
    )

    nc_times = ds_nc["time"].values
    time_idx = int(np.argmin(np.abs(nc_times - valid_time)))

    nc_fields = ds_nc["u"].isel(time=time_idx).values

    mse = np.mean((nc_fields - target0[None, :]) ** 2, axis=1)
    ensemble_idx = int(np.argmin(mse))
    rmse = float(np.sqrt(mse[ensemble_idx]))

    return ensemble_idx, time_idx, rmse


###### Main plot
def plot_two_samples_same_model():
    cfg = RUNS[RUN_KEY]
    dt_scale = cfg["dt_scale"]
    label = cfg["label"]

    rmse_df = load_metric(RUN_KEY, "test_rmse_per_sample.csv")
    energy_df = load_metric(RUN_KEY, "test_energy_rel_error_per_sample.csv")

    zarr_path = BASE_DIR / f"{cfg['result_dir']}.zarr"
    ds_zarr = xr.open_zarr(zarr_path)
    ds_nc = xr.open_dataset(NC_FILE)

    rmse_cols, rmse_rollouts = get_rollout_cols(rmse_df)
    energy_cols, energy_rollouts = get_rollout_cols(energy_df)

    x_rmse = rmse_rollouts * dt_scale
    x_energy = energy_rollouts * dt_scale

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    for sample_idx in SAMPLE_IDXS_TO_PLOT:
        ensemble_idx, time_idx, match_rmse = find_matching_nc_member(
            ds_nc=ds_nc,
            ds_zarr=ds_zarr,
            sample_idx=sample_idx,
            feature_idx=FEATURE_IDX,
        )

        A = float(ds_nc["A"].isel(ensemble_member=ensemble_idx).values)
        sigma_deg = float(ds_nc["sigma_deg"].isel(ensemble_member=ensemble_idx).values)

        curve_label = (
            rf"sample {sample_idx}, member {ensemble_idx}, "
            rf"$A={A:.2f}$, $\sigma={sigma_deg:.1f}^\circ$"
        )

        y_rmse = rmse_df.loc[sample_idx, rmse_cols].values.astype(float)
        y_energy = energy_df.loc[sample_idx, energy_cols].values.astype(float)

        axes[0].loglog(
            x_rmse,
            y_rmse,
            marker="o",
            linestyle="-",
            label=curve_label,
        )

        axes[1].loglog(
            x_energy,
            y_energy,
            marker="o",
            linestyle="-",
            label=curve_label,
        )

        print()
        print(f"sample_idx      = {sample_idx}")
        print(f"ensemble_idx    = {ensemble_idx}")
        print(f"time_idx        = {time_idx}")
        print(f"A               = {A:.4f}")
        print(f"sigma_deg       = {sigma_deg:.2f}")
        print(f"matching RMSE   = {match_rmse:.3e}")

    axes[0].set_ylabel("RMSE", fontsize=18)
    axes[0].grid(True, which="both", alpha=0.4)
    axes[0].legend(fontsize=11)

    axes[1].set_ylabel("Relative energy error", fontsize=18)
    axes[1].set_xlabel(r"Rollout horizon [$\Delta t_{\mathrm{base}}$]", fontsize=18)
    axes[1].grid(True, which="both", alpha=0.4)
    axes[1].legend(fontsize=11)

    fig.suptitle(
        rf"{label} model: RMSE and energy error for two test samples",
        fontsize=21,
    )

    fig.tight_layout()

    out_path = RESULTS_DIR / f"rmse_energy_two_samples_{SAMPLE_IDXS_TO_PLOT[0]}_{SAMPLE_IDXS_TO_PLOT[1]}_{RUN_KEY}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    ds_zarr.close()
    ds_nc.close()

    print()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    plot_two_samples_same_model()
    