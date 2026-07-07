from pathlib import Path
import sys 
sys.path.insert(0,"./")

from integrate_sphere.compute_energy import surface_mass_integration, compute_energy_over_time
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


BASE_DIR = Path("GNN_training/one_wave/different_mesh_size/final_results")
PLOT_DIR = Path("GNN_training/one_wave/different_mesh_size/final_results_plots/IB/final")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

sample_idx = 0
dt_type = 1
feature_idx = 0

ZARR_PATH = BASE_DIR / f"test_{dt_type}dt_zero.zarr"

LATLON_SOURCE = (
    "GNN_training/one_wave/nc_files/"
    "wave_200_dtsmall_min10_g4_sigamin6_sigmax12_correctT_plus160dt.nc"
)

save_name = f"zero_wave_rmse_sem_energy_{dt_type}dt_sampleidx{sample_idx}.png"

# SEM settings
SEM_N = 6
GRID_GENERATION = 4
R = 1.0
c = 1.0
ut_order = 4


def compute_rmse(target, prediction):
    return np.sqrt(np.mean((target - prediction) ** 2, axis=1))


def main():
    print(f"Loading: {ZARR_PATH}")
    ds = xr.open_zarr(ZARR_PATH)

    ds_geo = xr.open_dataset(LATLON_SOURCE)
    dt_base = float(ds_geo.attrs["dt"])
    dt = dt_type * dt_base

    target = ds["target"].isel(sample=sample_idx, state_feature=feature_idx).values
    prediction = ds["prediction"].isel(sample=sample_idx, state_feature=feature_idx).values

    # Shape: rollout_step x grid_index
    print("target shape:", target.shape)
    print("prediction shape:", prediction.shape)

    n_rollouts = prediction.shape[0]
    rollout_steps = np.arange(1, n_rollouts + 1)
    physical_time = rollout_steps * dt


    rmse = compute_rmse(target, prediction)


    sem_out = surface_mass_integration(
        N=SEM_N,
        generation=GRID_GENERATION,
        R=R,
    )
    true_energy = compute_energy_over_time(
        target,
        generation=GRID_GENERATION,
        R=R,
        c=c,
        N=SEM_N,
        dt=dt,
        out=sem_out,
        ut_order=ut_order,
    )

    pred_energy = compute_energy_over_time(
        prediction,
        generation=GRID_GENERATION,
        R=R,
        c=c,
        N=SEM_N,
        dt=dt,
        out=sem_out,
        ut_order=ut_order,
    )

    # Energy arrays are shorter because centered finite differences remove boundary points
    trim = ut_order // 2
    energy_time = physical_time[trim:-trim]

    # For zero wave, true_energy should be approximately zero.
    # Therefore relative energy error is undefined.
    energy_injection = pred_energy 

    print("true energy min/max:", np.nanmin(true_energy), np.nanmax(true_energy))
    print("pred energy min/max:", np.nanmin(pred_energy), np.nanmax(pred_energy))


    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(14, 10),
        constrained_layout=True,
    )

    axes[0].loglog(
        physical_time,
        rmse,
        marker=".",
        linewidth=2,
    )
    axes[0].set_ylabel("RMSE", fontsize=20)
    axes[0].set_title("Zero-wave RMSE", fontsize=20)
    axes[0].tick_params(axis="both", labelsize=18)
    axes[0].grid(True, alpha=0.3)

    axes[1].loglog(
        energy_time,
        energy_injection,
        marker=".",
        linewidth=2,
    )
    axes[1].set_xlabel("Physical time (s)", fontsize=20)
    axes[1].set_ylabel(r"Energy", fontsize=20)
    axes[1].set_title("Predicted energy", fontsize=20)
    axes[1].tick_params(axis="both", labelsize=18)
    axes[1].grid(True, alpha=0.3)

    out_path = PLOT_DIR / save_name
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()