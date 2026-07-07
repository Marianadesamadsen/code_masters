from pathlib import Path
import sys

sys.path.insert(0, "./")

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


BASE_DIR = Path("GNN_training/one_wave/different_mesh_size/final_results")
PLOT_DIR = Path("GNN_training/one_wave/different_mesh_size/final_results_plots/IB/final")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

sample_idx = 2
dt_type = 1
feature_idx = 0

ZARR_PATH = BASE_DIR / f"test_{dt_type}dt_zero.zarr"

LATLON_SOURCE = (
    "GNN_training/one_wave/nc_files/"
    "wave_200_dtsmall_min10_g4_sigamin6_sigmax12_correctT_plus160dt.nc"
)

save_name = f"zero_wave_rmse_graph_energy_{dt_type}dt_sampleidx{sample_idx}.png"

R = 1.0
c = 1.0
ut_order = 4


def compute_rmse(target, prediction):
    return np.sqrt(np.mean((target - prediction) ** 2, axis=1))


def compute_graph_energy_over_time(u, lat, lon, dt, c=1.0, R=1.0, ut_order=4):

    u = np.asarray(u)

    if u.ndim != 2:
        raise ValueError(f"Expected u with shape (time, grid_index), got {u.shape}")

    n_time, n_nodes = u.shape
    area_weight = 4.0 * np.pi * R**2 / n_nodes

  
    if ut_order == 2:
        u_t = (u[2:] - u[:-2]) / (2.0 * dt)
        u_mid = u[1:-1]
        trim = 1

    elif ut_order == 4:
        u_t = (-u[4:] + 8.0 * u[3:-1] - 8.0 * u[1:-3] + u[:-4]) / (12.0 * dt)
        u_mid = u[2:-2]
        trim = 2

    spatial_energy_density = 0.0

    kinetic_energy_density = u_t**2

    energy = 0.5 * np.sum(
        kinetic_energy_density + c**2 * spatial_energy_density,
        axis=1,
    ) * area_weight

    energy_time_trim = trim

    return energy, energy_time_trim


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

    true_energy, trim = compute_graph_energy_over_time(
        target,
        lat=None,
        lon=None,
        dt=dt,
        c=c,
        R=R,
        ut_order=ut_order,
    )

    pred_energy, trim = compute_graph_energy_over_time(
        prediction,
        lat=None,
        lon=None,
        dt=dt,
        c=c,
        R=R,
        ut_order=ut_order,
    )

    energy_time = physical_time[trim:-trim]


    energy_injection = pred_energy

    print("true graph energy min/max:", np.nanmin(true_energy), np.nanmax(true_energy))
    print("pred graph energy min/max:", np.nanmin(pred_energy), np.nanmax(pred_energy))

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(14, 10),
        constrained_layout=True,
    )

    axes[0].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[0].loglog(
        physical_time,
        rmse,
        marker=".",
        linewidth=2,
    )
    axes[0].set_ylabel("RMSE", fontsize=20)
    #axes[0].set_title("Zero-wave RMSE", fontsize=20)
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
    #axes[1].set_title("Predicted graph-node energy injection", fontsize=20)
    axes[1].tick_params(axis="both", labelsize=18)
    axes[1].grid(True, alpha=0.3)

    out_path = PLOT_DIR / save_name
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()