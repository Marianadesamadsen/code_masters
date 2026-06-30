from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


# Settings
BASE_DIR = Path("GNN_training/one_wave/different_mesh_size/final_results")
PLOT_DIR = Path("GNN_training/one_wave/different_mesh_size/final_results_plots/AR")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

NC_PATH = (
    "GNN_training/one_wave/nc_files/"
    "wave_200_dtsmall_min10_g4_sigamin6_sigmax12_correctT_plus160dt.nc"
)

RUNS = {
    # "1dt": {
    #     "label": r"$1\Delta t$", 
    #     "zarr_path": BASE_DIR / "test_1dt_AR.zarr",
    #     "dt_scale": 1,
    # },
    "10dt": {
        "label": r"$10\Delta t$",
        "zarr_path": BASE_DIR / "test_10dt.zarr",
        "dt_scale": 10,
    },
        "10dt AR2": {
        "label": r"$10\Delta t$ AR2",
        "zarr_path": BASE_DIR / "test_10dt_AR.zarr",
        "dt_scale": 10,
    },
    "20dt": {
        "label": r"$20\Delta t$",
        "zarr_path": BASE_DIR / "test_20dt.zarr",
        "dt_scale": 20,
    },
    "20dt AR2": {
        "label": r"$20\Delta t$ AR2",
        "zarr_path": BASE_DIR / "test_20dt_AR.zarr",
        "dt_scale": 20,
    },
    "40dt": {
        "label": r"$40\Delta t$",
        "zarr_path": BASE_DIR / "test_40dt.zarr",
        "dt_scale": 40,
    },
    "40dt AR2": {
        "label": r"$40\Delta t$ AR2",
        "zarr_path": BASE_DIR / "test_40dt_AR.zarr",
        "dt_scale": 40,
    },
    # "80dt": {
    #     "label": r"$80\Delta t$",
    #     "zarr_path": BASE_DIR / "test_80dt_AR.zarr",
    #     "dt_scale": 80,
    # },
}

sample_idx = 3
feature_idx = 0

n_bins = 50
base_dt = 0.0155152202228378
n_rollouts_to_plot = 5

save_name = f"radial_profile_all_sampleidx{sample_idx}.png"


# Geometry and matching
def load_nc_geometry(ds_nc):
    lat = ds_nc["lat"].values
    lon = ds_nc["lon"].values

    if np.nanmax(np.abs(lat)) > np.pi + 1e-6:
        lat = np.deg2rad(lat)

    if np.nanmax(np.abs(lon)) > 2 * np.pi + 1e-6:
        lon = np.deg2rad(lon)

    xyz = np.column_stack(
        [
            np.cos(lat) * np.cos(lon),
            np.cos(lat) * np.sin(lon),
            np.sin(lat),
        ]
    )

    xyz = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)

    return xyz


def find_matching_nc_member_and_time(ds_nc, ds_zarr, sample_idx, feature_idx):
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
    matched_time = float(nc_times[time_idx])

    # Shape: (ensemble_member, grid_index)
    nc_fields = ds_nc["u"].isel(time=time_idx).values

    # Compare all ensemble members at this time with the zarr target
    mse = np.mean((nc_fields - target0[None, :]) ** 2, axis=1)
    ensemble_idx = int(np.argmin(mse))
    matching_rmse = float(np.sqrt(mse[ensemble_idx]))

    A = float(ds_nc["A"].isel(ensemble_member=ensemble_idx).values)
    sigma = float(ds_nc["sigma"].isel(ensemble_member=ensemble_idx).values)
    sigma_deg = float(ds_nc["sigma_deg"].isel(ensemble_member=ensemble_idx).values)

    print()
    print("Matching zarr sample to NC:")
    print(f"zarr valid_time      = {valid_time:.6f}")
    print(f"closest NC time      = {matched_time:.6f}")
    print(f"NC time_idx          = {time_idx}")
    print(f"matched ensemble     = {ensemble_idx}")
    print(f"matching RMSE        = {matching_rmse:.6e}")
    print(f"A                    = {A:.4f}")
    print(f"sigma                = {sigma:.4f}")
    print(f"sigma_deg            = {sigma_deg:.2f}")

    if "center" in ds_nc:
        center = ds_nc["center"].isel(ensemble_member=ensemble_idx).values
        print(f"center               = {center}")

    return ensemble_idx, time_idx, A, sigma_deg, matching_rmse


def get_initial_center_from_nc(ds_nc, ensemble_idx, xyz):
    if "center" in ds_nc:
        center = ds_nc["center"].isel(ensemble_member=ensemble_idx).values
        center = center.astype(float)
        center = center / np.linalg.norm(center)
        print("Using center variable from NC file.")
    else:
        u0 = ds_nc["u"].isel(
            ensemble_member=ensemble_idx,
            time=0,
        ).values

        center_idx = int(np.argmax(np.abs(u0)))
        center = xyz[center_idx]
        center = center / np.linalg.norm(center)
        print("Using argmax of u(t=0) from NC file.")

    r_nodes = np.arccos(np.clip(xyz @ center, -1.0, 1.0))

    return r_nodes


# Radial profile
def make_bin_indices(r_nodes):
    bins = np.linspace(0.0, np.pi, n_bins + 1)

    bin_idx = np.digitize(r_nodes, bins) - 1

    # Make sure values exactly at pi are included in the last bin
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    return bin_idx


def get_profile(values, r_nodes, bin_idx):
    r_bin = []
    u_bin = []

    for b in range(n_bins):
        inside = bin_idx == b

        if np.any(inside):
            r_bin.append(np.mean(r_nodes[inside]))
            u_bin.append(np.mean(values[inside]))

    return np.array(r_bin), np.array(u_bin)



# Main plotting
def main():
    ds_nc = xr.open_dataset(NC_PATH)
    xyz = load_nc_geometry(ds_nc)

    fig, axes = plt.subplots(
        3,
        len(RUNS),
        figsize=(20, 13),
        sharex=True,
    )

    for col, (run_key, cfg) in enumerate(RUNS.items()):
        print()
        print("=" * 60)
        print(f"Loading {run_key}: {cfg['zarr_path']}")
        print("=" * 60)

        ds_zarr = xr.open_zarr(cfg["zarr_path"])

        ensemble_idx, time_idx, A, sigma_deg, matching_rmse = (
            find_matching_nc_member_and_time(
                ds_nc=ds_nc,
                ds_zarr=ds_zarr,
                sample_idx=sample_idx,
                feature_idx=feature_idx,
            )
        )

        r_nodes = get_initial_center_from_nc(
            ds_nc=ds_nc,
            ensemble_idx=ensemble_idx,
            xyz=xyz,
        )

        bin_idx = make_bin_indices(r_nodes)

        prediction = ds_zarr["prediction"]
        target = ds_zarr["target"]

        n_rollouts = prediction.sizes["rollout_step"]
        max_rollouts = min(n_rollouts_to_plot, n_rollouts)

        labels = [
            rf"$k={(k + 1):.2f}$"
            for k in range(max_rollouts)
        ]

        ax_pred = axes[0, col]
        ax_true = axes[1, col]
        ax_error = axes[2, col]
        
        if col == 0:
            max_err = 0
            min_err = 0
        for rollout_idx in range(max_rollouts):
            values_pred = prediction.isel(
                sample=sample_idx,
                rollout_step=rollout_idx,
                state_feature=feature_idx,
            ).values

            values_true = target.isel(
                sample=sample_idx,
                rollout_step=rollout_idx,
                state_feature=feature_idx,
            ).values

            # Prediction profile
            r_plot, pred_profile = get_profile(values_pred, r_nodes, bin_idx)
            ax_pred.plot(
                r_plot,
                pred_profile,
                marker=".",
                markersize=3,
                linewidth=1.5,
                label=labels[rollout_idx],
            )

            # Truth profile
            r_plot, true_profile = get_profile(values_true, r_nodes, bin_idx)
            ax_true.plot(
                r_plot,
                true_profile,
                marker=".",
                markersize=3,
                linewidth=1.5,
                label=labels[rollout_idx],
            )

            # Error profile: compute nodal error first, then bin
            values_error = values_true - values_pred
            r_plot, error_profile = get_profile(values_error, r_nodes, bin_idx)
            if col ==0:
                max_err = max(max_err,max(values_error))
                min_err = min(min_err,min(values_error))

            ax_error.plot(
                r_plot,
                error_profile,
                marker=".",
                markersize=3,
                linewidth=1.5,
                label=labels[rollout_idx],
            )

        ax_pred.set_title(
            rf"{cfg['label']}",
            fontsize=20,
        )

        for ax in [ax_pred, ax_true, ax_error]:
            ax.grid(True, alpha=0.4)
            ax.set_xlim(0.0, np.pi)
            ax.tick_params(axis="both", labelsize=12)

        ax_pred.set_ylim(-0.4, 0.6)
        ax_true.set_ylim(-0.4, 0.6)
        ax_error.set_ylim(min_err,max_err)

        ax_error.set_xlabel(
            "Angular distance [rad]",
            fontsize=14,
        )

        ds_zarr.close()

    axes[0, 0].set_ylabel(r"$\hat{u}_t$", fontsize=20)
    axes[1, 0].set_ylabel(r"$u_t$", fontsize=20)
    axes[2, 0].set_ylabel(r"$u_t - \hat{u}_t$", fontsize=20)

    axes[1, -1].legend(fontsize=12, title="Rollout")

    fig.suptitle(
        f"Radial profiles around initial wave center",
        fontsize=22,
    )

    fig.tight_layout()

    out_path = PLOT_DIR / save_name
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    ds_nc.close()

    print()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()