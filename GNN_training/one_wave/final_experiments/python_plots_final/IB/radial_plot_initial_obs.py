from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt



BASE_DIR = Path("GNN_training/one_wave/different_mesh_size/final_results")
#Path("GNN_training/one_wave/different_training_size")#
PLOT_DIR = Path("GNN_training/one_wave/different_mesh_size/final_results_plots/IB/final") 
PLOT_DIR.mkdir(parents=True, exist_ok=True)

NC_PATH = (
    "GNN_training/one_wave/nc_files/"
    "wave_200_dtsmall_min10_g4_sigamin6_sigmax12_correctT_plus160dt.nc"
)

RUNS = {
    "1dt": {
        "label": r"$1\Delta t$",
        "zarr_path": BASE_DIR / "test_1dt.zarr",
        "dt_scale": 1,
    },
}

sample_idx = 3
feature_idx = 0

n_bins = 20
base_dt = 0.0155152202228378
n_rollouts_to_plot = 10

save_name = f"Radial_profile_1dt_10dt_sampleidx{sample_idx}_all.png"


def load_nc_geometry(ds_nc):
    lat = ds_nc["lat"].values
    lon = ds_nc["lon"].values

    if np.nanmax(np.abs(lat)) > np.pi + 1e-6:
        lat = np.deg2rad(lat)

    if np.nanmax(np.abs(lon)) > 2 * np.pi + 1e-6:
        lon = np.deg2rad(lon)

    xyz = np.column_stack([
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat),
    ])

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

    for s in range(ds_zarr.sizes["sample"]):
        print()
        print("sample", s)
        print(ds_zarr["valid_time"].isel(sample=s).values[:5])

    nc_times = ds_nc["time"].values
    time_idx = int(np.argmin(np.abs(nc_times - valid_time)))
    matched_time = float(nc_times[time_idx])

    # Shape: (ensemble_member, grid_index)
    nc_fields = ds_nc["u"].isel(time=time_idx).values

    # Compare all ensemble members at this time with the zarr target
    mse = np.mean(( target0[None, :]-nc_fields) ** 2, axis=1)
    ensemble_idx = int(np.argmin(mse))

    print()
    print("Matching zarr sample to NC:")
    print("zarr valid_time:", valid_time)
    print("closest NC time:", matched_time)
    print("NC time_idx:", time_idx)
    print("matched ensemble_member:", ensemble_idx)
    print("matching RMSE:", np.sqrt(mse[ensemble_idx]))
    print(f"A               = {float(ds_nc['A'][ensemble_idx]):.4f}")
    print(f"sigma           = {float(ds_nc['sigma'][ensemble_idx]):.4f}")
    print(f"sigma_deg       = {float(ds_nc['sigma_deg'][ensemble_idx]):.2f}")
    print(f"center          = {ds_nc['center'][ensemble_idx].values}")

    return ensemble_idx, time_idx, float(ds_nc['A'][ensemble_idx]), float(ds_nc['sigma_deg'][ensemble_idx])


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


def make_bin_indices(r_nodes):
    bins = np.linspace(0.0, np.pi, n_bins + 1)
    bin_idx = np.digitize(r_nodes, bins) - 1
    return bin_idx


def get_profile(values, r_nodes, bin_idx):
    r_bin = []
    u_bin = []

    for b in range(n_bins):
        inside = bin_idx == b

        if np.sum(inside) > 0:
            r_bin.append(np.mean(r_nodes[inside]))
            u_bin.append(np.mean(values[inside]))

    return np.array(r_bin), np.array(u_bin)

def main(): 
    ds_nc = xr.open_dataset(NC_PATH)
    xyz = load_nc_geometry(ds_nc)

    fig, axes = plt.subplots(
        3,
        len(RUNS),
        figsize=(12, 10),
        sharex=True,
        #sharey=True,
    )

    for col, (run_key, cfg) in enumerate(RUNS.items()):
        print(f"\nLoading {run_key}: {cfg['zarr_path']}")

        ds_zarr = xr.open_zarr(cfg["zarr_path"])
        print(ds_zarr)

        ensemble_idx, time_idx, A, sigma_deg = find_matching_nc_member_and_time(
            ds_nc=ds_nc,
            ds_zarr=ds_zarr,
            sample_idx=sample_idx,
            feature_idx=feature_idx,
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

        physical_dt = base_dt * cfg["dt_scale"]

        labels = [rf"$k={k + 1}$" for k in range(max_rollouts)]

        ax_pred = axes[0]#axes[0, col]

        for rollout_idx in range(0,max_rollouts):
            values_pred = prediction.isel(
                sample=sample_idx,
                rollout_step=rollout_idx,
                state_feature=feature_idx,
            ).values

            r_plot, u_plot = get_profile(values_pred, r_nodes, bin_idx)

            ax_pred.plot(
                r_plot,
                u_plot,
                marker=".",
                markersize=3,
                linewidth=1.5,
                label=labels[rollout_idx],
            )

        # ax_pred.set_title(
        #     rf"{cfg['label']}, $\Delta t={physical_dt:.4f}$",
        #     fontsize=20,
        # )
        ax_pred.grid(True, alpha=0.4)
        ax_pred.set_xlim(0.0, np.pi/2)
        ax_pred.set_ylim(-0.3, 1.6)


        ax_true = axes[1]#, col]
        ax_error = axes[2]#, col]

        for rollout_idx in range(0,max_rollouts):
            values_true = target.isel(
                sample=sample_idx,
                rollout_step=rollout_idx,
                state_feature=feature_idx,
            ).values

            r_plot, u_plot = get_profile(values_true, r_nodes, bin_idx)

            ax_true.plot(
                r_plot,
                u_plot,
                marker=".",
                markersize=3,
                linewidth=1.5,
            )

            values_pred = prediction.isel(
                sample=sample_idx,
                rollout_step=rollout_idx,
                state_feature=feature_idx,
            ).values

            r_plot_pred, u_plot_pred = get_profile(values_pred, r_nodes, bin_idx)
            ax_error.plot(
                r_plot,
                u_plot-u_plot_pred,
                marker=".",
                markersize=3,
                linewidth=1.5,
                label=labels[rollout_idx],
            )

        ax_true.grid(True, alpha=0.4)
        ax_true.set_xlim(0.0, np.pi/2)
        ax_true.set_ylim(-0.3, 1.5)
        #ax_true.set_xlabel(
        #    "Angular distance from initial wave center [rad]",
        #    fontsize=20,
        #)
        ax_error.grid(True, alpha=0.4)
        #ax_error.set_xlim(0.0, np.pi)
        #ax_error.set_ylim(-0.3, 1.1)
        ax_error.set_xlabel(
            "Angular distance from initial wave center [rad]",
            fontsize=20,
        )

    axes[0].set_ylabel(r"$\hat{u}_t$", fontsize=20) 
    axes[1].set_ylabel("$u_t$", fontsize=20)
    axes[2].set_ylabel("$u - \\hat{u}$", fontsize=20)

    axes[0].legend(title="Rollout",title_fontsize=18, ncols=5, fontsize=18)
    axes[0].tick_params(axis="x", labelsize=18)
    axes[0].tick_params(axis="y", labelsize=18)
    axes[1].tick_params(axis="x", labelsize=18)
    axes[1].tick_params(axis="y", labelsize=18)
    axes[2].tick_params(axis="x", labelsize=18)
    axes[2].tick_params(axis="y", labelsize=18)

    # fig.suptitle(
    #     f"Radial profiles from initial wave center",# with A={A:.2f}, sigma={sigma_deg:.2f}",
    #     fontsize=22,
    # )

    fig.tight_layout()

    out_path = PLOT_DIR / save_name
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved: {out_path}")
    


if __name__ == "__main__":
    main()