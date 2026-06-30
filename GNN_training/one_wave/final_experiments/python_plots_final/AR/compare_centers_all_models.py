from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import trimesh



# Settings
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

SAMPLE_IDXS_TO_PLOT = [3, 9]
FEATURE_IDX = 0


def xyz_to_latlon_deg(xyz):
    xyz = xyz / np.linalg.norm(xyz, axis=-1, keepdims=True)

    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]

    lat = np.arcsin(np.clip(z, -1.0, 1.0))
    lon = np.arctan2(y, x)

    lon = (lon + np.pi) % (2 * np.pi) - np.pi

    return np.rad2deg(lat), np.rad2deg(lon)


def latlon_deg_to_xyz(lat_deg, lon_deg):
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)

    xyz = np.column_stack(
        [
            np.cos(lat) * np.cos(lon),
            np.cos(lat) * np.sin(lon),
            np.sin(lat),
        ]
    )

    xyz = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)

    return xyz


def load_nc_grid_xyz(ds_nc):
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


def load_icosahedral_mesh_xyz(subdivisions):
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1.0)

    vertices = np.asarray(mesh.vertices)
    vertices = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)

    return vertices


def nearest_node_geodesic_distance(center_xyz, nodes_xyz):
    center_xyz = center_xyz / np.linalg.norm(center_xyz)

    dots = nodes_xyz @ center_xyz
    dots = np.clip(dots, -1.0, 1.0)

    distances_rad = np.arccos(dots)

    nearest_idx = int(np.argmin(distances_rad))
    nearest_dist_rad = float(distances_rad[nearest_idx])
    nearest_dist_deg = float(np.rad2deg(nearest_dist_rad))

    return nearest_idx, nearest_dist_rad, nearest_dist_deg


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


#
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


def get_sample_info(ds_nc, ds_zarr, sample_idx, grid_xyz, mesh_xyz):
    ensemble_idx, time_idx, match_rmse = find_matching_nc_member(
        ds_nc=ds_nc,
        ds_zarr=ds_zarr,
        sample_idx=sample_idx,
        feature_idx=FEATURE_IDX,
    )

    A = float(ds_nc["A"].isel(ensemble_member=ensemble_idx).values)
    sigma_deg = float(ds_nc["sigma_deg"].isel(ensemble_member=ensemble_idx).values)

    center_xyz = ds_nc["center"].isel(ensemble_member=ensemble_idx).values
    center_xyz = center_xyz.astype(float)
    center_xyz = center_xyz / np.linalg.norm(center_xyz)

    nearest_grid_idx, grid_dist_rad, grid_dist_deg = nearest_node_geodesic_distance(
        center_xyz=center_xyz,
        nodes_xyz=grid_xyz,
    )

    nearest_mesh_idx, mesh_dist_rad, mesh_dist_deg = nearest_node_geodesic_distance(
        center_xyz=center_xyz,
        nodes_xyz=mesh_xyz,
    )

    return {
        "sample_idx": sample_idx,
        "ensemble_idx": ensemble_idx,
        "time_idx": time_idx,
        "match_rmse": match_rmse,
        "A": A,
        "sigma_deg": sigma_deg,
        "center_xyz": center_xyz,
        "nearest_grid_idx": nearest_grid_idx,
        "grid_dist_rad": grid_dist_rad,
        "grid_dist_deg": grid_dist_deg,
        "nearest_mesh_idx": nearest_mesh_idx,
        "mesh_dist_rad": mesh_dist_rad,
        "mesh_dist_deg": mesh_dist_deg,
    }



def plot_two_samples_all_models():
    ds_nc = xr.open_dataset(NC_FILE)

    grid_xyz = load_nc_grid_xyz(ds_nc)
    mesh_xyz = load_icosahedral_mesh_xyz(subdivisions=2)

    fig, axes = plt.subplots(
        2,
        len(RUNS),
        figsize=(23, 9),
        sharex="col",
        sharey="row",
    )

    all_sample_infos = {}

    for col, (run_key, cfg) in enumerate(RUNS.items()):

        rmse_df = load_metric(run_key, "test_rmse_per_sample.csv")
        energy_df = load_metric(run_key, "test_energy_rel_error_per_sample.csv")

        zarr_path = BASE_DIR / f"{cfg['result_dir']}.zarr"
        ds_zarr = xr.open_zarr(zarr_path)

        rmse_cols, rmse_rollouts = get_rollout_cols(rmse_df)
        energy_cols, energy_rollouts = get_rollout_cols(energy_df)

        x_rmse = rmse_rollouts * cfg["dt_scale"]
        x_energy = energy_rollouts * cfg["dt_scale"]

        ax_rmse = axes[0, col]
        ax_energy = axes[1, col]

        for sample_idx in SAMPLE_IDXS_TO_PLOT:

            info = get_sample_info(
                ds_nc=ds_nc,
                ds_zarr=ds_zarr,
                sample_idx=sample_idx,
                grid_xyz=grid_xyz,
                mesh_xyz=mesh_xyz,
            )

            all_sample_infos[(run_key, sample_idx)] = info

            curve_label = (
                rf"sample {sample_idx}, "
                rf"$A={info['A']:.2f}$, "
                rf"$\sigma={info['sigma_deg']:.1f}^\circ$"
                "\n"
                rf"$d_g={info['grid_dist_deg']:.2f}^\circ$, "
                rf"$d_m={info['mesh_dist_deg']:.2f}^\circ$"
            )

            y_rmse = rmse_df.loc[
                sample_idx,
                rmse_cols,
            ].values.astype(float)

            y_energy = energy_df.loc[
                sample_idx,
                energy_cols,
            ].values.astype(float)

            ax_rmse.loglog(
                x_rmse,
                y_rmse,
                marker="o",
                linewidth=2,
                label=curve_label,
            )

            ax_energy.loglog(
                x_energy,
                y_energy,
                marker="o",
                linewidth=2,
                label=curve_label,
            )

            print()
            print(f"run_key              = {run_key}")
            print(f"sample_idx           = {info['sample_idx']}")
            print(f"ensemble_idx         = {info['ensemble_idx']}")
            print(f"time_idx             = {info['time_idx']}")
            print(f"A                    = {info['A']:.4f}")
            print(f"sigma_deg            = {info['sigma_deg']:.2f}")
            print(f"nearest_grid_idx     = {info['nearest_grid_idx']}")
            print(f"grid_dist_rad        = {info['grid_dist_rad']:.6f}")
            print(f"grid_dist_deg        = {info['grid_dist_deg']:.4f}")
            print(f"nearest_mesh_idx     = {info['nearest_mesh_idx']}")
            print(f"mesh_dist_rad        = {info['mesh_dist_rad']:.6f}")
            print(f"mesh_dist_deg        = {info['mesh_dist_deg']:.4f}")
            print(f"matching RMSE        = {info['match_rmse']:.3e}")

        ax_rmse.set_title(
            cfg["label"],
            fontsize=18,
        )

        ax_rmse.grid(True, which="both", alpha=0.4)
        ax_energy.grid(True, which="both", alpha=0.4)

        ax_energy.set_xlabel(
            r"Rollout horizon [$\Delta t_{\mathrm{base}}$]",
            fontsize=13,
        )

        # Add legend to each column so the distance info is visible
        ax_rmse.legend(fontsize=8, loc="best")
        ax_energy.legend(fontsize=8, loc="best")

        ds_zarr.close()

    axes[0, 0].set_ylabel("RMSE", fontsize=18)
    axes[1, 0].set_ylabel("Relative energy error", fontsize=18)

    fig.suptitle(
        (
            rf"RMSE and relative energy error for samples "
            rf"{SAMPLE_IDXS_TO_PLOT[0]} and {SAMPLE_IDXS_TO_PLOT[1]}"
        ),
        fontsize=22,
    )

    fig.tight_layout()

    out_path = (
        RESULTS_DIR
        / f"rmse_energy_two_samples_all_models_"
        f"{SAMPLE_IDXS_TO_PLOT[0]}_{SAMPLE_IDXS_TO_PLOT[1]}.png"
    )

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    ds_nc.close()

    print()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    plot_two_samples_all_models()