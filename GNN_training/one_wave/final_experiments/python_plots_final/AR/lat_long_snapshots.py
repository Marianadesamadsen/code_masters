from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import trimesh


BASE_DIR = Path("GNN_training/one_wave/different_mesh_size/final_results")
PLOT_DIR = Path("GNN_training/one_wave/different_mesh_size/final_results_plots/AR")
PLOT_DIR.mkdir(parents=True, exist_ok=True)
MESH_SUBDIVISIONS = 2  # change to the mesh generation you want to show
PLOT_MESH = True
PLOT_GRID = True
GRID_SUBDIVISIONS = 4  

dt_type = 20
sample_idx = 3

RUNS = {
        f"{dt_type}dt 1": {
        "label": r"$\hat{u}_{t+1}$",
        "label2": r"$u_{t+1}$",
        "zarr_path": BASE_DIR / f"test_{dt_type}dt_AR.zarr",
    },
        f"{dt_type}dt 2": {
        "label": r"$\hat{u}_{t+2}$",
        "label2": r"$u_{t+2}$",
        "zarr_path": BASE_DIR / f"test_{dt_type}dt_AR.zarr",
    },
        f"{dt_type}dt 3": {
        "label": r"$\hat{u}_{t+3}$",
        "label2": r"$u_{t+3}$",
        "zarr_path": BASE_DIR / f"test_{dt_type}dt_AR.zarr",
    },
            f"{dt_type}dt 4": {
        "label": r"$\hat{u}_{t+4}$",
        "label2": r"$u_{t+4}$",
        "zarr_path": BASE_DIR / f"test_{dt_type}dt_AR.zarr",
    },
            f"{dt_type}dt 5": {
        "label": r"$\hat{u}_{t+5}$",
        "label2": r"$u_{t+5}$",
        "zarr_path": BASE_DIR / f"test_{dt_type}dt_AR.zarr",
    },
}

LATLON_SOURCE = (
    "GNN_training/one_wave/nc_files/"
    "wave_200_dtsmall_min10_g4_sigamin6_sigmax12_correctT_plus160dt.nc"
)

rollout_idx = 0
feature_idx = 0

n_lon = 360
n_lat = 180

save_name = f"latlon_heatmap_{dt_type}dt_prediction_sampleidx{sample_idx}.png"


def load_lat_lon():
    ds_geo = xr.open_dataset(LATLON_SOURCE)

    lat = ds_geo["lat"].values
    lon = ds_geo["lon"].values

    if np.nanmax(np.abs(lat)) <= np.pi + 1e-6:
        lat = np.rad2deg(lat)

    if np.nanmax(np.abs(lon)) <= 2 * np.pi + 1e-6:
        lon = np.rad2deg(lon)

    lon = ((lon + 180) % 360) - 180

    return lat, lon


def extract_rollout_step(ds, sample_idx, rollout_idx, feature_idx):
    target = ds["target"].values
    prediction = ds["prediction"].values

    print("target shape:", target.shape)
    print("prediction shape:", prediction.shape)

    target_step = target[sample_idx, rollout_idx, :, feature_idx]
    pred_step = prediction[sample_idx, rollout_idx, :, feature_idx]

    return target_step, pred_step


def interpolate_to_latlon_grid(lon, lat, values, n_lon=360, n_lat=180):
    lon_grid = np.linspace(-180, 180, n_lon)
    lat_grid = np.linspace(-90, 90, n_lat)

    LON, LAT = np.meshgrid(lon_grid, lat_grid)

    points = np.column_stack([lon, lat])

    grid_values = griddata(
        points,
        values,
        (LON, LAT),
        method="linear",
    )

    # Fill possible NaNs near boundaries with nearest-neighbour values
    nan_mask = np.isnan(grid_values)

    if np.any(nan_mask):
        grid_values_nearest = griddata(
            points,
            values,
            (LON, LAT),
            method="nearest",
        )
        grid_values[nan_mask] = grid_values_nearest[nan_mask]

    return lon_grid, lat_grid, grid_values

def xyz_to_latlon_deg(xyz):
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    lat = np.rad2deg(np.arcsin(z / np.linalg.norm(xyz, axis=1)))
    lon = np.rad2deg(np.arctan2(y, x))

    lon = ((lon + 180) % 360) - 180

    return lat, lon


def load_icosahedral_mesh_edges(subdivisions):
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=1.0)

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)

    lat_mesh, lon_mesh = xyz_to_latlon_deg(vertices)

    edges = set()
    for tri in faces:
        i, j, k = tri
        edges.add(tuple(sorted((i, j))))
        edges.add(tuple(sorted((j, k))))
        edges.add(tuple(sorted((k, i))))

    return lat_mesh, lon_mesh, list(edges)


def overlay_mesh_edges(ax, lon_mesh, lat_mesh, edges, color="k", linewidth=0.25, alpha=0.35,linestyle="-"):
    for i, j in edges:
        lon_pair = np.array([lon_mesh[i], lon_mesh[j]])
        lat_pair = np.array([lat_mesh[i], lat_mesh[j]])

        # Avoid drawing long lines across the map at the longitude seam
        if np.abs(lon_pair[0] - lon_pair[1]) > 180:
            continue

        ax.plot(
            lon_pair,
            lat_pair,
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            linestyle=linestyle,
            zorder=10,
        )

def plot_heatmap(ax, lon_grid, lat_grid, values_grid, title, vmin=None, vmax=None,last=False,first=False):
    im = ax.imshow(
        values_grid,
        extent=[
            lon_grid.min(),
            lon_grid.max(),
            lat_grid.min(),
            lat_grid.max(),
        ],
        origin="lower",
        aspect="auto",
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_title(title, fontsize=20)
    if last:
        ax.set_xlabel("Longitude",fontsize=20)
    if first:
        ax.set_ylabel("Latitude",fontsize=20)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.grid(False)

    return im


def main():
    lat, lon = load_lat_lon()
    if PLOT_MESH:
        lat_mesh, lon_mesh, mesh_edges = load_icosahedral_mesh_edges(MESH_SUBDIVISIONS)
    if PLOT_GRID:
        lat_grid_plot, lon_grid_plot, grid_edges = load_icosahedral_mesh_edges(GRID_SUBDIVISIONS)

    loaded = {}

    for run_key, cfg in RUNS.items():
        print(f"\nLoading {run_key}: {cfg['zarr_path']}")

        ds = xr.open_zarr(cfg["zarr_path"])
        print(ds)

        if run_key == f"{dt_type}dt 1":
            rollout_idx = 0
        elif run_key == f"{dt_type}dt 2":
            rollout_idx = 1
        elif run_key == f"{dt_type}dt 3":
            rollout_idx = 2
        elif run_key == f"{dt_type}dt 4":
            rollout_idx = 3
        elif run_key == f"{dt_type}dt 5":
            rollout_idx = 4

        target1, pred1 = extract_rollout_step(
            ds,
            sample_idx=sample_idx,
            rollout_idx=rollout_idx,
            feature_idx=feature_idx,
        )

        error1 = target1 - pred1

        loaded[run_key] = {
            "label": cfg["label"],
            "label2": cfg["label2"],    
            "target1": target1,
            "pred1": pred1,
            "error1": error1,
        }

    all_fields = []
    for run_key in RUNS:
        all_fields.extend([
            loaded[run_key]["target1"],
            loaded[run_key]["pred1"],
        ])

    vmax = max(np.nanmax(np.abs(v)) for v in all_fields)
    vmin = -vmax

    all_errors = [loaded[run_key]["error1"] for run_key in RUNS]
    err_vmax = max(np.nanmax(np.abs(v)) for v in all_errors)
    err_vmin = -err_vmax

    fig, axes = plt.subplots(
        nrows=5,
        ncols=3,
        figsize=(18, 12),
        constrained_layout=True,
    )

    for row, run_key in enumerate(RUNS):
        if row == 4:
            last = True
        else:
            last = False
        data = loaded[run_key]
        label = data["label"]
        label2 = data["label2"]

        lon_grid, lat_grid, target_grid = interpolate_to_latlon_grid(
            lon,
            lat,
            data["target1"],
            n_lon=n_lon,
            n_lat=n_lat,
        )

        _, _, pred_grid = interpolate_to_latlon_grid(
            lon,
            lat,
            data["pred1"],
            n_lon=n_lon,
            n_lat=n_lat,
        )

        _, _, error_grid = interpolate_to_latlon_grid(
            lon,
            lat,
            data["error1"],
            n_lon=n_lon,
            n_lat=n_lat,
        )

        im_truth = plot_heatmap(
            axes[row, 0],
            lon_grid,
            lat_grid,
            target_grid,
            f"{label2}",#: $u_{{t+4}}$",
            vmin=vmin,
            vmax=vmax,
            last = last,
            first = True
        )

        im_pred = plot_heatmap(
            axes[row, 1],
            lon_grid,
            lat_grid,
            pred_grid,
            f"{label}",#: $\\hat{{u}}_{{t+4}}$",#Model {label}: 
            vmin=vmin,
            vmax=vmax,
            last= last
        )

        im_err = plot_heatmap(
            axes[row, 2],
            lon_grid,
            lat_grid,
            error_grid,
            f"$u-\\hat{{u}}$",#: $ u_{{t+4}}-\\hat{{u}}_{{t+4}}$",
            vmin=err_vmin,
            vmax=err_vmax,
            last=last
        )
        if PLOT_MESH:
            for col in range(2):
                overlay_mesh_edges(
                    axes[row, col],
                    lon_mesh,
                    lat_mesh,
                    mesh_edges,
                    color="green",
                    linewidth=0.5,
                    alpha=0.8,
                )
                overlay_mesh_edges(
                    axes[row, col],
                    lon_grid_plot,
                    lat_grid_plot,
                    grid_edges,
                    color="black",
                    linewidth=0.5,
                    alpha=0.35,
                    linestyle="--",
                )
        

    cbar1 = fig.colorbar(im_pred, ax=axes[:, :2], shrink=0.8)
    cbar1.set_label("u", fontsize=20)
    cbar2 = fig.colorbar(im_err, ax=axes[:, 2], shrink=0.8)
    cbar2.set_label(r"$ u - \hat{u}$", fontsize=20)

    fig.suptitle(
        f"Prediction heatmaps across different rollouts model {dt_type}dt",
        fontsize=22,
    )

    out_path = PLOT_DIR / save_name
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()