from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import trimesh


BASE_DIR = Path("GNN_training/one_wave/different_mesh_size/final_results")
PLOT_DIR = Path("GNN_training/one_wave/different_mesh_size/final_results_plots/communicationdist")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

MESH_SUBDIVISIONS = 2
PLOT_MESH = False
PLOT_GRID = False
GRID_SUBDIVISIONS = 4

RUNS = {
    "40dt": {
        "rolloutidx": [1, 2, 3, 4, 5],
        "label": r"$40\Delta t$",
        "zarr_path": BASE_DIR / "test_40dt.zarr",
    },
    "40dt sub1": {
        "rolloutidx": [1, 2, 3, 4, 5],
        "label": r"$40\Delta t$ mesh sub 1",
        "zarr_path": BASE_DIR / "test_40dt_sub1.zarr",
    },
    "40dt mp2": {
        "rolloutidx": [1, 2, 3, 4, 5],
        "label": r"$40\Delta t$ mp 2",
        "zarr_path": BASE_DIR / "test_40dt_mp2.zarr",
    },
    "40dt sub 2 nn 91": {
        "rolloutidx": [1, 2, 3, 4, 5],
        "label": r"$40\Delta t$ sub 2 nn 91",
        "zarr_path": BASE_DIR / "test_40dt_sub2_nn91.zarr",
    },
}

LATLON_SOURCE = (
    "GNN_training/one_wave/nc_files/"
    "wave_200_dtsmall_min10_g4_sigamin6_sigmax12_correctT_plus160dt.nc"
)

sample_idx = 3
feature_idx = 0

n_lon = 360
n_lat = 180

save_name = "latlon_heatmap_true_and_models_rollouts_2.png"


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

    target_step = target[sample_idx, rollout_idx, :, feature_idx]
    pred_step = prediction[sample_idx, rollout_idx, :, feature_idx]

    return target_step, pred_step


def interpolate_to_latlon_grid(lon, lat, values, n_lon=360, n_lat=180):
    lon_grid = np.linspace(-180, 180, n_lon)
    lat_grid = np.linspace(-90, 90, n_lat)

    LON, LAT = np.meshgrid(lon_grid, lat_grid)
    points = np.column_stack([lon, lat])

    grid_values = griddata(points, values, (LON, LAT), method="nearest") # Linear

    nan_mask = np.isnan(grid_values)
    if np.any(nan_mask):
        grid_values_nearest = griddata(points, values, (LON, LAT), method="nearest")
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


def overlay_mesh_edges(
    ax,
    lon_mesh,
    lat_mesh,
    edges,
    color="k",
    linewidth=0.25,
    alpha=0.35,
    linestyle="-",
):
    for i, j in edges:
        lon_pair = np.array([lon_mesh[i], lon_mesh[j]])
        lat_pair = np.array([lat_mesh[i], lat_mesh[j]])

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


def plot_heatmap(
    ax,
    lon_grid,
    lat_grid,
    values_grid,
    title,
    vmin=None,
    vmax=None,
    last=False,
):
    im = ax.imshow(
        values_grid,
        extent=[lon_grid.min(), lon_grid.max(), lat_grid.min(), lat_grid.max()],
        origin="lower",
        aspect="auto",
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_title(title, fontsize=20)

    if last:
        ax.set_xlabel("Longitude", fontsize=18)

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.grid(False)

    return im


def main():
    lat, lon = load_lat_lon()

    if PLOT_MESH:
        lat_mesh, lon_mesh, mesh_edges = load_icosahedral_mesh_edges(MESH_SUBDIVISIONS)

    if PLOT_GRID:
        lat_grid_plot, lon_grid_plot, grid_edges = load_icosahedral_mesh_edges(
            GRID_SUBDIVISIONS
        )

    rollout_indices = list(next(iter(RUNS.values()))["rolloutidx"])

    loaded = {}

    for run_key, cfg in RUNS.items():
        print(f"\nLoading {run_key}: {cfg['zarr_path']}")
        ds = xr.open_zarr(cfg["zarr_path"])

        targets = []
        preds = []

        for rollout_idx in rollout_indices:
            target_step, pred_step = extract_rollout_step(
                ds,
                sample_idx=sample_idx,
                rollout_idx=rollout_idx,
                feature_idx=feature_idx,
            )

            targets.append(target_step)
            preds.append(pred_step)

        loaded[run_key] = {
            "label": cfg["label"],
            "targets": targets,
            "preds": preds,
        }

    first_run = next(iter(RUNS.keys()))
    true_fields = loaded[first_run]["targets"]

    all_fields = []
    all_fields.extend(true_fields)

    for run_key in RUNS:
        all_fields.extend(loaded[run_key]["preds"])

    vmax = max(np.nanmax(np.abs(v)) for v in all_fields)
    vmin = -vmax

    nrows = len(rollout_indices)
    ncols = 1 + len(RUNS)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5 * ncols, 3.5 * nrows),
        constrained_layout=True,
        squeeze=False,
    )

    im_u = None

    for row, rollout_idx in enumerate(rollout_indices):
        is_last_row = row == nrows - 1

        lon_grid, lat_grid, true_grid = interpolate_to_latlon_grid(
            lon,
            lat,
            true_fields[row],
            n_lon=n_lon,
            n_lat=n_lat,
        )

        true_title = "True" if row == 0 else ""

        im_u = plot_heatmap(
            axes[row, 0],
            lon_grid,
            lat_grid,
            true_grid,
            title=true_title,
            vmin=vmin,
            vmax=vmax,
            last=is_last_row,
        )

        axes[row, 0].set_ylabel(
            f"Rollout {rollout_idx}\nLatitude",
            fontsize=18,
        )

        for col, run_key in enumerate(RUNS, start=1):
            data = loaded[run_key]
            pred_field = data["preds"][row]

            _, _, pred_grid = interpolate_to_latlon_grid(
                lon,
                lat,
                pred_field,
                n_lon=n_lon,
                n_lat=n_lat,
            )

            pred_title = data["label"] if row == 0 else ""

            im_u = plot_heatmap(
                axes[row, col],
                lon_grid,
                lat_grid,
                pred_grid,
                title=pred_title,
                vmin=vmin,
                vmax=vmax,
                last=is_last_row,
            )

        if PLOT_MESH:
            for col in range(ncols):
                overlay_mesh_edges(
                    axes[row, col],
                    lon_mesh,
                    lat_mesh,
                    mesh_edges,
                    color="green",
                    linewidth=0.5,
                    alpha=0.8,
                )

        if PLOT_GRID:
            for col in range(ncols):
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

    cbar_u = fig.colorbar(im_u, ax=axes[:, :], shrink=0.8)
    cbar_u.set_label(r"$u$", fontsize=20)

    fig.suptitle(
        "True solution and model predictions across rollout steps",
        fontsize=22,
    )

    out_path = PLOT_DIR / save_name
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()