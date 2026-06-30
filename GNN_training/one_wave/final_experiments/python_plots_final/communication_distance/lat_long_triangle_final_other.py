from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import trimesh


BASE_DIR = Path("GNN_training/one_wave/different_mesh_size/final_results")
PLOT_DIR = Path("GNN_training/one_wave/different_mesh_size/final_results_plots/communicationdist/final")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

GRID_SUBDIVISIONS = 4
DRAW_TRIANGLE_EDGES = True

RUNS = {
    "40dt": {
        "rolloutidx": [5, 6, 7, 8, 9],
        "label": r"$40\Delta t$",
        "zarr_path": BASE_DIR / "test_40dt_2.zarr",
    },
        "40dt sub 1": {
        "rolloutidx": [5, 6, 7, 8, 9],
        "label": r"$40\Delta t$ sub 1",
        "zarr_path": BASE_DIR / "test_40dt_sub1.zarr",
    },
    "40dt sub 2 nn91": {
        "rolloutidx": [5, 6, 7, 8, 9],
        "label": r"$40\Delta t$ sub2 nn91",
        "zarr_path": BASE_DIR / "test_40dt_sub2_nn91.zarr",
    },
    "40dt sub 2 nn91 nn9": {
        "rolloutidx": [5, 6, 7, 8, 9],
        "label": r"$40\Delta t$ sub2 nn91 nn9",
        "zarr_path": BASE_DIR / "test_40dt_sub2_nn91_nn9.zarr",
    },
}



LATLON_SOURCE = (
    "GNN_training/one_wave/nc_files/"
    "wave_200_dtsmall_min10_g4_sigamin6_sigmax12_correctT_plus160dt.nc"
)

sample_idx = 3
feature_idx = 0
save_name = "latlon_triangular_true_and_models_rollouts_others.png"


def xyz_to_latlon_deg(xyz):
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    lat = np.rad2deg(np.arcsin(z / np.linalg.norm(xyz, axis=1)))
    lon = np.rad2deg(np.arctan2(y, x))
    lon = ((lon + 180) % 360) - 180

    return lat, lon


def load_lat_lon_and_triangles():
    ds_geo = xr.open_dataset(LATLON_SOURCE)

    lat = ds_geo["lat"].values
    lon = ds_geo["lon"].values

    if np.nanmax(np.abs(lat)) <= np.pi + 1e-6:
        lat = np.rad2deg(lat)

    if np.nanmax(np.abs(lon)) <= 2 * np.pi + 1e-6:
        lon = np.rad2deg(lon)

    lon = ((lon + 180) % 360) - 180

    mesh = trimesh.creation.icosphere(
        subdivisions=GRID_SUBDIVISIONS,
        radius=1.0,
    )

    triangles = np.asarray(mesh.faces, dtype=int)

    # Remove triangles crossing the longitude seam
    keep = []
    for tri in triangles:
        lon_tri = lon[tri]
        if np.max(lon_tri) - np.min(lon_tri) < 180:
            keep.append(tri)

    triangles = np.asarray(keep, dtype=int)

    triangulation = mtri.Triangulation(
        lon,
        lat,
        triangles=triangles,
    )

    return lon, lat, triangulation


def extract_rollout_step(ds, sample_idx, rollout_idx, feature_idx):
    target = ds["target"].values
    prediction = ds["prediction"].values

    target_step = target[sample_idx, rollout_idx, :, feature_idx]
    pred_step = prediction[sample_idx, rollout_idx, :, feature_idx]

    return target_step, pred_step


def plot_triangular_field(
    ax,
    triangulation,
    values,
    title,
    vmin=None,
    vmax=None,
    last=False,
):
    edgecolors = "k" if DRAW_TRIANGLE_EDGES else "none"
    linewidth = 0.08 if DRAW_TRIANGLE_EDGES else 0.0
    
    im = ax.tripcolor(
        triangulation,
        values,
        shading="flat",
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax,
        edgecolors=edgecolors,
        linewidth=linewidth,
    )

    ax.set_title(title, fontsize=20)

    if last:
        ax.set_xlabel("Longitude", fontsize=20)

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.grid(False)

    return im


def format_axis_ticks(ax, row, col, nrows):
    # show x tick labels only on last row
    show_x = row == nrows - 1

    # show y tick labels only on first column
    show_y = col == 0

    ax.tick_params(
        axis="both",
        labelsize=18,
        labelbottom=show_x,
        labelleft=show_y,
        bottom=show_x,
        left=show_y,
    )


def main():
    lon, lat, triangulation = load_lat_lon_and_triangles()

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

    vmax = 0.5# max(np.nanmax(np.abs(v)) for v in all_fields)
    vmin = -vmax

    nrows = len(rollout_indices)
    ncols = 1 + len(RUNS)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(18,10),#(5 * ncols, 3.5 * nrows),
        constrained_layout=True,
        squeeze=False,
    )

    im_u = None

    for row, rollout_idx in enumerate(rollout_indices):
        is_last_row = row == nrows - 1

        true_title = "True" if row == 0 else ""

        im_u = plot_triangular_field(
            axes[row, 0],
            triangulation,
            true_fields[row],
            title=true_title,
            vmin=vmin,
            vmax=vmax,
            last=is_last_row,
        )

        axes[row, 0].set_ylabel(
            f"Rollout {rollout_idx}\nLatitude",
            fontsize=20,
        )
        format_axis_ticks(axes[row, 0], row, 0, nrows)

        for col, run_key in enumerate(RUNS, start=1):
            data = loaded[run_key]
            pred_field = data["preds"][row]

            pred_title = data["label"] if row == 0 else ""

            im_u = plot_triangular_field(
                axes[row, col],
                triangulation,
                pred_field,
                title=pred_title,
                vmin=vmin,
                vmax=vmax,
                last=is_last_row,
            )
            format_axis_ticks(axes[row, col], row, 1, nrows)

    cbar_u = fig.colorbar(im_u, ax=axes[:, :], shrink=0.8)
    cbar_u.set_label(r"$u$", fontsize=20)
    cbar_u.ax.tick_params(labelsize=18)

    # fig.suptitle(
    #     "Predictions using different model architectures",
    #     fontsize=22,
    # )

    out_path = PLOT_DIR / save_name
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()