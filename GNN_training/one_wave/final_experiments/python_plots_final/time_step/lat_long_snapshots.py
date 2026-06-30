from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import trimesh


# -----------------------------
# Settings
# -----------------------------
BASE_DIR = Path("GNN_training/one_wave/different_mesh_size/final_results")
PLOT_DIR = Path("GNN_training/one_wave/different_mesh_size/final_results_plots/time_step/final")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

GRID_SUBDIVISIONS = 4
DRAW_TRIANGLE_EDGES = True

dt_type = 100
sample_idx = 3

RUNS = {
    f"{dt_type}dt 1": {
        "label": rf"$\hat{{u}}_{{t+\Delta t}}$",
        "label2": rf"$u_{{t+\Delta t}}$",
        "zarr_path": BASE_DIR / f"test_{dt_type}dt.zarr",
        "rollout_idx": 0,
    },
    f"{dt_type}dt 2": {
        "label": rf"$\hat{{u}}_{{t+2\Delta t}}$",
        "label2": rf"$u_{{t+2\Delta t}}$",
        "zarr_path": BASE_DIR / f"test_{dt_type}dt.zarr",
        "rollout_idx": 1,
    },
    f"{dt_type}dt 3": {
        "label": rf"$\hat{{u}}_{{t+3\Delta t}}$",
        "label2": rf"$u_{{t+3\Delta t}}$",
        "zarr_path": BASE_DIR / f"test_{dt_type}dt.zarr",
        "rollout_idx": 2,
    },
    f"{dt_type}dt 4": {
        "label": rf"$\hat{{u}}_{{t+4\Delta t}}$",
        "label2": rf"$u_{{t+4\Delta t}}$",
        "zarr_path": BASE_DIR / f"test_{dt_type}dt.zarr",
        "rollout_idx": 3,
    },
    f"{dt_type}dt 5": {
        "label": rf"$\hat{{u}}_{{t+5\Delta t}}$",
        "label2": rf"$u_{{t+5\Delta t}}$",
        "zarr_path": BASE_DIR / f"test_{dt_type}dt.zarr",
        "rollout_idx": 4,
    },
    # f"{dt_type}dt 6": {
    #     "label": rf"$\hat{{u}}_{{t+6\Delta t}}$",
    #     "label2": rf"$u_{{t+6\Delta t}}$",
    #     "zarr_path": BASE_DIR / f"test_{dt_type}dt.zarr",
    #     "rollout_idx": 5,
    # },
}

LATLON_SOURCE = (
    "GNN_training/one_wave/nc_files/"
    "wave_200_dtsmall_min10_g4_sigamin6_sigmax12_correctT_plus160dt.nc"
)

feature_idx = 0
save_name = f"latlon_triangular_{dt_type}dt_prediction_sampleidx{sample_idx}_test.png"



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

    print("target shape:", target.shape)
    print("prediction shape:", prediction.shape)

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
):
    edgecolors = "k" if DRAW_TRIANGLE_EDGES else "none"
    linewidth = 0.05 if DRAW_TRIANGLE_EDGES else 0.0

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
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.grid(False)

    return im


def format_axes(axes):
    n_rows, n_cols = axes.shape

    for i in range(n_rows):
        for j in range(n_cols):
            ax = axes[i, j]

            # Only show x-axis ticks/label on the last row
            if i == n_rows - 1:
                ax.set_xlabel("Longitude", fontsize=20)
                ax.tick_params(axis="x", labelsize=18)
            else:
                ax.set_xlabel("")
                ax.set_xticklabels([])
                ax.tick_params(axis="x", which="both", bottom=False, top=False)

            # Only show y-axis ticks/label on the first column
            if j == 0:
                ax.set_ylabel("Latitude", fontsize=20)
                ax.tick_params(axis="y", labelsize=18)
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])
                ax.tick_params(axis="y", which="both", left=False, right=False)



def main():
    lon, lat, triangulation = load_lat_lon_and_triangles()

    loaded = {}

    for run_key, cfg in RUNS.items():
        print(f"\nLoading {run_key}: {cfg['zarr_path']}")

        ds = xr.open_zarr(cfg["zarr_path"])
        rollout_idx = cfg["rollout_idx"]

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
        all_fields.extend(
            [
                loaded[run_key]["target1"],
                loaded[run_key]["pred1"],
            ]
        )

    vmax = 0.5
    vmin = -vmax

    all_errors = [loaded[run_key]["error1"] for run_key in RUNS]
    err_vmax = max(np.nanmax(np.abs(v)) for v in all_errors)
    err_vmin = -err_vmax

    fig, axes = plt.subplots(
        nrows=len(RUNS),
        ncols=3,
        figsize=(18, 12),
        constrained_layout=True,
        squeeze=False,
    )

    im_pred = None
    im_err = None

    for row, run_key in enumerate(RUNS):
        data = loaded[run_key]

        plot_triangular_field(
            axes[row, 0],
            triangulation,
            data["target1"],
            title=data["label2"],
            vmin=vmin,
            vmax=vmax,
        )

        im_pred = plot_triangular_field(
            axes[row, 1],
            triangulation,
            data["pred1"],
            title=data["label"],
            vmin=vmin,
            vmax=vmax,
        )

        im_err = plot_triangular_field(
            axes[row, 2],
            triangulation,
            data["error1"],
            title=r"$u-\hat{u}$",
            vmin=err_vmin,
            vmax=err_vmax,
        )

    format_axes(axes)

    cbar1 = fig.colorbar(im_pred, ax=axes[:, :2], shrink=0.8)
    cbar1.set_label(r"$u$", fontsize=20)
    cbar1.ax.tick_params(labelsize=18)

    cbar2 = fig.colorbar(im_err, ax=axes[:, 2], shrink=0.8)
    cbar2.set_label(r"$u-\hat{u}$", fontsize=20)
    cbar2.ax.tick_params(labelsize=18)

    fig.suptitle(
        f"Predictions across rollouts for {dt_type}$\\Delta t$ model",
        fontsize=22,
    )

    out_path = PLOT_DIR / save_name
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()