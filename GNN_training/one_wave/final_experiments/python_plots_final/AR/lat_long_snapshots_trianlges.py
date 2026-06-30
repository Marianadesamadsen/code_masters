from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import trimesh


# Settings
BASE_DIR = Path("GNN_training/one_wave/different_mesh_size/final_results")
PLOT_DIR = Path("GNN_training/one_wave/different_mesh_size/final_results_plots/AR/final")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

GRID_SUBDIVISIONS = 4
DRAW_TRIANGLE_EDGES = True

dt_type = 40
sample_idx = 3
feature_idx = 0

AR1_ZARR_PATH = BASE_DIR / f"test_{dt_type}dt.zarr"
AR2_ZARR_PATH = BASE_DIR / f"test_{dt_type}dt_AR.zarr"

save_name = f"latlon_triangular_{dt_type}dt_AR1_AR2_sampleidx{sample_idx}.png"

RUNS = {
    # f"{dt_type}dt 1": {
    #     "label_true": rf"$u_{{t+ {dt_type}\Delta t}}$",
    #     "label_ar1": rf"$\hat{{u}}^{{AR1}}_{{t+{dt_type}\Delta t}}$",
    #     "label_ar2": rf"$\hat{{u}}^{{AR2}}_{{t+{dt_type}\Delta t}}$",
    #     "rollout_idx": 0,
    # },
    f"{dt_type}dt 2": {
        "label_true": rf"$u_{{t+2\cdot {dt_type}\Delta t}}$",
        "label_ar1": rf"$\hat{{u}}^{{AR1}}_{{t+2\cdot {dt_type}\Delta t}}$",
        "label_ar2": rf"$\hat{{u}}^{{AR2}}_{{t+2\cdot {dt_type}\Delta t}}$",
        "rollout_idx": 1,
    },
    f"{dt_type}dt 3": {
        "label_true": rf"$u_{{t+3\cdot {dt_type}\Delta t}}$",
        "label_ar1": rf"$\hat{{u}}^{{AR1}}_{{t+3\cdot {dt_type}\Delta t}}$",
        "label_ar2": rf"$\hat{{u}}^{{AR2}}_{{t+3\cdot {dt_type}\Delta t}}$",
        "rollout_idx": 2,
    },
    f"{dt_type}dt 4": {
        "label_true": rf"$u_{{t+4\cdot {dt_type}\Delta t}}$",
        "label_ar1": rf"$\hat{{u}}^{{AR1}}_{{t+4\cdot {dt_type}\Delta t}}$",
        "label_ar2": rf"$\hat{{u}}^{{AR2}}_{{t+4\cdot {dt_type}\Delta t}}$",
        "rollout_idx": 3,
    },
    f"{dt_type}dt 5": {
        "label_true": rf"$u_{{t+5\cdot {dt_type}\Delta t}}$",
        "label_ar1": rf"$\hat{{u}}^{{AR1}}_{{t+5\cdot {dt_type}\Delta t}}$",
        "label_ar2": rf"$\hat{{u}}^{{AR2}}_{{t+5\cdot {dt_type}\Delta t}}$",
        "rollout_idx": 4,
    },
    # f"{dt_type}dt 6": {
    #     "label_true": rf"$u_{{t+6\cdot {dt_type}\Delta t}}$",
    #     "label_ar1": rf"$\hat{{u}}^{{AR1}}_{{t+6\cdot {dt_type}\Delta t}}$",
    #     "label_ar2": rf"$\hat{{u}}^{{AR2}}_{{t+6\cdot {dt_type}\Delta t}}$",
    #     "rollout_idx": 5,
    # },
}

LATLON_SOURCE = (
    "GNN_training/one_wave/nc_files/"
    "wave_200_dtsmall_min10_g4_sigamin6_sigmax12_correctT_plus160dt.nc"
)


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

    return triangulation


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
    first=False,
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

    ax.set_title(title, fontsize=26)

    if last:
        ax.set_xlabel("Longitude", fontsize=20)

    if first:
        ax.set_ylabel("Latitude", fontsize=20)

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
    triangulation = load_lat_lon_and_triangles()

    ds_ar1 = xr.open_zarr(AR1_ZARR_PATH)
    ds_ar2 = xr.open_zarr(AR2_ZARR_PATH)

    loaded = {}

    for run_key, cfg in RUNS.items():
        rollout_idx = cfg["rollout_idx"]

        target1, pred_ar1 = extract_rollout_step(
            ds_ar1,
            sample_idx=sample_idx,
            rollout_idx=rollout_idx,
            feature_idx=feature_idx,
        )

        _, pred_ar2 = extract_rollout_step(
            ds_ar2,
            sample_idx=sample_idx,
            rollout_idx=rollout_idx,
            feature_idx=feature_idx,
        )


        diff_error = np.abs(pred_ar2 - target1) - np.abs(pred_ar1 - target1)

        loaded[run_key] = {
            "label_true": cfg["label_true"],
            "label_ar1": cfg["label_ar1"],
            "label_ar2": cfg["label_ar2"],
            "target": target1,
            "pred_ar1": pred_ar1,
            "pred_ar2": pred_ar2,
            "diff": diff_error,
        }

    all_fields = []
    for run_key in RUNS:
        all_fields.extend(
            [
                loaded[run_key]["target"],
                loaded[run_key]["pred_ar1"],
                loaded[run_key]["pred_ar2"],
            ]
        )

    vmax = 0.5
    vmin = -vmax

    all_diffs = [loaded[run_key]["diff"] for run_key in RUNS]
    diff_vmax = max(np.nanmax(np.abs(v)) for v in all_diffs)
    diff_vmin = -diff_vmax

    fig, axes = plt.subplots(
        nrows=len(RUNS),
        ncols=4,
        figsize=(24, 12),
        constrained_layout=True,
        squeeze=False,
    )

    im_u = None
    im_diff = None
    nrows = len(RUNS)
    for row, run_key in enumerate(RUNS):
        last = row == len(RUNS) - 1
        data = loaded[run_key]

        im_u = plot_triangular_field(
            axes[row, 0],
            triangulation,
            data["target"],
            title=data["label_true"],
            vmin=vmin,
            vmax=vmax,
            last=last,
            first=True,
        )
        format_axis_ticks(axes[row, 0], row, 0, nrows)

        im_u = plot_triangular_field(
            axes[row, 1],
            triangulation,
            data["pred_ar1"],
            title=data["label_ar1"],
            vmin=vmin,
            vmax=vmax,
            last=last,
            first=False,
        )
        format_axis_ticks(axes[row, 1], row, 0, nrows)

        im_u = plot_triangular_field(
            axes[row, 2],
            triangulation,
            data["pred_ar2"],
            title=data["label_ar2"],
            vmin=vmin,
            vmax=vmax,
            last=last,
            first=False,
        )
        axes[row, 2].tick_params(axis="both", labelsize=18)
        format_axis_ticks(axes[row, 2], row, 0, nrows)

        im_diff = plot_triangular_field(
            axes[row, 3],
            triangulation,
            data["diff"],
            title=r"$|u-\hat{u}^{AR2}|-|u-\hat{u}^{AR1}$|",
            vmin=diff_vmin,
            vmax=diff_vmax,
            last=last,
            first=False,
        )
        axes[row, 3].tick_params(axis="both", labelsize=18)
        format_axis_ticks(axes[row, 3], row, 0, nrows)


    cbar1 = fig.colorbar(im_u, ax=axes[:, :3], shrink=0.8)
    cbar1.set_label(r"$u$", fontsize=22)

    cbar2 = fig.colorbar(im_diff, ax=axes[:, 3], shrink=0.8)
    cbar2.set_label(r"$|u-\hat{u}^{AR2}|-|u-\hat{u}^{AR1}|$", fontsize=22)

    # fig.suptitle(
    #     f"AR1 and AR2 predictions across rollouts for {dt_type}$\\Delta t$ model",
    #     fontsize=22,
    # )

    out_path = PLOT_DIR / save_name
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()