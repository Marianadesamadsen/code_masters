from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import trimesh


BASE_DIR = Path("GNN_training/one_wave/different_mesh_size/final_results")
PLOT_DIR = Path("GNN_training/one_wave/different_mesh_size/final_results_plots/time_step/final")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

GRID_SUBDIVISIONS = 4
DRAW_TRIANGLE_EDGES = True

COMMON_HORIZON_DT = 120
N_INIT_STATES = 2

RUNS = {
    "10dt": {
        "label": r"Model: $10\Delta t$" + "\n" + r"$\hat{u}_{10\Delta t+11\cdot 10\Delta t}$",
        "label2": r"Model: $10\Delta t$" + "\n" + r"$u_{10\Delta t+11\cdot 10\Delta t}$",
        "zarr_path": BASE_DIR / "test_10dt.zarr",
        "dt_scale": 10,
    },
    "20dt": {
        "label": r"Model: $20\Delta t$" + "\n" + r"$\hat{u}_{20\Delta t+5\cdot 20\Delta t}$",
        "label2": r"Model: $20\Delta t$" + "\n" + r"$u_{20\Delta t+5\cdot 20\Delta t}$",
        "zarr_path": BASE_DIR / "test_20dt.zarr",
        "dt_scale": 20,
    },
    "40dt": {
        "label": r"Model: $40\Delta t$" + "\n" + r"$\hat{u}_{40\Delta t+2\cdot 40\Delta t}$",
        "label2": r"Model: $40\Delta t$" + "\n" + r"$u_{40\Delta t+2\cdot 40\Delta t}$",
        "zarr_path": BASE_DIR / "test_40dt.zarr",
        "dt_scale": 40,
    },
}

LATLON_SOURCE = (
    "GNN_training/one_wave/nc_files/"
    "wave_200_dtsmall_min10_g4_sigamin6_sigmax12_correctT_plus160dt.nc"
)

sample_idx = 3
feature_idx = 0

save_name = "latlon_triangular_same_physical_time_80dt.png"


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


def get_rollout_idx_from_horizon(dt_scale):
    rollout_idx_float = COMMON_HORIZON_DT / dt_scale - N_INIT_STATES

    if not np.isclose(rollout_idx_float, round(rollout_idx_float)):
        raise ValueError(
            f"COMMON_HORIZON_DT={COMMON_HORIZON_DT} is not compatible with "
            f"dt_scale={dt_scale} and N_INIT_STATES={N_INIT_STATES}."
        )

    rollout_idx = int(round(rollout_idx_float))

    if rollout_idx < 0:
        raise ValueError(
            f"Computed rollout_idx={rollout_idx}. "
            f"The horizon is too short for dt_scale={dt_scale}."
        )

    return rollout_idx


def extract_rollout_step(ds, sample_idx, rollout_idx, feature_idx):
    target = ds["target"].values
    prediction = ds["prediction"].values

    print("target shape:", target.shape)
    print("prediction shape:", prediction.shape)

    if rollout_idx >= target.shape[1]:
        raise IndexError(
            f"rollout_idx={rollout_idx} is outside target rollout dimension "
            f"with size {target.shape[1]}."
        )

    target_step = target[sample_idx, rollout_idx, :, feature_idx]
    pred_step = prediction[sample_idx, rollout_idx, :, feature_idx]

    return target_step, pred_step


def physical_horizon_dt(dt_scale, rollout_idx):
    return (N_INIT_STATES + rollout_idx) * dt_scale


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

    ax.set_title(title, fontsize=22)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.grid(False)

    return im


def format_axes(axes):
    n_rows, n_cols = axes.shape

    for i in range(n_rows):
        for j in range(n_cols):
            ax = axes[i, j]

            # x-axis only on last row
            if i == n_rows - 1:
                ax.set_xlabel("Longitude", fontsize=20)
                ax.tick_params(axis="x", labelsize=18)
            else:
                ax.set_xlabel("")
                ax.set_xticklabels([])
                ax.tick_params(axis="x", which="both", bottom=False, top=False)

            # y-axis only on first column
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

        dt_scale = cfg["dt_scale"]
        rollout_idx = get_rollout_idx_from_horizon(dt_scale)
        horizon_dt = physical_horizon_dt(dt_scale, rollout_idx)

        print(
            f"{run_key}: rollout_idx={rollout_idx}, "
            f"dt_scale={dt_scale}, "
            f"physical horizon={horizon_dt} Δt"
        )

        if horizon_dt != COMMON_HORIZON_DT:
            raise ValueError(
                f"{run_key} does not match common horizon. "
                f"Got {horizon_dt} Δt, expected {COMMON_HORIZON_DT} Δt."
            )

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
            "rollout_idx": rollout_idx,
            "horizon_dt": horizon_dt,
        }

    all_fields = []
    for run_key in RUNS:
        all_fields.extend(
            [
                loaded[run_key]["target1"],
                loaded[run_key]["pred1"],
            ]
        )

    vmax = 0.4
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

    out_path = PLOT_DIR / save_name
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()