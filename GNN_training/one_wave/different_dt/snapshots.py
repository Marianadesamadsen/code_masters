import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import SymLogNorm
from scipy.interpolate import griddata

sys.path.insert(0, "./")


ENSEMBLE_MEMBER = 54   # choose any ensemble member that exists in both
SAMPLE_IDX = 0         # choose same start index if available


RUN_CONFIGS = {
    "normal_dt": {
        "label": r"$\Delta t$",
        "raw_dir": (
            "GNN_training/one_wave/different_dt/"
            "test_75_results_new.zarr"
        ),
        "result_dir": (
            "GNN_training/one_wave/different_dt/"
            "test_75_results_new"
        ),
        "rollout_indices": (0, 19),  # rollout 1 and 20
    },
    "double_dt": {
        "label": r"$2\Delta t$",
        "raw_dir": (
            "GNN_training/one_wave/different_dt/"
            "test_75_results_double_dt.zarr"
        ),
        "result_dir": (
            "GNN_training/one_wave/different_dt/"
            "test_75_results_double_dt"
        ),
        "rollout_indices": (0, 9),  # rollout 1 and 10
    },
}


def field_to_latlon_grid(lon, lat, field, n_lon=360, n_lat=180):
    lon_grid = np.linspace(-180, 180, n_lon)
    lat_grid = np.linspace(-90, 90, n_lat)
    Lon, Lat = np.meshgrid(lon_grid, lat_grid)

    field_grid = griddata(
        points=np.column_stack([lon, lat]),
        values=field,
        xi=(Lon, Lat),
        method="linear",
    )

    missing = np.isnan(field_grid)
    if np.any(missing):
        nearest = griddata(
            points=np.column_stack([lon, lat]),
            values=field,
            xi=(Lon, Lat),
            method="nearest",
        )
        field_grid[missing] = nearest[missing]

    return lon_grid, lat_grid, field_grid


def get_lon_lat_from_P(P):
    if P.shape[0] == 3 and P.shape[1] != 3:
        P = P.T

    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)

    lon = np.degrees(np.arctan2(y, x))
    lat = np.degrees(np.arcsin(z / r))

    return lon, lat

def print_best_ensemble_member_normal_dt():
    cfg = RUN_CONFIGS["normal_dt"]

    ds_all = xr.open_zarr(cfg["raw_dir"])

    metric_path = os.path.join(
        cfg["result_dir"],
        "test_rmse_per_sample.csv",
    )

    metadata_path = os.path.join(
        cfg["result_dir"],
        "test_metadata.csv",
    )

    metric_df = pd.read_csv(metric_path)
    metadata = pd.read_csv(metadata_path)

    n_saved_samples = ds_all.sizes["sample"]

    metric_df = metric_df.iloc[:n_saved_samples].reset_index(drop=True)
    metadata = metadata.iloc[:n_saved_samples].reset_index(drop=True)

    mean_scores = metric_df.mean(axis=1)

    best_idx = int(mean_scores.idxmin())

    best_row = metadata.iloc[best_idx]

    print("\nBest normal dt trajectory among saved Zarr samples:")
    print(f"zarr sample index : {best_idx}")
    print(f"ensemble member   : {best_row['ensemble_member']}")
    print(f"sample idx        : {best_row['sample_idx']}")
    print(f"mean RMSE         : {mean_scores.iloc[best_idx]}")


def find_sample_by_metadata(result_dir, raw_dir, ensemble_member, sample_idx):
    ds_all = xr.open_zarr(raw_dir)
    metadata = pd.read_csv(os.path.join(result_dir, "test_metadata.csv"))
    metadata = metadata.iloc[:ds_all.sizes["sample"]].reset_index(drop=True)

    matches = np.where(
        (metadata["ensemble_member"].to_numpy() == ensemble_member)
        & (metadata["sample_idx"].to_numpy() == sample_idx)
    )[0]

    if len(matches) == 0:
        raise ValueError(
            f"No sample found for ensemble_member={ensemble_member}, "
            f"sample_idx={sample_idx} in {result_dir}"
        )

    return int(matches[0])


def plot_same_wave_dt_snapshots(
    ds_geo_dir,
    plot_dir,
    ensemble_member=87,
    sample_idx=0,
    linthresh=1e-3,
):
    os.makedirs(plot_dir, exist_ok=True)

    ds_geo = xr.open_dataset(ds_geo_dir)
    P = ds_geo["P"].values
    lon, lat = get_lon_lat_from_P(P)

    rows = []
    all_fields = []
    all_errors = []

    for run_key, cfg in RUN_CONFIGS.items():
        ds_all = xr.open_zarr(cfg["raw_dir"])

        zarr_sample = find_sample_by_metadata(
            result_dir=cfg["result_dir"],
            raw_dir=cfg["raw_dir"],
            ensemble_member=ensemble_member,
            sample_idx=sample_idx,
        )

        pred = ds_all["prediction"].isel(sample=zarr_sample).values[:, :, 0]
        target = ds_all["target"].isel(sample=zarr_sample).values[:, :, 0]

        for ridx in cfg["rollout_indices"]:
            rows.append({
                "run_key": run_key,
                "run_label": cfg["label"],
                "rollout_idx": ridx,
                "prediction": pred[ridx],
                "target": target[ridx],
                "error": target[ridx] - pred[ridx],
            })

            all_fields.append(pred[ridx])
            all_fields.append(target[ridx])
            all_errors.append(target[ridx] - pred[ridx])

    u_min = float(np.nanmin([np.nanmin(f) for f in all_fields]))
    u_max = float(np.nanmax([np.nanmax(f) for f in all_fields]))
    err_abs = float(np.nanmax([np.nanmax(np.abs(e)) for e in all_errors]))

    field_norm = colors.Normalize(vmin=u_min, vmax=u_max)
    error_norm = SymLogNorm(
        linthresh=linthresh,
        linscale=1.0,
        vmin=-err_abs,
        vmax=err_abs,
        base=10,
    )

    n_rows = len(rows)
    n_cols = 3

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.2 * n_cols, 2.8 * n_rows),
        sharex=True,
        sharey=True,
    )

    for row_idx, row_info in enumerate(rows):
        fields = [
            ("Prediction", row_info["prediction"], "viridis", field_norm),
            ("Target", row_info["target"], "viridis", field_norm),
            ("Error", row_info["error"], "coolwarm", error_norm),
        ]

        for col, (title, field, cmap, norm) in enumerate(fields):
            ax = axes[row_idx, col]

            _, _, field_grid = field_to_latlon_grid(
                lon,
                lat,
                field,
                n_lon=360,
                n_lat=180,
            )

            im = ax.imshow(
                field_grid,
                extent=[-180, 180, -90, 90],
                origin="lower",
                aspect="auto",
                cmap=cmap,
                norm=norm,
            )

            if row_idx == 0:
                ax.set_title(title, fontsize=20, fontweight="bold")

            if col == 0:
                ax.set_ylabel(
                    f"{row_info['run_label']}\n"
                    f"Rollout {row_info['rollout_idx'] + 1}\n"
                    f"lat [deg]",
                    fontsize=18,
                )
                ax.tick_params(axis="y", labelsize=14)

            if row_idx == n_rows - 1:
                ax.set_xlabel("lon [deg]", fontsize=18)
                ax.tick_params(axis="x", labelsize=13)

            ax.set_xlim(-180, 180)
            ax.set_ylim(-90, 90)
            ax.grid(True, alpha=0.25)

            if col != 0:
                cbar = fig.colorbar(im, ax=ax, shrink=0.85)
                cbar.set_label("$u_t-\hat u_t$" if title == "Error" else "u",fontsize=20)
                cbar.ax.tick_params(labelsize=14)

   
    fig.suptitle(
        rf"Comparison of $\Delta t$ model and $2\Delta t$ model",
        fontsize=22,
    )

    fig.subplots_adjust(right=0.86)

    out_path = os.path.join(
        plot_dir,
        f"same_wave_dt_snapshots_wave{ensemble_member}_start{sample_idx}.png",
    )

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved figure to: {out_path}")


if __name__ == "__main__":
    ds_geo_dir = (
        "GNN_training/one_wave/nc_files/"
        "wave_200_ts_600_g4_sigmamin_6.nc"
    )

    plot_dir = (
        "GNN_training/one_wave/different_dt/"
        "all_results_plot/other_results"
    )
    print_best_ensemble_member_normal_dt()

    plot_same_wave_dt_snapshots(
        ds_geo_dir=ds_geo_dir,
        plot_dir=plot_dir,
        ensemble_member=ENSEMBLE_MEMBER,
        sample_idx=SAMPLE_IDX,
        linthresh=1e-3,
    )
