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

    # Fill possible gaps with nearest-neighbour interpolation
    missing = np.isnan(field_grid)
    if np.any(missing):
        field_nearest = griddata(
            points=np.column_stack([lon, lat]),
            values=field,
            xi=(Lon, Lat),
            method="nearest",
        )
        field_grid[missing] = field_nearest[missing]

    return lon_grid, lat_grid, field_grid

def get_lon_lat_from_P(P):
    if P.shape[0] == 3 and P.shape[1] != 3:
        P = P.T

    x = P[:, 0]
    y = P[:, 1]
    z = P[:, 2]

    r = np.sqrt(x**2 + y**2 + z**2)

    lon = np.degrees(np.arctan2(y, x))
    lat = np.degrees(np.arcsin(z / r))

    return lon, lat


def plot_latlon_snapshots_best_worst_rmse(
    ds_geo_dir,
    raw_dir,
    result_dir,
    train_size,
    plot_dir,
    metric_key="rmse",
    rollout_indices=(0, 9, 19),
):
    os.makedirs(plot_dir, exist_ok=True)

    ds_geo = xr.open_dataset(ds_geo_dir)
    ds_all = xr.open_zarr(raw_dir)

    metric_path = os.path.join(result_dir, f"test_{metric_key}_per_sample.csv")
    metadata_path = os.path.join(result_dir, "test_metadata.csv")

    metric_df = pd.read_csv(metric_path)
    metadata = pd.read_csv(metadata_path)

    n_saved_samples = ds_all.sizes["sample"]

    metric_df = metric_df.iloc[:n_saved_samples].reset_index(drop=True)
    metadata = metadata.iloc[:n_saved_samples].reset_index(drop=True)

    mean_scores = metric_df.mean(axis=1)

    best_idx = int(mean_scores.idxmin())
    worst_idx = int(mean_scores.idxmax())

    print("\nBest saved RMSE trajectory:")
    print("Zarr sample:", best_idx)
    print(metadata.iloc[best_idx])
    print("Mean RMSE:", mean_scores.iloc[best_idx])

    print("\nWorst saved RMSE trajectory:")
    print("Zarr sample:", worst_idx)
    print(metadata.iloc[worst_idx])
    print("Mean RMSE:", mean_scores.iloc[worst_idx])

    P = ds_geo["P"].values
    lon, lat = get_lon_lat_from_P(P)

    selected = [
        (best_idx, "Best RMSE"),
        (worst_idx, "Worst RMSE"),
    ]

    all_fields = []
    all_errors = []

    for sample_idx, _ in selected:
        pred = ds_all["prediction"].isel(sample=sample_idx).values[:, :, 0]
        target = ds_all["target"].isel(sample=sample_idx).values[:, :, 0]

        for ridx in rollout_indices:
            all_fields.append(pred[ridx])
            all_fields.append(target[ridx])
            all_errors.append(pred[ridx] - target[ridx])

    u_min = float(np.nanmin([np.nanmin(f) for f in all_fields]))
    u_max = float(np.nanmax([np.nanmax(f) for f in all_fields]))

    err_abs = float(np.nanmax([np.nanmax(np.abs(e)) for e in all_errors]))

    field_norm = colors.Normalize(vmin=u_min, vmax=u_max)
    error_norm = SymLogNorm(
        linthresh=1e-3,
        linscale=1.0,
        vmin=-err_abs,
        vmax=err_abs,
        base=10,
    )

    n_rows = len(selected) * len(rollout_indices)
    n_cols = 3

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(15, 3.2 * n_rows),
        sharex=True,
        sharey=True,
    )

    row = 0

    for sample_idx, group_label in selected:
        pred = ds_all["prediction"].isel(sample=sample_idx).values[:, :, 0]
        target = ds_all["target"].isel(sample=sample_idx).values[:, :, 0]

        for ridx in rollout_indices:
            fields = [
                ("Prediction", pred[ridx], "viridis", field_norm),
                ("Target", target[ridx], "viridis", field_norm),
                ("Error", pred[ridx] - target[ridx], "coolwarm", error_norm),
            ]

            for col, (title, field, cmap, norm) in enumerate(fields):
                ax = axes[row, col]

                lon_grid, lat_grid, field_grid = field_to_latlon_grid(
                    lon,
                    lat,
                    field,
                    n_lon=360,
                    n_lat=180,
                )

                sc = ax.imshow(
                    field_grid,
                    extent=[-180, 180, -90, 90],
                    origin="lower",
                    aspect="auto",
                    cmap=cmap,
                    norm=norm,
                )
                if row == 0:
                    ax.set_title(title, fontsize=13, fontweight="bold")

                if col == 0:
                    ax.set_ylabel(
                        f"{group_label}\nrollout {ridx + 1}\nlatitude [deg]"
                    )

                ax.set_xlim(-180, 180)
                ax.set_ylim(-90, 90)
                ax.grid(True, alpha=0.3)

                cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
                cbar.set_label("error" if title == "Error" else "u")

            row += 1

    for ax in axes[-1, :]:
        ax.set_xlabel("longitude [deg]")

    fig.suptitle(
        f"Best and worst RMSE trajectories projected to latitude-longitude "
        f"(train size {train_size})",
        fontsize=16,
    )

    fig.tight_layout()

    out_path = os.path.join(
        plot_dir,
        f"best_worst_rmse_latlon_snapshots_train{train_size}.png",
    )

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved figure to: {out_path}")


if __name__ == "__main__":
    train_size = 25

    ds_geo_dir = (
        "GNN_training/one_wave/nc_files/"
        "wave_200_ts_600_g4_sigmamin_6.nc"
    )

    raw_dir = (
        "GNN_training/one_wave/different_training_size/"
        f"test_{train_size}_results_new.zarr"
    )

    result_dir = (
        "GNN_training/one_wave/different_training_size/"
        f"test_{train_size}_results_new"
    )

    plot_dir = (
        "GNN_training/one_wave/different_training_size/"
        "all_results_plot/other_results"
    )

    plot_latlon_snapshots_best_worst_rmse(
        ds_geo_dir=ds_geo_dir,
        raw_dir=raw_dir,
        result_dir=result_dir,
        train_size=train_size,
        plot_dir=plot_dir,
        metric_key="rmse",
        rollout_indices=(0, 9, 19),
    )
