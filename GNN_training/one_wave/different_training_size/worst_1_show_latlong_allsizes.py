import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
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


def get_worst_saved_sample_index(raw_dir, result_dir, metric_key="rmse"):
    ds_all = xr.open_zarr(raw_dir)

    metric_path = os.path.join(result_dir, f"test_{metric_key}_per_sample.csv")
    metadata_path = os.path.join(result_dir, "test_metadata.csv")

    metric_df = pd.read_csv(metric_path)
    metadata = pd.read_csv(metadata_path)

    n_saved_samples = ds_all.sizes["sample"]

    metric_df = metric_df.iloc[:n_saved_samples].reset_index(drop=True)
    metadata = metadata.iloc[:n_saved_samples].reset_index(drop=True)

    mean_scores = metric_df.mean(axis=1)

    worst_idx = int(mean_scores.idxmax())
    worst_row = metadata.iloc[worst_idx]

    print("\nWorst reference sample:")
    print("saved sample index:", worst_idx)
    print(worst_row)
    print("mean score:", mean_scores.iloc[worst_idx])

    return worst_idx, worst_row, float(mean_scores.iloc[worst_idx])


def find_matching_zarr_sample(raw_dir, result_dir, ensemble_member, sample_idx):
    ds_all = xr.open_zarr(raw_dir)

    metadata = pd.read_csv(
        os.path.join(result_dir, "test_metadata.csv")
    )

    n_saved_samples = ds_all.sizes["sample"]
    metadata = metadata.iloc[:n_saved_samples].reset_index(drop=True)

    matches = np.where(
        (metadata["ensemble_member"].to_numpy() == ensemble_member)
        & (metadata["sample_idx"].to_numpy() == sample_idx)
    )[0]

    if len(matches) == 0:
        return None

    return int(matches[0])


def plot_pred_target_error_for_worst_train1(
    ds_geo_dir,
    run_configs,
    plot_dir,
    metric_key="rmse",
    reference_train_size=1,
    rollout_index=19,
    linthresh=1e-3,
):
    os.makedirs(plot_dir, exist_ok=True)

    ds_geo = xr.open_dataset(ds_geo_dir)

    sigmas = ds_geo["sigma_deg"].values
    amplitudes = ds_geo["A"].values
    P = ds_geo["P"].values

    lon, lat = get_lon_lat_from_P(P)

    reference_cfg = run_configs[reference_train_size]

    worst_idx, worst_row, worst_score = get_worst_saved_sample_index(
        raw_dir=reference_cfg["raw_dir"],
        result_dir=reference_cfg["result_dir"],
        metric_key=metric_key,
    )

    ensemble_member = int(worst_row["ensemble_member"])
    sample_idx = int(worst_row["sample_idx"])

    sigma = sigmas[ensemble_member]
    amplitude = amplitudes[ensemble_member]


    ds_reference = xr.open_zarr(reference_cfg["raw_dir"])

    common_target = (
        ds_reference["target"]
        .isel(sample=worst_idx)
        .values[:, :, 0]
    )

    common_target_field = common_target[rollout_index]

    selected_samples = {}

    all_pred_values = []
    all_errors = []

    for train_size, cfg in run_configs.items():
        match_idx = find_matching_zarr_sample(
            raw_dir=cfg["raw_dir"],
            result_dir=cfg["result_dir"],
            ensemble_member=ensemble_member,
            sample_idx=sample_idx,
        )

        selected_samples[train_size] = match_idx

        if match_idx is None:
            print(f"Missing matching sample for train size {train_size}")
            continue

        ds_all = xr.open_zarr(cfg["raw_dir"])

        pred = ds_all["prediction"].isel(sample=match_idx).values[:, :, 0]
        pred_field = pred[rollout_index]

        error_field = common_target_field - pred_field 

        all_pred_values.append(pred_field)
        all_errors.append(error_field)

    field_abs = float(
        np.nanmax(
            [
                np.nanmax(np.abs(v))
                for v in all_pred_values + [common_target_field]
            ]
        )
    )

    err_abs = float(
        np.nanmax(
            [
                np.nanmax(np.abs(e))
                for e in all_errors
            ]
        )
    )

    error_norm = SymLogNorm(
        linthresh=linthresh,
        linscale=1.0,
        vmin=-err_abs,
        vmax=err_abs,
        base=10,
    )

    train_sizes = list(run_configs.keys())

    fig, axes = plt.subplots(
        len(train_sizes),
        3,
        figsize=(15, 4 * len(train_sizes)),
        sharex=True,
        sharey=True,
    )

    if len(train_sizes) == 1:
        axes = axes[None, :]

    im_field = None
    im_error = None

    for row, train_size in enumerate(train_sizes):
        cfg = run_configs[train_size]
        match_idx = selected_samples[train_size]

        if match_idx is None:
            for col in range(3):
                axes[row, col].axis("off")
            continue

        ds_all = xr.open_zarr(cfg["raw_dir"])

        pred = ds_all["prediction"].isel(sample=match_idx).values[:, :, 0]

        pred_field = pred[rollout_index]
        target_field = common_target_field
        error_field = target_field - pred_field

        fields = [
            ("Prediction", pred_field),
            ("Target", target_field),
            ("Error", error_field),
        ]

        for col, (title, field) in enumerate(fields):
            ax = axes[row, col]

            _, _, field_grid = field_to_latlon_grid(
                lon,
                lat,
                field,
                n_lon=360,
                n_lat=180,
            )

            if title == "Error":
                im = ax.imshow(
                    field_grid,
                    extent=[-180, 180, -90, 90],
                    origin="lower",
                    aspect="auto",
                    cmap="coolwarm",
                    norm=error_norm,
                )

                cbar = fig.colorbar(
                    im,
                    ax=ax,
                    fraction=0.046,
                    pad=0.04,
                )
                cbar.set_label("Error",fontsize=14)

            else:
                im = ax.imshow(
                    field_grid,
                    extent=[-180, 180, -90, 90],
                    origin="lower",
                    aspect="auto",
                    cmap="viridis",
                    vmin=-field_abs,
                    vmax=field_abs,
                )

                cbar = fig.colorbar(
                    im,
                    ax=ax,
                    fraction=0.046,
                    pad=0.04,
                )
                cbar.set_label("u",fontsize=14)

            if row == 0:
                ax.set_title(title, fontsize=20, fontweight="bold")

            if col == 0:
                ax.set_ylabel(
                    f"Train size {train_size}\nlat [deg]",
                    fontsize=20,
                )

            if row == len(train_sizes) - 1:
                ax.set_xlabel("lon [deg]",fontsize=20)

            ax.set_xlim(-180, 180)
            ax.set_ylim(-90, 90)
            ax.grid(True, alpha=0.25)

    fig.suptitle(
        f"Worst {metric_key.upper()} trajectory from train size {reference_train_size}\n"
        f"$\\sigma$={sigma:.2f} deg, A={amplitude:.2f}, "
        f"rollout {rollout_index + 1}, ",
        fontsize=22,
    )

    fig.tight_layout()

    out_path = os.path.join(
        plot_dir,
        f"worst_train{reference_train_size}_{metric_key}_"
        f"pred_target_error_rollout{rollout_index + 1}.png",
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
        "GNN_training/one_wave/different_training_size/"
        "all_results_plot/other_results"
    )

    run_configs = {
        1: {
            "raw_dir": (
                "GNN_training/one_wave/different_training_size/"
                "test_1_results_new.zarr"
            ),
            "result_dir": (
                "GNN_training/one_wave/different_training_size/"
                "test_1_results_new"
            ),
        },
        25: {
            "raw_dir": (
                "GNN_training/one_wave/different_training_size/"
                "test_25_results_new.zarr"
            ),
            "result_dir": (
                "GNN_training/one_wave/different_training_size/"
                "test_25_results_new"
            ),
        },
        50: {
            "raw_dir": (
                "GNN_training/one_wave/different_training_size/"
                "test_50_results_new.zarr"
            ),
            "result_dir": (
                "GNN_training/one_wave/different_training_size/"
                "test_50_results_new"
            ),
        },
        75: {
            "raw_dir": (
                "GNN_training/one_wave/different_training_size/"
                "test_75_results_new.zarr"
            ),
            "result_dir": (
                "GNN_training/one_wave/different_training_size/"
                "test_75_results_new"
            ),
        },
    }

    plot_pred_target_error_for_worst_train1(
        ds_geo_dir=ds_geo_dir,
        run_configs=run_configs,
        plot_dir=plot_dir,
        metric_key="rmse",
        reference_train_size=1,
        rollout_index=19,
        linthresh=1e-3,
    )