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


def get_best_worst_saved_sample_indices(raw_dir, result_dir, metric_key="rmse"):
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

    print(f"\n{result_dir}")
    print("Best saved sample:", best_idx)
    print(metadata.iloc[best_idx])
    print("Mean score:", mean_scores.iloc[best_idx])

    print("Worst saved sample:", worst_idx)
    print(metadata.iloc[worst_idx])
    print("Mean score:", mean_scores.iloc[worst_idx])

    return best_idx, worst_idx

def get_best_worst_reference_samples(
    raw_dir,
    result_dir,
    metric_key="rmse",
):
    ds_all = xr.open_zarr(raw_dir)

    metric_df = pd.read_csv(
        os.path.join(result_dir, f"test_{metric_key}_per_sample.csv")
    )
    metadata = pd.read_csv(
        os.path.join(result_dir, "test_metadata.csv")
    )

    n_saved_samples = ds_all.sizes["sample"]

    metric_df = metric_df.iloc[:n_saved_samples].reset_index(drop=True)
    metadata = metadata.iloc[:n_saved_samples].reset_index(drop=True)

    mean_scores = metric_df.mean(axis=1)

    best_idx = int(mean_scores.idxmin())
    worst_idx = int(mean_scores.idxmax())

    selected = {}

    for label, idx in [("Best", best_idx), ("Worst", worst_idx)]:
        row = metadata.iloc[idx]

        selected[label] = {
            "ensemble_member": int(row["ensemble_member"]),
            "sample_idx": int(row["sample_idx"]),
            "reference_zarr_sample": idx,
            "score": float(mean_scores.iloc[idx]),
        }

        print(f"\nReference {label}:")
        print(selected[label])

    return selected


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


def plot_error_heatmaps_best_worst_training_sizes(
    ds_geo_dir,
    run_configs,
    plot_dir,
    metric_key="rmse",
    reference_train_size=75,
    rollout_indices=(0, 9, 19),
    linthresh=1e-3,
):
    os.makedirs(plot_dir, exist_ok=True)

    ds_geo = xr.open_dataset(ds_geo_dir)
    sigmas = ds_geo["sigma_deg"].values
    amplitudes = ds_geo["A"].values
    P = ds_geo["P"].values
    lon, lat = get_lon_lat_from_P(P)

    reference_cfg = run_configs[reference_train_size]

    reference_samples = get_best_worst_reference_samples(
        raw_dir=reference_cfg["raw_dir"],
        result_dir=reference_cfg["result_dir"],
        metric_key=metric_key,
    )

    selected_samples = {}
    all_errors = []

    # Find same physical trajectories in all train-size runs
    for train_size, cfg in run_configs.items():
        ds_all = xr.open_zarr(cfg["raw_dir"])
        selected_samples[train_size] = {}

        for group_label, ref in reference_samples.items():
            sample_idx = find_matching_zarr_sample(
                raw_dir=cfg["raw_dir"],
                result_dir=cfg["result_dir"],
                ensemble_member=ref["ensemble_member"],
                sample_idx=ref["sample_idx"],
            )

            selected_samples[train_size][group_label] = sample_idx

            if sample_idx is None:
                print(
                    f"Missing {group_label} trajectory in train {train_size}: "
                    f"ensemble_member={ref['ensemble_member']}, "
                    f"sample_idx={ref['sample_idx']}"
                )
                continue

            pred = ds_all["prediction"].isel(sample=sample_idx).values[:, :, 0]
            target = ds_all["target"].isel(sample=sample_idx).values[:, :, 0]

            for ridx in rollout_indices:
                all_errors.append(pred[ridx] - target[ridx])

    err_abs = float(np.nanmax([np.nanmax(np.abs(e)) for e in all_errors]))

    error_norm = SymLogNorm(
        linthresh=linthresh,
        linscale=1.0,
        vmin=-err_abs,
        vmax=err_abs,
        base=10,
    )

    row_labels = []
    for group_label in ["Best", "Worst"]:
        for ridx in rollout_indices:
            row_labels.append((group_label, ridx))

    train_sizes = list(run_configs.keys())

    n_rows = len(row_labels)
    n_cols = len(train_sizes)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.2 * n_cols, 2.8 * n_rows),
        sharex=True,
        sharey=True,
    )

    if n_cols == 1:
        axes = axes[:, None]

    im = None

    for col, train_size in enumerate(train_sizes):
        cfg = run_configs[train_size]
        ds_all = xr.open_zarr(cfg["raw_dir"])

        axes[0, col].set_title(
            f"Train size {train_size}",
            fontsize=13,
            fontweight="bold",
        )

        for row, (group_label, ridx) in enumerate(row_labels):
            ax = axes[row, col]

            sample_idx = selected_samples[train_size][group_label]

            if sample_idx is None:
                ax.axis("off")
                continue

            pred = ds_all["prediction"].isel(sample=sample_idx).values[:, :, 0]
            target = ds_all["target"].isel(sample=sample_idx).values[:, :, 0]
            error = pred[ridx] - target[ridx]

            _, _, error_grid = field_to_latlon_grid(
                lon,
                lat,
                error,
                n_lon=360,
                n_lat=180,
            )

            im = ax.imshow(
                error_grid,
                extent=[-180, 180, -90, 90],
                origin="lower",
                aspect="auto",
                cmap="coolwarm",
                norm=error_norm,
            )

            if col == 0:
                ref = reference_samples[group_label]

                metadata_ref = pd.read_csv(
                    os.path.join(
                        run_configs[reference_train_size]["result_dir"],
                        "test_metadata.csv",
                    )
                )

                ref_row = metadata_ref.iloc[ref["reference_zarr_sample"]]

                sigma = sigmas[ref["ensemble_member"]]
                amplitude = amplitudes[ref["ensemble_member"]]

                ax.set_ylabel(
                    f"{group_label}\n"
                    f"$\\sigma$={sigma:.2f} deg\n"
                    f"A={amplitude:.2f}\n"
                    f"wave {ref['ensemble_member']}, "
                    f"start {ref['sample_idx']}\n"
                    f"rollout {ridx + 1}\n"
                    f"lat [deg]"
                )

            if row == n_rows - 1:
                ax.set_xlabel("lon [deg]")

            ax.set_xlim(-180, 180)
            ax.set_ylim(-90, 90)
            ax.grid(True, alpha=0.25)

    cbar = fig.colorbar(
        im,
        ax=axes,
        shrink=0.85,
        pad=0.015,
    )
    cbar.set_label("Prediction error")

    fig.suptitle(
        f"Prediction error for best/worst {metric_key} trajectories "
        f"selected from train size {reference_train_size}",
        fontsize=16,
    )

    fig.tight_layout()

    out_path = os.path.join(
        plot_dir,
        f"best_worst_{metric_key}_error_heatmaps_reference_train"
        f"{reference_train_size}.png",
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

    plot_error_heatmaps_best_worst_training_sizes(
        ds_geo_dir=ds_geo_dir,
        run_configs=run_configs,
        plot_dir=plot_dir,
        metric_key="rmse",
        reference_train_size=25,
        rollout_indices=(0, 9, 19),
        linthresh=1e-3,
    )   