import os
import sys
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, "./")

import scripts.PY_files.eval_models_scripts.plot_animations as plot_animations
import scripts.PY_files.eval_models_scripts.helper_functions_ensemble as helper


def plot_best_worst_rmse_trajectories(
    ds_geo_dir,
    raw_dir,
    result_dir,
    train_size,
    plot_dir,
    anim_dir,
    metric_key="rmse",
    azim=135,
    elev=30,
):
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(anim_dir, exist_ok=True)

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
    tri = ds_geo["tri"].values
    R = ds_geo.attrs["R"]
    rollout_steps = ds_all["rollout_step"].values

    rows = []

    for sample_idx, label in [
        (best_idx, "Best RMSE trajectory"),
        (worst_idx, "Worst RMSE trajectory"),
    ]:
        pred = ds_all["prediction"].isel(sample=sample_idx).values[:, :, 0]
        target = ds_all["target"].isel(sample=sample_idx).values[:, :, 0]
        error = pred - target

        ds_pred = helper.setup_simple_xarray(pred, rollout_steps, P, tri, R=R)
        ds_target = helper.setup_simple_xarray(target, rollout_steps, P, tri, R=R)
        ds_error = helper.setup_simple_xarray(error, rollout_steps, P, tri, R=R)

        rows.append((ds_pred, ds_target, ds_error, label))

    out_path = os.path.join(
        anim_dir,
        f"best_worst_rmse_train{train_size}.mp4",
    )

    anim = plot_animations.animate_sphere_rows(
        rows=rows,
        out_path=out_path,
        fps=10,
        interval=100,
        pred_target_cmap="viridis",
        error_cmap="coolwarm",
        pred_target_norm=None,
        error_norm=None,
        titles=("Prediction", "Target", "Error"),
        colorbar_label="u",
        azim=azim,
        elev=elev,
    )

    print(f"\nSaved animation to: {out_path}")
    return anim


if __name__ == "__main__":
    train_size = 75

    ds_geo_dir = (
        "GNN_training/one_wave/nc_files/"
        "wave_200_ts_600_g4_sigmamin_6.nc"
    )

    raw_dir = (
        "GNN_training/one_wave/different_training_size/"
        "test_75_results_100.zarr"
    )

    result_dir = (
        "GNN_training/one_wave/different_training_size/"
        "test_75_results_100"
    )

    plot_dir = (
        "GNN_training/one_wave/different_training_size/"
        "all_results_plot/other_results"
    )

    anim_dir = (
        "GNN_training/one_wave/different_training_size/"
        "all_results_plot/animations100"
    )

    plot_best_worst_rmse_trajectories(
        ds_geo_dir=ds_geo_dir,
        raw_dir=raw_dir,
        result_dir=result_dir,
        train_size=train_size,
        plot_dir=plot_dir,
        anim_dir=anim_dir,
        metric_key="rmse",
        azim=135,
        elev=30,
    )