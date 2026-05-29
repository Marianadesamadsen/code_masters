import os
import sys
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, "./")

import scripts.PY_files.eval_models_scripts.plot_animations as plot_animations
import scripts.PY_files.eval_models_scripts.helper_functions_ensemble as helper


def add_best_worst_rows(
    rows,
    ds_geo,
    raw_dir,
    result_dir,
    row_prefix,
    metric_key="rmse",
):
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

    print(f"\n{row_prefix} best saved RMSE trajectory:")
    print("Zarr sample:", best_idx)
    print(metadata.iloc[best_idx])
    print("Mean RMSE:", mean_scores.iloc[best_idx])

    print(f"\n{row_prefix} worst saved RMSE trajectory:")
    print("Zarr sample:", worst_idx)
    print(metadata.iloc[worst_idx])
    print("Mean RMSE:", mean_scores.iloc[worst_idx])

    P = ds_geo["P"].values
    tri = ds_geo["tri"].values
    R = ds_geo.attrs["R"]
    rollout_steps = ds_all["rollout_step"].values

    for sample_idx, label in [
        (best_idx, f"{row_prefix}: Best RMSE"),
        (worst_idx, f"{row_prefix}: Worst RMSE"),
    ]:
        pred = ds_all["prediction"].isel(sample=sample_idx).values[:, :, 0]
        target = ds_all["target"].isel(sample=sample_idx).values[:, :, 0]
        error = pred - target

        ds_pred = helper.setup_simple_xarray(pred, rollout_steps, P, tri, R=R)
        ds_target = helper.setup_simple_xarray(target, rollout_steps, P, tri, R=R)
        ds_error = helper.setup_simple_xarray(error, rollout_steps, P, tri, R=R)

        rows.append((ds_pred, ds_target, ds_error, label))


def plot_best_worst_rmse_mp1_mp2(
    ds_geo_dir,
    mp_runs,
    plot_dir,
    anim_dir,
    metric_key="rmse",
    azim=135,
    elev=30,
):
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(anim_dir, exist_ok=True)

    ds_geo = xr.open_dataset(ds_geo_dir)

    rows = []

    for row_prefix, paths in mp_runs.items():
        add_best_worst_rows(
            rows=rows,
            ds_geo=ds_geo,
            raw_dir=paths["raw_dir"],
            result_dir=paths["result_dir"],
            row_prefix=row_prefix,
            metric_key=metric_key,
        )

    out_path = os.path.join(
        anim_dir,
        f"best_worst_{metric_key}_mp1_mp2.mp4",
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
    ds_geo_dir = (
        "GNN_training/one_wave/nc_files/"
        "wave_200_ts_600_g4_sigmamin_6.nc"
    )

    plot_dir = (
        "GNN_training/one_wave/different_mp/"
        "results/other_results"
    )

    anim_dir = (
        "GNN_training/one_wave/different_mp/"
        "results/animations"
    )

    mp_runs = {
        "MP1": {
            "raw_dir": (
                "GNN_training/one_wave/different_mp/"
                "test_mp1_results_new.zarr"
            ),
            "result_dir": (
                "GNN_training/one_wave/different_mp/"
                "test_mp1_results_new"
            ),
        },
        "MP2": {
            "raw_dir": (
                "GNN_training/one_wave/different_mp/"
                "test_mp2_results_new.zarr"
            ),
            "result_dir": (
                "GNN_training/one_wave/different_mp/"
                "test_mp2_results_new"
            ),
        },
    }

    plot_best_worst_rmse_mp1_mp2(
        ds_geo_dir=ds_geo_dir,
        mp_runs=mp_runs,
        plot_dir=plot_dir,
        anim_dir=anim_dir,
        metric_key="rmse",
        azim=135,
        elev=30,
    )