import os
import re
from time import time
import torch
import xarray as xr
import numpy as np
import sys

sys.path.insert(0, "./")
import scripts.PY_files.eval_models_scripts.helper_functions_ensemble as helper
import scripts.PY_files.eval_models_scripts.plot_functions as plot_funcs
import matplotlib.pyplot as plt
from integrate_sphere.compute_energy import compute_energy_over_time
from data_generation_functions import DataPlotterAll
from matplotlib import colors

def plot_results(ds_geo_dir,raw_dir,plot_dir,anim_dir, generations, plot_animations_1step = False, plot_animations_rollout = True,azim=160, elev=20):
    
    ds_geo = xr.open_dataset(ds_geo_dir)

    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(anim_dir, exist_ok=True)

    # Load all raw predictions
    ds_all = xr.open_zarr(raw_dir)

    pred_all = ds_all["prediction"].values
    target_all = ds_all["target"].values
    time_all = ds_all["valid_time"].values
    rollout_steps = ds_all.rollout_step.values
    num_samples = ds_all.sizes["sample"]
    num_nodes = ds_all.sizes["grid_index"]
    num_features = ds_all.sizes["state_feature"]

    # Geometry from original dataset
    P = ds_geo["P"].values
    tri = ds_geo["tri"].values
    R = ds_geo.attrs["R"]

    #####################################################################

    # Full rollout metrics
    # Average over nodes and features, keep rollout and time
    errors_dict = helper.compute_errors(pred_all, target_all, generations, axis=(2, 3))

    # 1-step 1-wave 1-feature
    pred_1step_1wave = pred_all[:, 0, :, 0]
    target_1step_1wave = target_all[:, 0, :, 0]
    time_1step_1wave = time_all[:, 0]
    error_1step_1wave = pred_1step_1wave - target_1step_1wave

    # Plotting error metrics over time 1-step 1-wave
    # fig_errors = plot_funcs.plot_error_metrics(errors_dict["rmse"][:,:].mean(axis=1), errors_dict["mae"][:,:].mean(axis=1), errors_dict["err_max"][:, :].mean(axis=1), time_1step_1wave)
    # fig_errors.savefig(os.path.join(plot_dir, f"error_metrics_1_step.png"), dpi=200, bbox_inches="tight")

    # fig_hist = plot_funcs.plot_error_histogram(error_1step_1wave)
    # fig_hist.savefig(os.path.join(plot_dir, f"error_histogram_1_step.png"), dpi=200, bbox_inches="tight")

    # fig_rmse_heatmap = plot_funcs.plot_rmse_heatmap(errors_dict["rmse"][:, :])
    # fig_rmse_heatmap.savefig(os.path.join(plot_dir, f"heatmap_rmse.png"), dpi=200, bbox_inches="tight")

    # fig_rollout_error = plot_funcs.plot_rollout_error_growth(errors_dict["rmse"][:, :].mean(axis=0), errors_dict["mae"][:, :].mean(axis=0))
    # fig_rollout_error.savefig(os.path.join(plot_dir, f"rollout_error_growth_wave.png"), dpi=200, bbox_inches="tight")

    # fig_max_error = plot_funcs.plot_max_over_time(errors_dict["max_pred"][:, :].mean(axis=0), errors_dict["max_true"][:, :].mean(axis=0))
    # fig_max_error.savefig(os.path.join(plot_dir, f"plot_max_over_time_overall.png"), dpi=200, bbox_inches="tight")

    # Computing energy
    pred_energy_input = pred_all[:, :, :, 0].mean(axis=0)
    target_energy_input = target_all[:, :, :, 0].mean(axis=0)

    if isinstance(pred_energy_input, torch.Tensor):
        pred_energy_input = pred_energy_input.detach().cpu().numpy()

    if isinstance(target_energy_input, torch.Tensor):
        target_energy_input = target_energy_input.detach().cpu().numpy()

    dt = float(ds_geo.attrs["dt"])
    E_pred = compute_energy_over_time(
        pred_energy_input,
        generation=generations,
        R=1,
        c=1,
        N=6,
        dt=dt,
    )
    E_target = compute_energy_over_time(
        target_energy_input,
        generation=generations,
        R=1,
        c=1,
        N=6,
        dt=dt,
    )

    # fig_energy = plot_funcs.plot_energy_over_time(E_pred, E_target)
    # fig_energy.savefig(os.path.join(plot_dir, f"energy_over_time.png"))

    plt.close("all")

    if plot_animations_1step:

        # This setup is for dataplotter 3Ds
        ds_pred = helper.setup_simple_xarray(pred_1step_1wave[:100,:], time_1step_1wave[:100], P, tri, R=R)
        ds_true = helper.setup_simple_xarray(target_1step_1wave[:100,:], time_1step_1wave[:100], P, tri, R=R)
        ds_err = helper.setup_simple_xarray(error_1step_1wave[:100,:], time_1step_1wave[:100], P, tri, R=R)

        # Color scales
        field_norm = helper.color_scales(pred_1step_1wave, target_1step_1wave)

        err_abs = float(np.nanmax(np.abs(error_1step_1wave)))
        err_norm = colors.Normalize(vmin=-err_abs, vmax=err_abs)


    if plot_animations_rollout:
        pred_all_1feature = pred_all[:,:,:,0]
        target_all_1feature = target_all[:,:,:,0]
        error_all_1feature = pred_all_1feature - target_all_1feature
        rolloutidx = 0

        # This setup is for dataplotter 3Ds
        ds_pred = helper.setup_simple_xarray(pred_all_1feature[rolloutidx], rollout_steps, P, tri, R=R)
        ds_true = helper.setup_simple_xarray(target_all_1feature[rolloutidx], rollout_steps, P, tri, R=R)
        ds_err = helper.setup_simple_xarray(error_all_1feature[rolloutidx], rollout_steps, P, tri, R=R)

        # Color scales
        field_norm = helper.color_scales(pred_all_1feature[rolloutidx], target_all_1feature[rolloutidx])

        err_abs = float(np.nanmax(np.abs(error_all_1feature[rolloutidx])))
        err_norm = colors.Normalize(vmin=-err_abs, vmax=err_abs)

        # Sphere animations
        plotter = DataPlotterAll.DataPlotter(ds=ds_true)

        anim = plotter.animate_three_spheres(
            ds_pred=ds_pred,
            ds_target=ds_true,
            ds_error=ds_err,
            out_path=os.path.join(anim_dir, f"result_rollout_idx{rolloutidx}.mp4"),
            fps=10,
            interval=100,
            pred_target_cmap="viridis",
            error_cmap="coolwarm",
            pred_target_norm=field_norm,
            error_norm=err_norm,
            titles=("Prediction", "Target", "Error"),
            colorbar_label="u",
            azim =azim, elev=elev
        )

