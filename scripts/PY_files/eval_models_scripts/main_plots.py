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

from data_generation_functions import DataPlotter
from matplotlib import colors

plot_animations = True

# paths
ds_geo = xr.open_dataset(r"./data/nc_files/start_up_tests/wave_ensemble_10_coarse.nc")
raw_dir = r"./evaluation_results/waves10_test/raw_preds"
plot_dir = r"./results/test2/plots"
anim_dir = r"./results/test2/animations"

os.makedirs(plot_dir, exist_ok=True)
os.makedirs(anim_dir, exist_ok=True)

# Load all raw predictions
pred_files, target_files, time_files = helper.all_raw_files(raw_dir)

pred_all, target_all, time_all = helper.concat_all_batches(
    raw_dir, pred_files, target_files, time_files
)

# Split up to waves and time steps
num_time_steps_all, num_roll_outs, num_nodes, num_features = pred_all.shape
num_waves = int(np.ceil(len(pred_all)/len(ds_geo["time"])))
num_time_steps = int(len(pred_all)/num_waves) #ds_geo.attrs["tmax"]/ds_geo.attrs["dt"]

pred_all = pred_all.reshape(num_time_steps, num_waves, num_roll_outs, num_nodes, num_features)
target_all = target_all.reshape(num_time_steps, num_waves, num_roll_outs, num_nodes, num_features)
time_all = time_all.reshape(num_time_steps, num_waves, num_roll_outs)

print("combined pred shape:  ", tuple(pred_all.shape))
print("combined target shape:", tuple(target_all.shape))
print("combined time shape:  ", tuple(time_all.shape))

# Geometry from original dataset
P = ds_geo["P"].values
tri = ds_geo["tri"].values
R = ds_geo.attrs["R"]

#####################################################################

# Full rollout metrics
# Average over nodes and features, keep rollout and time
errors_dict = helper.compute_errors(pred_all, target_all, axis=(3, 4))

for wave in range(num_waves):

    pred_1wave = pred_all[:,wave,:,:,:]
    target_1wave = target_all[:,wave,:,:,:]
    time_1wave = time_all[:,wave,:]

    # 1-step 1-wave 1-feature
    pred_1step_1wave = pred_1wave[:, 0, :, 0]
    target_1step_1wave = target_1wave[:, 0, :, 0]
    time_1step_1wave = time_1wave[:, 0]
    error_1step_1wave = pred_1step_1wave - target_1step_1wave

    # This setup is for dataplotter 3Ds
    ds_pred = helper.setup_simple_xarray(pred_1step_1wave[:100,:], time_1step_1wave[:100], P, tri, R=R)
    ds_true = helper.setup_simple_xarray(target_1step_1wave[:100,:], time_1step_1wave[:100], P, tri, R=R)
    ds_err = helper.setup_simple_xarray(error_1step_1wave[:100,:], time_1step_1wave[:100], P, tri, R=R)

    # Plotting error metrics over time 1-step 1-wave
    fig_errors = plot_funcs.plot_error_metrics(errors_dict["rmse"][:,wave,:].mean(axis=1), errors_dict["mae"][:,wave,:].mean(axis=1), errors_dict["err_max"][:,wave,:].mean(axis=1), time_1step_1wave)
    fig_errors.savefig(os.path.join(plot_dir, f"error_metrics_1_step_wave{wave}.png"), dpi=200, bbox_inches="tight")

    fig_hist = plot_funcs.plot_error_histogram(error_1step_1wave)
    fig_hist.savefig(os.path.join(plot_dir, f"error_histogram_1_step_wave{wave}.png"), dpi=200, bbox_inches="tight")

    fig_rmse_heatmap = plot_funcs.plot_rmse_heatmap(errors_dict["rmse"][:, wave, :])
    fig_rmse_heatmap.savefig(os.path.join(plot_dir, f"heatmap_rmse_wave{wave}.png"), dpi=200, bbox_inches="tight")

    fig_rollout_error = plot_funcs.plot_rollout_error_growth(errors_dict["rmse"][:, wave, :].mean(axis=0), errors_dict["mae"][:, wave, :].mean(axis=0))
    fig_rollout_error.savefig(os.path.join(plot_dir, f"rollout_error_growth_wave{wave}.png"), dpi=200, bbox_inches="tight")

    fig_heatmap_L2norm = plot_funcs.plot_L2_norm_heatmap(errors_dict["L2_error"][:, wave, :])
    fig_heatmap_L2norm.savefig(os.path.join(plot_dir, f"heatmap_L2_norm_wave{wave}.png"), dpi=200, bbox_inches="tight")

    fig_L2_norm = plot_funcs.plot_L2_norm(errors_dict["L2_pred"][:, wave, :].mean(axis=1), errors_dict["L2_true"][:, wave, :].mean(axis=1))
    fig_L2_norm.savefig(os.path.join(plot_dir, f"plot_l2_energy_overall_wave{wave}.png"), dpi=200, bbox_inches="tight")

    fig_max_error = plot_funcs.plot_max_over_time(errors_dict["max_pred"][:, wave, :].mean(axis=0), errors_dict["max_true"][:, wave, :].mean(axis=0))
    fig_max_error.savefig(os.path.join(plot_dir, f"plot_max_over_time_overall_wave{wave}.png"), dpi=200, bbox_inches="tight")


    if plot_animations:
        # Color scales
        field_norm = helper.color_scales(pred_1step_1wave, target_1step_1wave)

        err_abs = float(np.nanmax(np.abs(error_1step_1wave)))
        err_norm = colors.Normalize(vmin=-err_abs, vmax=err_abs)

        # Sphere animations
        plotter_pred = DataPlotter.DataPlotter(ds=ds_pred)
        plotter_true = DataPlotter.DataPlotter(ds=ds_true)
        plotter_err = DataPlotter.DataPlotter(ds=ds_err)

        anim_pred = plotter_pred.animate_sphere(
            norm=field_norm,
            out_path=os.path.join(anim_dir, f"pred_all_wave{wave}.gif"),
            fps=10,
        )

        anim_true = plotter_true.animate_sphere(
            norm=field_norm,
            out_path=os.path.join(anim_dir, f"true_all_wave{wave}.gif"),
            fps=10,
        )

        anim_err = plotter_err.animate_sphere(
            norm=err_norm,
            out_path=os.path.join(anim_dir, f"error_all_wave{wave}.gif"),
            fps=10,
        )


