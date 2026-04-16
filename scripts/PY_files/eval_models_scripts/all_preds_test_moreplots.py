import os
import re
from time import time
import torch
import xarray as xr
import numpy as np
import sys

sys.path.insert(0, "./")
import scripts.PY_files.eval_models.helper_functions as helper
import scripts.PY_files.eval_models.plot_functions as plot_funcs
import matplotlib.pyplot as plt

from data_generation_functions import DataPlotter
from matplotlib import colors

plot_animations = False

# paths
ds_train = xr.open_dataset(r"./data/nc_files/wave_ensemble_100_coarse.nc")
raw_dir = r"./test_eval_same_mesh_10/raw_preds"
plot_dir = r"./results/evaluation/plots"
anim_dir = r"./results/evaluation/animations"

os.makedirs(plot_dir, exist_ok=True)
os.makedirs(anim_dir, exist_ok=True)

# Load all raw predictions
pred_files, target_files, time_files = helper.all_raw_files(raw_dir)

pred_all, target_all, time_all = helper.concat_all_batches(
    raw_dir, pred_files, target_files, time_files
)

print("combined pred shape:  ", tuple(pred_all.shape))
print("combined target shape:", tuple(target_all.shape))
print("combined time shape:  ", tuple(time_all.shape))

# 1-step data across all rollouts
uhat_1step, utrue_1step, err_1step, elapsed_time = helper.one_step_all_batches(
    pred_all, target_all, time_all
)

# 1-step scalar metrics over the full test set
errors_1step = helper.compute_errors(uhat_1step, utrue_1step, axis=1)

rmse_1step = errors_1step["rmse"]       # shape: (n_rollouts,)
mae_1step = errors_1step["mae"]         # shape: (n_rollouts,)
max_err_1step = errors_1step["err_max"] # shape: (n_rollouts,)

rel_rmse_1step = rmse_1step / (
    np.sqrt(np.mean(utrue_1step**2, axis=1)) + 1e-12
)

# Full rollout metrics
# Average over nodes and features, keep rollout and time
errors_rollout = helper.compute_errors(pred_all, target_all, axis=(2, 3))

# Geometry from original dataset
P = ds_train["P"].values
tri = ds_train["tri"].values
R = ds_train.attrs["R"]

# Xarray datasets for sphere animations
ds_pred = helper.setup_simple_xarray(uhat_1step, elapsed_time, P, tri, R=R)
ds_true = helper.setup_simple_xarray(utrue_1step, elapsed_time, P, tri, R=R)
ds_err = helper.setup_simple_xarray(err_1step, elapsed_time, P, tri, R=R)

if plot_animations:
    # Color scales
    field_norm = helper.color_scales(uhat_1step, utrue_1step)

    err_abs = float(np.nanmax(np.abs(err_1step)))
    err_norm = colors.Normalize(vmin=-err_abs, vmax=err_abs)

    # Sphere animations
    plotter_pred = DataPlotter.DataPlotter(ds=ds_pred)
    plotter_true = DataPlotter.DataPlotter(ds=ds_true)
    plotter_err = DataPlotter.DataPlotter(ds=ds_err)

    anim_pred = plotter_pred.animate_sphere(
        norm=field_norm,
        out_path=os.path.join(anim_dir, "eval_pred_all.gif"),
        fps=10,
    )

    anim_true = plotter_true.animate_sphere(
        norm=field_norm,
        out_path=os.path.join(anim_dir, "eval_true_all.gif"),
        fps=10,
    )

    anim_err = plotter_err.animate_sphere(
        norm=err_norm,
        out_path=os.path.join(anim_dir, "eval_error_all.gif"),
        fps=10,
    )

# Other plots
fig2 = plot_funcs.plot_error_metrics(
    rmse_1step, mae_1step, max_err_1step, rel_rmse_1step, elapsed_time
)
fig2.savefig(os.path.join(plot_dir, "plot_error_metrics_1_step.png"), dpi=200, bbox_inches="tight")

fig3 = plot_funcs.plot_error_histogram(err_1step)
fig3.savefig(os.path.join(plot_dir, "plot_error_histogram_1_step.png"), dpi=200, bbox_inches="tight")

fig4 = plot_funcs.plot_rmse_heatmap(pred_all, target_all)
fig4.savefig(os.path.join(plot_dir, "plot_rmse_heatmap_overall.png"), dpi=200, bbox_inches="tight")

fig5 = plot_funcs.plot_rollout_error_growth(pred_all, target_all)
fig5.savefig(os.path.join(plot_dir, "plot_rollout_error_growth_overall.png"), dpi=200, bbox_inches="tight")

fig6 = plot_funcs.plot_pred_vs_true_scatter(uhat_1step, utrue_1step)
fig6.savefig(os.path.join(plot_dir, "plot_pred_vs_true_scatter_1_step.png"), dpi=200, bbox_inches="tight")

fig7 = plot_funcs.plot_l2_energy(pred_all, target_all)
fig7.savefig(os.path.join(plot_dir, "plot_l2_energy_overall.png"), dpi=200, bbox_inches="tight")

fig8 = plot_funcs.plot_amplitude_over_time(pred_all, target_all)
fig8.savefig(os.path.join(plot_dir, "plot_amplitude_over_time_overall.png"), dpi=200, bbox_inches="tight")

fig9 = plot_funcs.plot_l2_energy_heatmap(pred_all, target_all)
fig9.savefig(os.path.join(plot_dir, "plot_l2_energy_heatmap_overall.png"), dpi=200, bbox_inches="tight")

#fig10 = plot_funcs.plot_correlation_over_time(pred_all, target_all)
#fig10.savefig(os.path.join(plot_dir, "plot_correlation_over_time_overall.png"), dpi=200, bbox_inches="tight")

plt.show()