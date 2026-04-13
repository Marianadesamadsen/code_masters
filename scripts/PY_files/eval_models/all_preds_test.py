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

ds_train = xr.open_dataset(r"./data/nc_files/wave_ensemble_100_coarse.nc")
raw_dir = r"./test_eval_same_mesh_10/raw_preds"

pred_files, target_files, time_files = helper.all_raw_files(raw_dir)

pred_all, target_all, time_all = helper.concat_all_batches(raw_dir, pred_files, target_files, time_files)

print("combined pred shape:  ", tuple(pred_all.shape))
print("combined target shape:", tuple(target_all.shape))
print("combined time shape:  ", tuple(time_all.shape))

uhat_1step, utrue_1step, err_1step, elapsed_time = helper.one_step_all_batches(pred_all, target_all, time_all)

errors = helper.compute_errors(pred_all, target_all, axis=1)

# geometry from original dataset
P = ds_train["P"].values
tri = ds_train["tri"].values
R = ds_train.attrs["R"]

# Xarray datasets for plotting
ds_pred = helper.setup_simple_xarray(uhat_1step, elapsed_time, P, tri, R=1)
ds_true = helper.setup_simple_xarray(utrue_1step, elapsed_time, P, tri, R=1)
ds_err = helper.setup_simple_xarray(err_1step, elapsed_time, P, tri, R=1)
 
# Color scales for consistent plotting
field_norm = helper.color_scales(uhat_1step, utrue_1step)

err_abs = float(np.nanmax(np.abs(err_1step)))
err_norm = colors.Normalize(vmin=-err_abs, vmax=err_abs)

plotter_pred = DataPlotter.DataPlotter(ds=ds_pred)
plotter_true = DataPlotter.DataPlotter(ds=ds_true)
plotter_err = DataPlotter.DataPlotter(ds=ds_err)

anim_pred = plotter_pred.animate_sphere(
    norm=field_norm,
    out_path="data/animation/test_animations/eval_pred_all.gif",
    fps=10,
)

anim_true = plotter_true.animate_sphere(
    norm=field_norm,
    out_path="data/animation/test_animations/eval_true_all.gif",
    fps=10,
)

anim_err = plotter_err.animate_sphere(
    norm=err_norm,
    out_path="data/animation/test_animations/eval_error_all.gif",
    fps=10,
)

plt.show()