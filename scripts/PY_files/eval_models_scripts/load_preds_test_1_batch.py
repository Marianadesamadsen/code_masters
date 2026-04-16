import os
import torch
import xarray as xr
import numpy as np
import sys
sys.path.insert(0, "./")
import matplotlib.pyplot as plt

from data_generation_functions import DataPlotter
from matplotlib import colors

# geometry
ds_train = xr.open_dataset(r"./data/nc_files/wave_ensemble_100_coarse.nc")

# load one saved batch
raw_dir = r"./test_eval_same_mesh_10/raw_preds"

pred = torch.load(os.path.join(raw_dir, "pred_batch_10.pt"))
target = torch.load(os.path.join(raw_dir, "target_batch_10.pt"))
times = torch.load(os.path.join(raw_dir, "time_batch_10.pt"))

print("pred shape:", pred.shape)
print("target shape:", target.shape)
print("times shape:", times.shape)

# choose one rollout inside the batch
sample_idx = 0

# choose how many rollout times to plot
time_slice = slice(None)   # all times
# time_slice = slice(0, 1) # only first time

u_pred = pred[sample_idx, time_slice, :, 0].numpy()      # (time, node)
u_true = target[sample_idx, time_slice, :, 0].numpy()    # (time, node)
time_vals = times[sample_idx].numpy()[time_slice]
# u_true = (
#     ds_train["u"]
#     .sel(time=time_vals, method="nearest")   # match times
#     .isel(ensemble=0)                        # choose ensemble if needed
#     .values
# )

elapsed_time = time_vals - time_vals[0]

# geometry from training dataset
P = ds_train["P"].values
tri = ds_train["tri"].values
R = ds_train.attrs["R"]

# error in model scale
u_err = u_pred - u_true

# shared color scale for fair comparison
vmin = float(np.nanmin([u_pred.min(), u_true.min()]))
vmax = float(np.nanmax([u_pred.max(), u_true.max()]))
norm = colors.Normalize(vmin=vmin, vmax=vmax)

# symmetric color scale for error
err_abs = float(np.nanmax(np.abs(u_err)))
err_norm = colors.Normalize(vmin=-err_abs, vmax=err_abs)

ds_pred = xr.Dataset(
    data_vars={
        "u": (("time", "node"), u_pred),
        "P": (("node", "coord"), P),
        "tri": (("face", "vertex"), tri),
    },
    coords={
        "time": elapsed_time,
        "node": np.arange(P.shape[0]),
    },
    attrs={"R": R},
)

ds_true = xr.Dataset(
    data_vars={
        "u": (("time", "node"), u_true),
        "P": (("node", "coord"), P),
        "tri": (("face", "vertex"), tri),
    },
    coords={
        "time": elapsed_time,
        "node": np.arange(P.shape[0]),
    },
    attrs={"R": R},
)
ds_err = xr.Dataset(
    data_vars={
        "u": (("time", "node"), u_err),
        "P": (("node", "coord"), P),
        "tri": (("face", "vertex"), tri),
    },
    coords={
        "time": elapsed_time,
        "node": np.arange(P.shape[0]),
    },
    attrs={"R": R},
)

plotter_pred = DataPlotter.DataPlotter(ds=ds_pred)
plotter_true = DataPlotter.DataPlotter(ds=ds_true)
plotter_err = DataPlotter.DataPlotter(ds=ds_err)

anim_pred = plotter_pred.animate_sphere(
    norm=norm,
    out_path="eval_pred_batch10_sample0.gif",
    fps=10,
)

anim_true = plotter_true.animate_sphere(
    norm=norm,
    out_path="eval_true_batch10_sample0.gif",
    fps=10,
)

anim_err = plotter_err.animate_sphere(
    norm=err_norm,
    out_path="eval_error_batch10_sample0.gif",
    fps=10,
)
plt.show()