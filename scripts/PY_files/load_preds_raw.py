import os
import torch
import xarray as xr
import numpy as np
import sys
sys.path.insert(0, "./")
import matplotlib.pyplot as plt

from data_generation_functions import DataPlotter
from matplotlib import colors

# training geometry
ds_train = xr.open_dataset(r"./data/nc_files/wave_ensemble_100_coarse.nc")

# load one saved batch
raw_dir = r"./test_eval_same_mesh_10/raw_preds"

pred = torch.load(os.path.join(raw_dir, "pred_batch_0.pt"))
target = torch.load(os.path.join(raw_dir, "target_batch_0.pt"))
times = torch.load(os.path.join(raw_dir, "time_batch_0.pt"))

# shapes to inspect
print("pred shape:", pred.shape)
print("target shape:", target.shape)
print("times shape:", times.shape)

# choose one rollout inside the batch
sample_idx = 0

u_plot = pred[sample_idx, :, :, 0].numpy()   # (time, node)
time_vals = times[sample_idx].numpy()

elapsed_sec = (time_vals - time_vals[0]) 

# geometry from training dataset
P = ds_train["P"].values
tri = ds_train["tri"].values
R = ds_train.attrs["R"]

ds_plot = xr.Dataset(
    data_vars={
        "u": (("time", "node"), u_plot),
        "P": (("node", "coord"), P),
        "tri": (("face", "vertex"), tri),
    },
    coords={
        "time": elapsed_sec,
        "node": np.arange(P.shape[0]),
    },
    attrs={
        "R": R,
    }
)

u_min = float(np.nanmin(ds_plot["u"].values))
u_max = float(np.nanmax(ds_plot["u"].values))
norm = colors.Normalize(vmin=u_min, vmax=u_max)

plotter = DataPlotter.DataPlotter(ds=ds_plot)
anim = plotter.animate_sphere(
    norm=norm,
    out_path="eval_coarse_data_same_mesh_10.gif",
    fps=10,
)
plt.show()