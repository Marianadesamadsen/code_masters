import xarray as xr
import numpy as np
import sys
sys.path.insert(0, "./")
import matplotlib.pyplot as plt

from data_generation_functions import DataPlotter
from matplotlib import colors

ds_train = xr.open_dataset(r"./data/nc_files/wave_ensemble_100_coarse.nc")
ds_pred = xr.open_zarr(r"./test_eval_same_mesh.zarr")

u = ds_pred["state"].sel(state_feature="u") 

u_plot = u.isel(start_time=0) 
 
# geometry from training dataset
P = ds_train["P"].values
tri = ds_train["tri"].values
R = ds_train.attrs["R"]

n_time = u.sizes["elapsed_forecast_duration"]
time_sec = u["elapsed_forecast_duration"]

ds_plot = xr.Dataset(
    data_vars={
        "u": (("time", "node"), u_plot.values),
        "P": (("node", "coord"), P),
        "tri": (("face", "vertex"), tri),
    },
    coords={
        "time": time_sec,
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
anim = plotter.animate_sphere(norm=norm, out_path="eval_coarse_data_same_mesh_10.gif", fps=10)
plt.show()



