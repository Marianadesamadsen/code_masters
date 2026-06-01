import sys

from matplotlib import colors 
sys.path.insert(0, "./")
import scripts.PY_files.eval_models_scripts.plot_results_with_zarr as plot_results
import scripts.PY_files.eval_models_scripts.plot_animations as plot_animations
import os
import xarray as xr
from data_generation_functions import DataPlotterAll

ds_geo_dir = "GNN_training/one_wave/nc_files/wave_200_ts_100_Tmax6_g5.nc"

plot_dir = "GNN_training/one_wave/new_dt/newsetup_mp3_results"
anim_dir = "GNN_training/one_wave/new_dt/newsetup_mp3_results"
  
raw_dir_10 = "GNN_training/one_wave/new_dt/test_new_setup_mp3.zarr"

plot_results.plot_results(ds_geo_dir, raw_dir_10, plot_dir, anim_dir, generations=5, plot_animations_1step=False, plot_animations_rollout=True, azim=135, elev=30)






