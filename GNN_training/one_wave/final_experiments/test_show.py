import sys

from matplotlib import colors 
sys.path.insert(0, "./")
import scripts.PY_files.eval_models_scripts.plot_results_with_zarr as plot_results
import scripts.PY_files.eval_models_scripts.plot_animations as plot_animations
import os
import xarray as xr
from data_generation_functions import DataPlotterAll

ds_geo_dir = "GNN_training/one_wave/nc_files/wave_200_dtsmall_min10_g4_sigamin6_sigmax12_correctT_plus160dt.nc"

plot_dir = "GNN_training/one_wave/different_mesh_size/correct_results/dt40_mp2_min1roll"
anim_dir = "GNN_training/one_wave/different_mesh_size/correct_results/dt40_mp2_min1roll"
  
raw_dir_10 = "GNN_training/one_wave/different_mesh_size/correct_results/test_dt40_minroll1_mp2.zarr"

plot_results.plot_results(ds_geo_dir, raw_dir_10, plot_dir, anim_dir, generations=4, plot_animations_1step=False, plot_animations_rollout=True, azim=135, elev=30)


 

