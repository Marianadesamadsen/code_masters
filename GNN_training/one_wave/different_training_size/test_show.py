import sys

from matplotlib import colors 
sys.path.insert(0, "./")
import scripts.PY_files.eval_models_scripts.plot_results_with_zarr as plot_results
import scripts.PY_files.eval_models_scripts.plot_animations as plot_animations
import os
import xarray as xr
from data_generation_functions import DataPlotterAll

ds_geo_dir = "GNN_training/one_wave/nc_files/wave_200_ts_600_g4_sigmamin_15.nc"

plot_dir = "GNN_training/one_wave/different_training_size/all_results_plot/other_results"
anim_dir = "GNN_training/one_wave/different_training_size/all_results_plot/animations"
  
#raw_dir_10 = "GNN_training/one_wave/different_training_size/test_10_results.zarr"

#plot_results.plot_results(ds_geo_dir, raw_dir_10, plot_dir, anim_dir, generations=4, plot_animations_1step=False, plot_animations_rollout=False, azim=135, elev=30)
plot_animations.plot_results(
    ds_geo_dir=ds_geo_dir,
    raw_dirs=[
        "GNN_training/one_wave/different_training_size/test_5_results_new.zarr",
        "GNN_training/one_wave/different_training_size/test_25_results_new.zarr",
        "GNN_training/one_wave/different_training_size/test_50_results_new.zarr",
        "GNN_training/one_wave/different_training_size/test_100_results_new.zarr",
    ],
    plot_dir=plot_dir,
    anim_dir=anim_dir,
    generations=4,
    training_size_labels=[
        "Train size: 1 wave",
        "Train size: 5 waves",
        "Train size: 25 waves",
        "Train size: 50 waves",
    ],
    rolloutidx=150,
    azim=135,
    elev=30,
    plot_animations_rollout=True
)


