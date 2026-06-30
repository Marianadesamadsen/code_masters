import numpy as np
import xarray as xr
import zarr

src = zarr.open("GNN_training/one_wave/yaml_files/wave_full_data_grid4.zarr",mode="r")
dst = zarr.open("GNN_training/one_wave/yaml_files/yamlfiles_for_trainsize/wave_125_train.zarr",mode="r+")

# for key in [
#     "state__train__mean",
#     "state__train__std",
#     "state__train__diff_mean",
#     "state__train__diff_std",
# ]:
#     dst[key][:] = src[key][:]#dst[var].values[:] = src[var].values

print(np.allclose(src["state__train__mean"], dst["state__train__mean"]))
print(np.allclose(src["state__train__std"], dst["state__train__std"]))