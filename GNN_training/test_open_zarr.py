
import xarray as xr

for p in [
    "GNN_training/one_wave/yaml_files/wave_10_train.zarr","GNN_training/one_wave/yaml_files/wave_25_train.zarr","GNN_training/one_wave/yaml_files/wave_50_train.zarr",
    "GNN_training/one_wave/yaml_files/wave_75_train.zarr",
    "GNN_training/one_wave/yaml_files/wave_100_train.zarr"
]:
    print("\n", p)
    ds = xr.open_zarr(p)
    print(ds.sizes)
