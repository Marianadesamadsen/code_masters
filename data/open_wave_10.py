import xarray as xr

ds = xr.open_zarr("data/yaml_files/faster_training_tuning/data_chunking_3.zarr")
print(ds)
print(ds["state"].chunks)
