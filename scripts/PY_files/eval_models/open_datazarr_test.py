import xarray as xr

ds = xr.open_dataset(r"./data/coarse_data.zarr")
print(ds["time"])

