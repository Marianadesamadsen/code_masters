import xarray as xr
ds = xr.open_zarr("data/coarse_data_100_waves.zarr")

print(ds["state__train__mean"].values)
print(ds["state__train__std"].values)
print(ds["state__train__diff_mean"].values)
print(ds["state__train__diff_std"].values)
print(ds["ensemble_member"].values)

