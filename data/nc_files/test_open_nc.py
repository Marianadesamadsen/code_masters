import xarray as xr

ds = xr.open_dataset("data/nc_files/wave_ensemble_20_coarse_500_timesteps_sub4_wp1.nc")

print(ds["time"].values[-1])
print(len(ds["time"].values))
print(ds["u"].shape)


