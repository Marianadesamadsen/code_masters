import xarray as xr

ds = xr.open_dataset("GNN_training/one_wave/nc_files/wave_28_ts_600_g4_sigmamin_15.nc")

print(ds["time"].values[-1])
print(len(ds["time"].values))
print(ds["u"].shape)
print(ds.attrs["dx"])
print(ds.attrs["dt"])

print("test size:", 601*4)
print("val size:", 601*4)
print("train size:",601*20)
print("total:",)