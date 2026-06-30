import xarray as xr
import numpy as np

ds = xr.open_dataset(
    "GNN_training/one_wave/nc_files/wave_200_dtsmall_min10_g4_sigamin6_sigmax12_correctT_plus200dt_plus2dt_test.nc"
)

# ds2 = xr.open_dataset(
#     "GNN_training/one_wave/nc_files/wave_200_40dt_g4_sigamin6_sigmax12_correctTmax.nc"
# )

# ds3 = xr.open_dataset(
#     "GNN_training/one_wave/nc_files/wave_200_80dt_g4_sigamin6_sigmax12_correctTmax.nc"
# )

# ds4 = xr.open_dataset(
#     "GNN_training/one_wave/nc_files/wave_200_160dt_g4_sigamin6_sigmax12_correctTmax.nc"
# )


print("Final time:")
print(ds["time"].values[-1])

print("\nNumber of timesteps:")
print(len(ds["time"].values))

print("\nu shape:")
print(ds["u"].shape)

print("\ndx:")
print(ds.attrs["dx"])

print("\ndt:")
print(ds.attrs["dt"])

# Dataset split sizes
print("\nDataset sizes:")
print("test size:", 601 * 4)
print("val size:", 601 * 4)
print("train size:", 601 * 20)
print("total:", 601 * (4 + 4 + 20))


sigma_string = "sigma"

vals = ds[sigma_string].values

print(f"\n{sigma_string}")
print("shape:", vals.shape)
print("dims:", ds[sigma_string].dims)

print("min sigma:", np.nanmin(vals))
print("max sigma:", np.nanmax(vals))

unique_vals = np.unique(vals)

print("number unique:", len(unique_vals))
print("first unique values:", unique_vals[:10])

