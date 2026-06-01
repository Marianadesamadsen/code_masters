import xarray as xr
import numpy as np

ds = xr.open_dataset(
    "GNN_training/one_wave/nc_files/wave_200_ts_100_Tmax6_g5.nc"
)

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

# --------------------------------------------------
# Dataset split sizes
# --------------------------------------------------

print("\nDataset sizes:")
print("test size:", 601 * 4)
print("val size:", 601 * 4)
print("train size:", 601 * 20)
print("total:", 601 * (4 + 4 + 20))

# --------------------------------------------------
# Sigma check
# --------------------------------------------------

print("\nSigma check:")

sigma_candidates = [
    name for name in ds.variables
    if "sigma" in name.lower()
]

print("sigma-like variables:", sigma_candidates)

for name in sigma_candidates:

    vals = ds[name].values

    print(f"\n{name}")
    print("shape:", vals.shape)
    print("dims:", ds[name].dims)

    print("min sigma:", np.nanmin(vals))
    print("max sigma:", np.nanmax(vals))

    unique_vals = np.unique(vals)

    print("number unique:", len(unique_vals))
    print("first unique values:", unique_vals[:10])

# --------------------------------------------------
# Print all variables if sigma not found
# --------------------------------------------------

if len(sigma_candidates) == 0:
    print("\nNo sigma variable found.")
    print("\nAvailable variables:")
    print(list(ds.variables))