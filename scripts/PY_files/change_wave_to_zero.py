from pathlib import Path
import shutil
import xarray as xr
import numpy as np

ZERO_ZARR = Path("GNN_training/one_wave/yaml_files/wave_test_zero_data.zarr")
TMP_ZARR = Path("GNN_training/one_wave/yaml_files/wave_test_zero_data_tmp.zarr")

ZERO_MEMBER = 50

# Open and fully load into memory
ds = xr.open_zarr(ZERO_ZARR, consolidated=False).load()

# Close file handles before writing
ds.close()

# Change only ensemble member 50
ds["state"].loc[dict(ensemble_member=ZERO_MEMBER)] = 0.0

print("before save nan:", np.isnan(ds["state"].values).any())
print("member 50 min/max:",
      float(ds["state"].sel(ensemble_member=ZERO_MEMBER).min()),
      float(ds["state"].sel(ensemble_member=ZERO_MEMBER).max()))

# Write to temporary zarr first
shutil.rmtree(TMP_ZARR, ignore_errors=True)
ds.to_zarr(TMP_ZARR, mode="w")

# Replace original only after successful write
shutil.rmtree(ZERO_ZARR, ignore_errors=True)
shutil.move(str(TMP_ZARR), str(ZERO_ZARR))

# Check
check = xr.open_zarr(ZERO_ZARR)
u = check["state"].values

print("after save nan:", np.isnan(u).any())
print("after save min/max:", np.nanmin(u), np.nanmax(u))
print("member 50 min/max:",
      float(check["state"].sel(ensemble_member=ZERO_MEMBER).min()),
      float(check["state"].sel(ensemble_member=ZERO_MEMBER).max()))