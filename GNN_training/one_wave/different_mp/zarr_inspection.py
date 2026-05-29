from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

BASE_DIR = Path("GNN_training/one_wave/different_mp")

MP_RUNS = {
    1: "test_mp1_results_new",
    2: "test_mp2_results_new",
    3: "test_mp3_results_new",
}

for mp_step, folder in MP_RUNS.items():
    print("\n" + "=" * 80)
    print(f"MP {mp_step}")
    print("=" * 80)

    zarr_path = BASE_DIR / f"test_mp{mp_step}_results_new.zarr"
    meta_path = BASE_DIR / folder / "test_metadata.csv"

    ds = xr.open_zarr(zarr_path)
    meta = pd.read_csv(meta_path)

    print("\nZarr path:")
    print(zarr_path)

    print("\nZarr dataset:")
    print(ds)

    print("\nVariables:")
    print(list(ds.data_vars))

    print("\nDimensions:")
    print(ds.sizes)

    n_zarr_samples = ds.sizes["sample"]

    print("\nNumber of Zarr samples:")
    print(n_zarr_samples)

    print("\nNumber of metadata rows:")
    print(len(meta))

    meta_zarr = meta.iloc[:n_zarr_samples].reset_index(drop=True)

    print("\nFirst 20 metadata rows corresponding to Zarr samples:")
    print(meta_zarr.head(20))

    print("\nUnique ensemble members inside saved Zarr samples:")
    print(np.unique(meta_zarr["ensemble_member"].values))

    print("\nNumber of unique ensemble members inside saved Zarr samples:")
    print(meta_zarr["ensemble_member"].nunique())

    print("\nCounts per ensemble member inside saved Zarr samples:")
    print(meta_zarr["ensemble_member"].value_counts().sort_index())

    print("\nSample index range in metadata subset:")
    print(meta_zarr["sample_idx"].min(), "to", meta_zarr["sample_idx"].max())

    if "valid_time" in ds:
        print("\nvalid_time:")
        print(ds["valid_time"])

    if "prediction" in ds:
        print("\nprediction dims:")
        print(ds["prediction"].dims)
        print(ds["prediction"].shape)

    if "target" in ds:
        print("\ntarget dims:")
        print(ds["target"].dims)
        print(ds["target"].shape)