import xarray as xr
import numpy as np

paths = [
    "GNN_training/one_wave/yaml_files/wave_25_train.zarr",
    "GNN_training/one_wave/yaml_files/wave_75_train.zarr",
    "GNN_training/one_wave/yaml_files/wave_100_train.zarr",
]

for p in paths:
    print("\n========================")
    print(p)

    ds = xr.open_zarr(p)

    print("sizes:", ds.sizes)
    print("state dims:", ds["state"].dims)

    splits = ds["splits"].sel(split_name="train").values
    train_start = int(splits[0])
    train_end = int(splits[1])

    print("train split:", train_start, "to", train_end)

    u_train = ds["state"].sel(
        ensemble_member=slice(train_start, train_end)
    ).values

    print("u_train shape:", u_train.shape)
    print("has nan:", np.isnan(u_train).any())
    print("has inf:", np.isinf(u_train).any())

    manual_mean = np.nanmean(u_train, axis=(1, 2, 3))
    manual_std = np.nanstd(u_train, axis=(1, 2, 3))

    # Mimic current calc_stats mutation:
    # diff_mean uses first temporal difference
    diff_once = np.diff(u_train, axis=2)
    manual_diff_mean = np.nanmean(diff_once, axis=(1, 2, 3))

    # diff_std uses second temporal difference because ds was mutated after diff_mean
    diff_twice = np.diff(diff_once, axis=2)
    manual_diff_std = np.nanstd(diff_twice, axis=(1, 2, 3))

    saved_mean = ds["state__train__mean"].values
    saved_std = ds["state__train__std"].values
    saved_diff_mean = ds["state__train__diff_mean"].values
    saved_diff_std = ds["state__train__diff_std"].values

    print("\nManual vs saved")
    print("mean:")
    print("  manual:", manual_mean)
    print("  saved: ", saved_mean)
    print("  close: ", np.allclose(manual_mean, saved_mean, equal_nan=True))

    print("std:")
    print("  manual:", manual_std)
    print("  saved: ", saved_std)
    print("  close: ", np.allclose(manual_std, saved_std, equal_nan=True))

    print("diff_mean:")
    print("  manual:", manual_diff_mean)
    print("  saved: ", saved_diff_mean)
    print("  close: ", np.allclose(manual_diff_mean, saved_diff_mean, equal_nan=True))

    print("diff_std:")
    print("  manual:", manual_diff_std)
    print("  saved: ", saved_diff_std)
    print("  close: ", np.allclose(manual_diff_std, saved_diff_std, equal_nan=True))