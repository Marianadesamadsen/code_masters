import xarray as xr
import numpy as np

paths = [
    #"GNN_training/one_wave/yaml_files/wave_1_train.zarr",
    #"GNN_training/one_wave/yaml_files/wave_10_train.zarr",
    #"GNN_training/one_wave/yaml_files/wave_25_train.zarr",
    "GNN_training/one_wave/yaml_files/wave_full_data_grid4.zarr",
    #"GNN_training/one_wave/yaml_files/wave_75_train.zarr",
   #"GNN_training/one_wave/yaml_files/wave_100_train.zarr",
]

for p in paths:

    ds = xr.open_zarr(p)

    print("sizes:", ds.sizes)
    print("state dims:", ds["state"].dims)

    ensemble_members = ds["ensemble_member"].values

    for split_name in ds["split_name"].values:

        split_vals = ds["splits"].sel(split_name=split_name).values

        split_start = int(split_vals[0])
        split_end = int(split_vals[1])

        members_in_split = ds["ensemble_member"].sel(
            ensemble_member=slice(split_start, split_end)
        ).values

        print(f"\n{split_name}:")
        print(f"  start: {split_start}")
        print(f"  end:   {split_end}")
        print(f"  count: {len(members_in_split)}")
        print(f"  ensemble members: {members_in_split}")


    # Train split
    splits = ds["splits"].sel(split_name="train").values
    train_start = int(splits[0])
    train_end = int(splits[1])

    print("\nTrain split used for stats:")
    print("train split:", train_start, "to", train_end)

    u_train = ds["state"].sel(
        ensemble_member=slice(train_start, train_end)
    ).values

    print("u_train shape:", u_train.shape)
    print("has nan:", np.isnan(u_train).any())
    print("has inf:", np.isinf(u_train).any())


    # Manual statistics
    manual_mean = np.nanmean(u_train, axis=(1, 2, 3))
    manual_std = np.nanstd(u_train, axis=(1, 2, 3))
    diff_once = np.diff(u_train, axis=2)
    manual_diff_mean = np.nanmean(diff_once, axis=(1, 2, 3))
    manual_diff_std  = np.nanstd(diff_once, axis=(1, 2, 3))

    # Saved statistics
    saved_mean = ds["state__train__mean"].values
    saved_std = ds["state__train__std"].values
    saved_diff_mean = ds["state__train__diff_mean"].values
    saved_diff_std = ds["state__train__diff_std"].values

    # Compare
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