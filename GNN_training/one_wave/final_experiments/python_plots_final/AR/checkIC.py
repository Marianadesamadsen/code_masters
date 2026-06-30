from pathlib import Path

import numpy as np
import xarray as xr


# Paths
BASE_DIR = Path("GNN_training/one_wave/different_mesh_size/final_results")

NC_PATH = (
    "GNN_training/one_wave/nc_files/"
    "wave_200_dtsmall_min10_g4_sigamin6_sigmax12_correctT_plus160dt.nc"
)

ZARR_PATH = BASE_DIR / "test_1dt.zarr"


# Matching function
def find_matching_nc_member(ds_nc, ds_zarr, sample_idx, feature_idx=0):

    target0 = ds_zarr["target"].isel(
        sample=sample_idx,
        rollout_step=0,
        state_feature=feature_idx,
    ).values

    valid_time = float(
        ds_zarr["valid_time"].isel(
            sample=sample_idx,
            rollout_step=0,
        ).values
    )

    nc_times = ds_nc["time"].values

    time_idx = int(np.argmin(np.abs(nc_times - valid_time)))

    nc_fields = ds_nc["u"].isel(time=time_idx).values
    # shape = (ensemble_member, grid_index)

    mse = np.mean((nc_fields - target0[None, :]) ** 2, axis=1)

    ensemble_idx = int(np.argmin(mse))
    rmse = float(np.sqrt(mse[ensemble_idx]))

    return ensemble_idx, time_idx, rmse


# Main
def main():

    ds_nc = xr.open_dataset(NC_PATH)
    ds_zarr = xr.open_zarr(ZARR_PATH)

    print()
    print("=" * 120)
    print(
        f"{'sample':>6} {'member':>8} {'A':>8} {'sigma_deg':>10} {'sigma_rad':>10} {'RMSE':>12}"
    )
    print("=" * 120)

    for sample_idx in range(20):

        ensemble_idx, time_idx, rmse = find_matching_nc_member(
            ds_nc,
            ds_zarr,
            sample_idx,
        )

        A = float(
            ds_nc["A"]
            .isel(ensemble_member=ensemble_idx)
            .values
        )

        sigma = float(
            ds_nc["sigma"]
            .isel(ensemble_member=ensemble_idx)
            .values
        )

        sigma_deg = float(
            ds_nc["sigma_deg"]
            .isel(ensemble_member=ensemble_idx)
            .values
        )

        center = (
            ds_nc["center"]
            .isel(ensemble_member=ensemble_idx)
            .values
        )

        print(
            f"{sample_idx:6d} "
            f"{ensemble_idx:8d} "
            f"{A:8.4f} "
            f"{sigma_deg:10.2f} "
            f"{sigma:10.4f} "
            f"{rmse:12.3e}"
        )

        print(
            f"         center = "
            f"[{center[0]: .4f}, {center[1]: .4f}, {center[2]: .4f}]"
        )
        print()

    ds_zarr.close()
    ds_nc.close()


if __name__ == "__main__":
    main()