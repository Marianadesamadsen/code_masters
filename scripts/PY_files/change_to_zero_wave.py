import xarray as xr

in_path = "GNN_training/one_wave/nc_files/wave_200_dtsmall_min10_g4_sigamin6_sigmax12_correctT_plus160dt_plus2dt_testdata.nc"
out_path = "GNN_training/one_wave/nc_files/wave_zero_member50_only.nc"

ds = xr.open_dataset(in_path)

ds50 = ds.sel(ensemble_member=50)

# keep ensemble_member dimension
ds50 = ds50.expand_dims(ensemble_member=[50])

ds50.to_netcdf(out_path)