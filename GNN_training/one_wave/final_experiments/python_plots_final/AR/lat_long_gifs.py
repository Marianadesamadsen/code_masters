from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.interpolate import griddata


# Paths
ds_geo_dir = "GNN_training/one_wave/nc_files/wave_200_dtsmall_min10_g4_sigamin6_sigmax12_correctT_plus160dt.nc"
raw_dir = "GNN_training/one_wave/different_mesh_size/final_results/test_80dt.zarr"

anim_dir = Path("GNN_training/one_wave/different_mesh_size/final_results_plots/latlon_gif")
anim_dir.mkdir(parents=True, exist_ok=True)

# Settings
sample_idx = 7          # choose which test sample to animate
state_feature = 0
fps = 4
n_lon = 360
n_lat = 180

output_path = anim_dir / f"latlon_rollout_sample_{sample_idx}_80dt.gif"


# Load data
ds_geo = xr.open_dataset(ds_geo_dir)
ds = xr.open_zarr(raw_dir)

print(ds)

pred = ds["prediction"].isel(sample=sample_idx, state_feature=state_feature).values
target = ds["target"].isel(sample=sample_idx, state_feature=state_feature).values

# shape should be: rollout_step, grid_index
pred = np.squeeze(pred)
target = np.squeeze(target)

lat = ds_geo["lat"].values
lon = ds_geo["lon"].values

# Convert lon to [-180, 180] if needed
lon_plot = ((lon + 180) % 360) - 180

# Regular lat-lon grid
lon_grid = np.linspace(-180, 180, n_lon)
lat_grid = np.linspace(-90, 90, n_lat)
LON, LAT = np.meshgrid(lon_grid, lat_grid)

points = np.column_stack([lon_plot, lat])


pred_grids = []
target_grids = []
error_grids = []

for t in range(pred.shape[0]):
    pred_i = griddata(points, pred[t], (LON, LAT), method="linear")
    target_i = griddata(points, target[t], (LON, LAT), method="linear")

    pred_nearest = griddata(points, pred[t], (LON, LAT), method="nearest")
    target_nearest = griddata(points, target[t], (LON, LAT), method="nearest")

    pred_i = np.where(np.isnan(pred_i), pred_nearest, pred_i)
    target_i = np.where(np.isnan(target_i), target_nearest, target_i)

    pred_grids.append(pred_i)
    target_grids.append(target_i)
    error_grids.append(pred_i - target_i)

pred_grids = np.array(pred_grids)
target_grids = np.array(target_grids)
error_grids = np.array(error_grids)


# Shared color limits
vmin = min(pred_grids.min(), target_grids.min())
vmax = max(pred_grids.max(), target_grids.max())

err_abs = np.nanmax(np.abs(error_grids))
err_vmin, err_vmax = -err_abs, err_abs


# Make animation
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, sharey=True)

titles = ["Target", "Prediction", "Prediction - Target"]

im0 = axes[0].imshow(
    target_grids[0],
    extent=[-180, 180, -90, 90],
    origin="lower",
    aspect="auto",
    vmin=vmin,
    vmax=vmax,
)

im1 = axes[1].imshow(
    pred_grids[0],
    extent=[-180, 180, -90, 90],
    origin="lower",
    aspect="auto",
    vmin=vmin,
    vmax=vmax,
)

im2 = axes[2].imshow(
    error_grids[0],
    extent=[-180, 180, -90, 90],
    origin="lower",
    aspect="auto",
    vmin=err_vmin,
    vmax=err_vmax,
    cmap="RdBu_r",
)

for ax, title in zip(axes, titles):
    ax.set_title(title, fontsize=14)
    ax.set_ylabel("Latitude")

axes[-1].set_xlabel("Longitude")

cbar0 = fig.colorbar(im0, ax=axes[:2], fraction=0.025, pad=0.02)
cbar0.set_label("u")

cbar1 = fig.colorbar(im2, ax=axes[2], fraction=0.025, pad=0.02)
cbar1.set_label("error")

suptitle = fig.suptitle("", fontsize=16)


def update(frame):
    im0.set_data(target_grids[frame])
    im1.set_data(pred_grids[frame])
    im2.set_data(error_grids[frame])

    rollout_step = int(ds["rollout_step"].values[frame])
    suptitle.set_text(
        f"Lat-lon rollout, sample {sample_idx}, rollout step {rollout_step}"
    )

    return im0, im1, im2, suptitle


anim = FuncAnimation(
    fig,
    update,
    frames=pred_grids.shape[0],
    interval=1000 / fps,
    blit=False,
)

anim.save(output_path, writer=PillowWriter(fps=fps))
plt.close(fig)

print(f"Saved GIF to: {output_path}")