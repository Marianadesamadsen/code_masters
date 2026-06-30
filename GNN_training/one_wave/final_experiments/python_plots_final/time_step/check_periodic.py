from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

NC_PATH = Path(
    "GNN_training/one_wave/nc_files/"
    "wave_200_dtsmall_min10_g4_sigamin6_sigmax12_correctT_plus160dt.nc"
)

PLOT_DIR = Path("GNN_training/one_wave/different_mesh_size/final_results_plots/time_step/final")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

ensemble_idx = 60
n_bins = 40
save_name = f"analytical_radial_profiles_t0_pi_2pi_member{ensemble_idx}.png"


def load_nc_geometry(ds_nc):
    lat = ds_nc["lat"].values
    lon = ds_nc["lon"].values

    if np.nanmax(np.abs(lat)) > np.pi + 1e-6:
        lat = np.deg2rad(lat)

    if np.nanmax(np.abs(lon)) > 2 * np.pi + 1e-6:
        lon = np.deg2rad(lon)

    xyz = np.column_stack(
        [
            np.cos(lat) * np.cos(lon),
            np.cos(lat) * np.sin(lon),
            np.sin(lat),
        ]
    )

    xyz = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)
    return xyz


def get_initial_center_from_nc(ds_nc, ensemble_idx, xyz):
    if "center" in ds_nc:
        center = ds_nc["center"].isel(ensemble_member=ensemble_idx).values
        center = center.astype(float)
        center = center / np.linalg.norm(center)
        print("Using center variable from NC file.")
    else:
        u0 = ds_nc["u"].isel(
            ensemble_member=ensemble_idx,
            time=0,
        ).values

        center_idx = int(np.argmax(np.abs(u0)))
        center = xyz[center_idx]
        center = center / np.linalg.norm(center)
        print("Using argmax of u(t=0) from NC file.")

    r_nodes = np.arccos(np.clip(xyz @ (center), -1.0, 1.0))
    return r_nodes


def make_bin_indices(r_nodes, n_bins):
    bins = np.linspace(0.0, np.pi, n_bins + 1)
    bin_idx = np.digitize(r_nodes, bins) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    return bin_idx


def get_profile(values, r_nodes, bin_idx, n_bins):
    r_bin = []
    u_bin = []

    for b in range(n_bins):
        inside = bin_idx == b

        if np.any(inside):
            r_bin.append(np.mean(r_nodes[inside]))
            u_bin.append(np.mean(values[inside]))

    return np.array(r_bin), np.array(u_bin)


def get_closest_time_index(ds_nc, target_time):
    times = ds_nc["time"].values
    time_idx = int(np.argmin(np.abs(times - target_time)))
    actual_time = float(times[time_idx])

    print(
        f"Target time {target_time:.6f} -> "
        f"closest index {time_idx}, actual time {actual_time:.6f}"
    )

    return time_idx, actual_time


def main():
    ds_nc = xr.open_dataset(NC_PATH)

    xyz = load_nc_geometry(ds_nc)

    r_nodes = get_initial_center_from_nc(
        ds_nc=ds_nc,
        ensemble_idx=ensemble_idx,
        xyz=xyz,
    )

    bin_idx = make_bin_indices(r_nodes, n_bins)

    target_times = [0.0, np.pi, 2 * np.pi]
    labels = [r"$t=0$", r"$t=\pi$", r"$t=2\pi$"]

    fig, ax = plt.subplots(figsize=(9, 6))

    for target_time, label in zip(target_times, labels):
        time_idx, actual_time = get_closest_time_index(ds_nc, target_time)

        values = ds_nc["u"].isel(
            ensemble_member=ensemble_idx,
            time=time_idx,
        ).values

        r_plot, profile = get_profile(
            values=values,
            r_nodes=r_nodes,
            bin_idx=bin_idx,
            n_bins=n_bins,
        )

        ax.plot(
            r_plot,
            profile,
            marker=".",
            markersize=4,
            linewidth=1.8,
            label=label + rf" $(t={actual_time:.3f})$",
        )

    ax.set_xlabel("Angular distance from initial center [rad]", fontsize=14)
    ax.set_ylabel(r"$u$", fontsize=16)
    ax.set_xlim(0.0, np.pi)
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=12)

    fig.tight_layout()

    out_path = PLOT_DIR / save_name
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    ds_nc.close()

    print()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()