from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


BASE_DIR = Path("GNN_training/one_wave/different_mesh_size/final_results")
RESULTS_DIR = Path("GNN_training/one_wave/different_mesh_size/final_results_plots/AR")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NC_FILE = Path(
    "GNN_training/one_wave/nc_files/"
    "wave_200_dtsmall_min10_g4_sigamin6_sigmax12_correctT_plus160dt.nc"
)

TEST_TIME_STRIDE = 10
TEST_MEMBER_START = 50
TEST_MEMBER_END = 100
FEATURE_IDX = 0

DT_COLORS = {
    10: "tab:blue",
    20: "tab:orange",
    40: "tab:green",
}

RUN_DIRS = {
    "10dt AR2": {"label": r"$10\Delta t$ AR2", "result_dir": "test_10dt_AR_2", "dt_scale": 10},
    "20dt AR2": {"label": r"$20\Delta t$ AR2", "result_dir": "test_20dt_AR_2", "dt_scale": 20},
    "40dt AR2": {"label": r"$40\Delta t$ AR2", "result_dir": "test_40dt_AR_2", "dt_scale": 40},
    "10dt": {"label": r"$10\Delta t$", "result_dir": "test_10dt_2", "dt_scale": 10},
    "20dt": {"label": r"$20\Delta t$", "result_dir": "test_20dt_2", "dt_scale": 20},
    "40dt": {"label": r"$40\Delta t$", "result_dir": "test_40dt_2", "dt_scale": 40},
}


def get_rollout_cols(df):
    cols = [c for c in df.columns if c.startswith("rollout_")]
    cols = sorted(cols, key=lambda c: int(c.split("_")[-1]))
    rollouts = np.array([int(c.split("_")[-1]) for c in cols])
    return cols, rollouts


def load_metric(filename):
    data = {}

    for dt_key, cfg in RUN_DIRS.items():
        csv_path = BASE_DIR / cfg["result_dir"] / filename

        if not csv_path.exists():
            raise FileNotFoundError(csv_path)

        data[dt_key] = pd.read_csv(csv_path)

    return data


def load_true_u():
    ds = xr.open_dataset(NC_FILE)

    if "u" not in ds:
        raise KeyError("Variable 'u' was not found in the nc file.")

    u = ds["u"].transpose("ensemble_member", "time", "grid_index")
    print(ds["u"].shape)
    print(ds["time"].values[:5])
    print(ds["time"].values[80])
    print("min",ds["u"].max().item())
    print("max",ds["u"].min().item())

    return u


def compute_persistence_rmse_curve_from_nc(u, rollouts, dt_scale):

    max_horizon = int(np.max(rollouts) * dt_scale)
    #u["time"] = u["time"][:405]
    n_time = u.sizes["time"]

    start_indices = np.arange(
        0,
        n_time - max_horizon,
        TEST_TIME_STRIDE,
        dtype=int,
    )

    member_indices = np.arange(TEST_MEMBER_START, TEST_MEMBER_END)

    y_persistence = []

    for rollout in rollouts:
        horizon = int(rollout * dt_scale)

        rmse_values = []

        for member in member_indices:
            u_member = u.isel(ensemble_member=member).values
            u0 = u.sel(ensemble_member=member).isel(time=0).values
            uh = u.sel(ensemble_member=member).isel(time=horizon).values
            diff = uh - u0

            # print("max abs diff:", np.max(np.abs(diff)))
            # print("mean abs diff:", np.mean(np.abs(diff)))
            # print("RMSE:", np.sqrt(np.mean(diff**2)))

            # print("fraction |diff| > 0.1:", np.mean(np.abs(diff) > 0.1))
            # print("fraction |diff| > 0.5:", np.mean(np.abs(diff) > 0.5))
            # u0 = u_member[start_indices, :]
            # uh = u_member[start_indices + horizon, :]

            rmse_per_start = np.sqrt(np.mean((uh - u0) ** 2, axis=0))
            rmse_values.append(rmse_per_start)

        #rmse_values = np.concatenate(rmse_values)
        y_persistence.append(np.mean(rmse_values))

    return np.array(y_persistence)


def plot_rollout_rmse_energy():
    data_rmse = load_metric("test_rmse_per_sample.csv")
    data_energy = load_metric("test_energy_rel_error_per_sample.csv")
    u = load_true_u()

    member = 50
    start = 0
    horizon = 80

    u0 = u.sel(ensemble_member=member).isel(time=start).values
    uh = u.sel(ensemble_member=member).isel(time=start+horizon).values

    rmse = np.sqrt(np.mean((uh-u0)**2))
    print(rmse)

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    for dt_key, cfg in RUN_DIRS.items():
        dt_scale = cfg["dt_scale"]
        label = cfg["label"]

        color = DT_COLORS[dt_scale]
        linestyle = "--" if "AR2" in dt_key else "-"

        rmse_df = data_rmse[dt_key]
        rmse_cols, rmse_rollouts = get_rollout_cols(rmse_df)

        x_rmse = rmse_rollouts * dt_scale
        y_rmse = rmse_df[rmse_cols].mean(axis=0).values

        axes[0].loglog(
            x_rmse,
            y_rmse,
            marker="o",
            linestyle=linestyle,
            color=color,
            label=label,
        )

        energy_df = data_energy[dt_key]
        energy_cols, energy_rollouts = get_rollout_cols(energy_df)

        x_energy = energy_rollouts * dt_scale
        y_energy = energy_df[energy_cols].mean(axis=0).values

        axes[1].loglog(
            x_energy,
            y_energy,
            marker="o",
            linestyle=linestyle,
            color=color,
            label=label,
        )

        if dt_scale == 90: #or dt_scale == 20:
            y_persistence = compute_persistence_rmse_curve_from_nc(
                u=u,
                rollouts=rmse_rollouts,
                dt_scale=dt_scale,
            )

            axes[0].loglog(
                x_rmse,
                y_persistence,
                marker="s",
                linestyle=":",
                linewidth=2.0,
                label=rf"Persistence {label}",
            )


    axes[0].set_ylabel("RMSE", fontsize=20)
    axes[0].legend(fontsize=20)
    axes[0].grid(True, which="both", alpha=0.4)

    axes[1].set_ylabel("Relative energy error", fontsize=20)
    axes[1].set_xlabel("Rollout horizon", fontsize=20)
    axes[1].legend(fontsize=20)
    axes[1].grid(True, which="both", alpha=0.4)

    fig.suptitle(
        "RMSE and relative energy error over matched physical rollout time",
        fontsize=22,
    )

    fig.tight_layout()

    out_path = RESULTS_DIR / "rmse_energy_with_persistence_from_nc.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    plot_rollout_rmse_energy()
