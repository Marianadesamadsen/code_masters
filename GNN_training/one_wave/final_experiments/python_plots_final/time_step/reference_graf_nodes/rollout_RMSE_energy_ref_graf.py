from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


BASE_DIR = Path("GNN_training/one_wave/different_mesh_size/final_results")
RESULTS_DIR = Path("GNN_training/one_wave/different_mesh_size/final_results_plots/time_step/final")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NC_FILE = Path(
    "GNN_training/one_wave/nc_files/"
    "wave_200_dtsmall_min10_g4_sigamin6_sigmax12_correctT_plus160dt.nc"
)

ANALYTICAL_ENERGY_FILE = Path(
    "GNN_training/one_wave/energy/analytical_energy_sem.nc"
)

TEST_TIME_STRIDE = 10
TEST_MEMBER_START = 50
TEST_MEMBER_END = 100

RUN_DIRS = {
    #"1dt": {"label": r"$1\Delta t$", "result_dir": "test_1dt", "dt_scale": 1},
    "10dt": {"label": r"$10\Delta t$", "result_dir": "test_10dt_2", "dt_scale": 10},
    "20dt": {"label": r"$20\Delta t$", "result_dir": "test_20dt_2", "dt_scale": 20},
    "40dt": {"label": r"$40\Delta t$", "result_dir": "test_40dt_2", "dt_scale": 40},
    "80dt": {"label": r"$80\Delta t$", "result_dir": "test_80dt_2", "dt_scale": 80},
    "100dt": {"label": r"$100\Delta t$", "result_dir": "test_100dt_2", "dt_scale": 100},
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


def load_metadata(dt_key):
    cfg = RUN_DIRS[dt_key]
    metadata_path = BASE_DIR / cfg["result_dir"] / "test_metadata.csv"

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing metadata file: {metadata_path}\n"
            "You need this file to align each energy row with ensemble_member and sample_idx."
        )

    return pd.read_csv(metadata_path)


def load_true_u():
    ds = xr.open_dataset(NC_FILE)

    if "u" not in ds:
        raise KeyError("Variable 'u' was not found in the nc file.")

    u = ds["u"].transpose("ensemble_member", "time", "grid_index")
    return u,ds

def load_analytical_energy():
    ds_E = xr.open_dataset(ANALYTICAL_ENERGY_FILE)

    if "analytical_energy_sem" not in ds_E:
        raise KeyError("Variable 'analytical_energy_sem' was not found.")

    analytical_energy = ds_E["analytical_energy_sem"].transpose("ensemble_member", "time")

    # Keep dataset attrs on the DataArray
    analytical_energy.attrs.update(ds_E.attrs)

    return analytical_energy


def compute_base_dt_persistence_rmse_from_nc_vectorized(
    u,
    max_horizon=None,
    test_member_start=50,
    test_member_end=100,
):
    # Select test members once
    u_test = u.sel(
        ensemble_member=slice(test_member_start, test_member_end - 1)
    ).values
    # shape: (n_members, n_time, n_grid)

    n_members, n_time, n_grid = u_test.shape

    if max_horizon is None:
        max_horizon = n_time - 1

    horizons = np.arange(0, max_horizon + 1,10, dtype=int)
    y_persistence = np.empty(len(horizons), dtype=float)

    for i, horizon in enumerate(horizons):
        u0 = u_test[:, : n_time - horizon, :]
        uh = u_test[:, horizon:, :]

        # RMSE over grid, then mean over members and start times
        rmse = np.sqrt(np.mean((uh - u0) ** 2, axis=2))
        y_persistence[i] = np.mean(rmse)

    return horizons, y_persistence

def plot_rollout_rmse_energy():
    data_rmse = load_metric("test_rmse_per_sample.csv")
    data_energy_pred = load_metric("test_energy_pred_per_sample.csv")
    data_energy_true = load_metric("test_energy_target_per_sample.csv")
    data_energy_rel = load_metric("test_energy_rel_error_per_sample.csv")

    analytical_energy = load_analytical_energy()
    u,ds = load_true_u()

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    dt_base = 0.015515220223

    horizons, y_pers_rmse = compute_base_dt_persistence_rmse_from_nc_vectorized(
        u=u,
        max_horizon=400,
    )

    x_pers_rmse = horizons * dt_base

    axes[0].loglog(
        x_pers_rmse,
        y_pers_rmse,
        linestyle="--",
        color="black",
        alpha=0.7,
        label="Persistence",
    )

    for dt_key, cfg in RUN_DIRS.items():
        dt_scale = cfg["dt_scale"]
        label = cfg["label"]

        # RMSE
        rmse_df = data_rmse[dt_key]
        rmse_cols, rmse_rollouts = get_rollout_cols(rmse_df)

        x_rmse = rmse_rollouts * dt_scale * 0.0155
        y_rmse = rmse_df[rmse_cols].mean(axis=0).values


        if dt_key == "100dt":
            x_rmse = x_rmse[:-1]
            y_rmse = y_rmse[:-1]

        axes[0].loglog(
            x_rmse,
            y_rmse,
            marker="o",
            linestyle="-",
            label=label,
        )

        # Relative energy error
        rel_energy_df = data_energy_rel[dt_key]
        metadata_df = load_metadata(dt_key)


        energy_cols, energy_rollouts = get_rollout_cols(rel_energy_df)

        x_energy = energy_rollouts * dt_scale * 0.0155
        y_energy = rel_energy_df[energy_cols].mean(axis=0).values

        if dt_key == "100dt":
            x_energy = x_energy[:-1]
            y_energy = y_energy[:-1]

        axes[1].loglog(
            x_energy,
            y_energy,
            marker="o",
            linestyle="-",
            label=label,
        )

    
    axes[0].set_ylabel("RMSE", fontsize=20)
    axes[0].legend(fontsize=20)
    axes[0].grid(True, which="both", alpha=0.4)

    axes[1].set_ylabel("Relative energy error", fontsize=20)
    axes[1].set_xlabel(r"Physical time (s)", fontsize=20)
    axes[1].legend(fontsize=20)
    axes[1].grid(True, which="both", alpha=0.4)
    axes[0].tick_params(axis="x", labelsize=18)
    axes[0].tick_params(axis="y", labelsize=18)
    axes[1].tick_params(axis="x", labelsize=18)
    axes[1].tick_params(axis="y", labelsize=18)

    fig.tight_layout()

    out_path = RESULTS_DIR / "rmse_energy_graf_reference.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path}")

    times, rmses = diagnose_persistence_period(u, member=50, start_idx=10)

    plt.figure(figsize=(10, 5))
    plt.plot(times, rmses)
    plt.axvline(2*np.pi, linestyle="--", color="black", label=r"$2\pi$")
    plt.xlabel("Time separation")
    plt.ylabel("Persistence RMSE")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.savefig("persistence_diagnositcs.png")

    dt_base = 0.015515220223

    print("Persistence horizons:")
    for r in rmse_rollouts:
        h = r * dt_scale * dt_base
        print(r, h, "distance to 2pi:", h - 2*np.pi)

    plt.figure()
    plt.plot(ds.time, u.sel(ensemble_member=50).isel(grid_index=1000))
    plt.savefig("Something.png")

    print(ds.time.values[:10])
    print(ds.attrs["Lmax"])
    print(ds.attrs["C"])


if __name__ == "__main__":

    plot_rollout_rmse_energy()

    

