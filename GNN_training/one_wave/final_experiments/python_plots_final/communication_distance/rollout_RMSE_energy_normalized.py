from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import sys

sys.path.insert(0, "/zhome/5e/a/152106/code_masters")

import torch
from integrate_sphere.compute_energy import (
    surface_mass_integration,
    energy_out_to_torch,
    compute_energy_over_time_torch,
)


BASE_DIR = Path("GNN_training/one_wave/different_mesh_size/final_results")
RESULTS_DIR = Path("GNN_training/one_wave/different_mesh_size/final_results_plots/communicationdist")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NC_FILE = Path(
    "GNN_training/one_wave/nc_files/"
    "wave_200_dtsmall_min10_g4_sigamin6_sigmax12_correctT_plus160dt.nc"
)

TEST_TIME_STRIDE = 10
TEST_MEMBER_START = 50
TEST_MEMBER_END = 100
FEATURE_IDX = 0


RUN_DIRS = {
    "40dt": {"label": r"$40\Delta t$", "result_dir": "test_40dt_2", "dt_scale": 40,"line":"-","color":"blue"},
    "40dt mp 2": {"label": r"$40\Delta t$ mp 2", "result_dir": "test_40dt_mp2_new_final", "dt_scale": 40,"line":"-","color":"green"},
    "40dt sub 1": {"label": r"$40\Delta t$ sub 1", "result_dir": "test_40dt_sub1", "dt_scale": 40,"line":"--","color":"red"},
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

    print("u shape:", ds["u"].shape)
    print("time first values:", ds["time"].values[:5])
    print("time[80]:", ds["time"].values[80])
    print("u max:", ds["u"].max().item())
    print("u min:", ds["u"].min().item())

    return u


def compute_persistence_rmse_curve_from_nc(u, rollouts, dt_scale):
    max_horizon = int(np.max(rollouts) * dt_scale)
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

            u0 = u_member[start_indices, :]
            uh = u_member[start_indices + horizon, :]

            rmse_per_start = np.sqrt(np.mean((uh - u0) ** 2, axis=1))
            rmse_values.append(rmse_per_start)

        rmse_values = np.concatenate(rmse_values)
        y_persistence.append(np.mean(rmse_values))

    return np.array(y_persistence)


def compute_persistence_energy_rel_error_from_csv(energy_target_df):
    energy_cols, energy_rollouts = get_rollout_cols(energy_target_df)

    e_persistence = energy_target_df[energy_cols[0]].values[:, None]
    e_target = energy_target_df[energy_cols].values

    rel_error = np.abs(e_persistence - e_target) / (np.abs(e_target) + 1e-12)

    y_persistence_energy = rel_error.mean(axis=0)

    return y_persistence_energy, energy_rollouts


def compute_persistence_energy_rel_error_from_nc(u, rollouts, dt_scale):
    max_horizon = int(np.max(rollouts) * dt_scale)
    n_time = u.sizes["time"]

    start_indices = np.arange(
        0,
        n_time - max_horizon - dt_scale,
        TEST_TIME_STRIDE,
        dtype=int,
    )

    member_indices = np.arange(TEST_MEMBER_START, TEST_MEMBER_END)

    energy_out_np = surface_mass_integration(N=6, generation=4, R=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    energy_out = energy_out_to_torch(
        energy_out_np,
        device=device,
        dtype=torch.float32,
    )

    ds = xr.open_dataset(NC_FILE)
    base_dt = float(ds.attrs["dt"])
    physical_dt = base_dt * dt_scale

    pers_rel_errors = []

    rollout_indices = np.array([0, 1] + list(rollouts), dtype=int)

    for member in member_indices:
        u_member = u.isel(ensemble_member=member).values

        for start in start_indices:
            time_indices = start + rollout_indices * dt_scale

            target_full_np = u_member[time_indices, :]

            pers_full_np = target_full_np.copy()
            pers_full_np[2:, :] = target_full_np[1, :]

            target_full = torch.tensor(
                target_full_np[None, :, :],
                dtype=torch.float32,
                device=device,
            )

            pers_full = torch.tensor(
                pers_full_np[None, :, :],
                dtype=torch.float32,
                device=device,
            )

            E_target = compute_energy_over_time_torch(
                target_full,
                out=energy_out,
                dt=physical_dt,
                c=1.0,
                ut_order=4,
            )

            E_pers = compute_energy_over_time_torch(
                pers_full,
                out=energy_out,
                dt=physical_dt,
                c=1.0,
                ut_order=4,
            )

            rel_error = torch.abs(E_pers - E_target) / (torch.abs(E_target) + 1e-12)

            pers_rel_errors.append(rel_error[0, 2:].detach().cpu().numpy())

    pers_rel_errors = np.stack(pers_rel_errors, axis=0)

    return pers_rel_errors.mean(axis=0)


def plot_rollout_rmse_energy():
    data_rmse = load_metric("test_rmse_per_sample.csv")

    # Raw predicted and true energy
    data_energy_pred = load_metric("test_energy_pred_per_sample.csv")
    data_energy_true = load_metric("test_energy_target_per_sample.csv")

    u = load_true_u()

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    plotted_true_energy = False

    for dt_key, cfg in RUN_DIRS.items():
        dt_scale = cfg["dt_scale"]
        label = cfg["label"]


        rmse_df = data_rmse[dt_key]
        rmse_cols, rmse_rollouts = get_rollout_cols(rmse_df)

        x_rmse = rmse_rollouts * dt_scale
        y_rmse = rmse_df[rmse_cols].mean(axis=0).values

        if dt_key == "100dt":
            x_rmse = x_rmse[:-1]
            y_rmse = y_rmse[:-1]

        print(f"RMSE: model {dt_key} one step:", y_rmse[0])
        print(f"RMSE: model {dt_key} all:", y_rmse)

        axes[0].loglog(
            x_rmse,
            y_rmse,
            marker="o",
            linestyle="-",
            label=label,
        )

 
        energy_pred_df = data_energy_pred[dt_key]
        energy_pred_cols, energy_rollouts = get_rollout_cols(energy_pred_df)

        x_energy = energy_rollouts * dt_scale
        y_energy_pred = energy_pred_df[energy_pred_cols].mean(axis=0).values

        if dt_key == "100dt":
            x_energy = x_energy[:-1]
            y_energy_pred = y_energy_pred[:-1]

        print(f"E_pred: model {dt_key} one step:", y_energy_pred[0])
        print(f"E_pred: model {dt_key} all:", y_energy_pred)

        axes[1].semilogx(
            x_energy,
            y_energy_pred,
            marker="o",
            linestyle="-",
            label=label,
        )

        energy_true_df = data_energy_true[dt_key]
        energy_true_cols, energy_true_rollouts = get_rollout_cols(energy_true_df)

        x_energy_true = energy_true_rollouts * dt_scale
        y_energy_true = energy_true_df[energy_true_cols].mean(axis=0).values

        if dt_key == "100dt":
            x_energy_true = x_energy_true[:-1]
            y_energy_true = y_energy_true[:-1]

        if not plotted_true_energy:
            axes[1].semilogx(
                x_energy_true,
                y_energy_true,
                linestyle="--",
                color="black",
                linewidth=2.0,
                alpha=0.8,
                label="True energy",
            )
            plotted_true_energy = True

    axes[0].set_ylabel("RMSE", fontsize=20)
    axes[0].legend(fontsize=16)
    axes[0].grid(True, which="both", alpha=0.4)

    axes[1].set_ylabel(r"Energy $E(t)$", fontsize=20)
    axes[1].set_xlabel("Physical time (s)", fontsize=20)
    axes[1].legend(fontsize=16)
    axes[1].grid(True, which="both", alpha=0.4)

    fig.suptitle(
        "RMSE and raw predicted energy over matched physical time",
        fontsize=22,
    )

    fig.tight_layout()

    out_path = RESULTS_DIR / "rmse_raw_predicted_true_energy.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    plot_rollout_rmse_energy()