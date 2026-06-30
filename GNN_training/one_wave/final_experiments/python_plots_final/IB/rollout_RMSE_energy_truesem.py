from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


BASE_DIR = Path("GNN_training/one_wave/different_mesh_size/final_results")
RESULTS_DIR = Path("GNN_training/one_wave/different_mesh_size/final_results_plots/IB/final")
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
    "1dt": {"result_dir": "test_1dt", "dt_scale": 1},
    # "10dt": {"label": r"$10\Delta t$", "result_dir": "test_10dt_2", "dt_scale": 10},
    # "20dt": {"label": r"$20\Delta t$", "result_dir": "test_20dt_2", "dt_scale": 20},
    # "40dt": {"label": r"$40\Delta t$", "result_dir": "test_40dt_2", "dt_scale": 40},
    # "80dt": {"label": r"$80\Delta t$", "result_dir": "test_80dt_2", "dt_scale": 80},
    # "100dt": {"label": r"$100\Delta t$", "result_dir": "test_100dt_2", "dt_scale": 100},
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


def compute_relative_energy_error_from_sem_nc(
    pred_energy_df,
    metadata_df,
    analytical_energy,
    dt_scale,
):

    energy_cols, energy_rollouts = get_rollout_cols(pred_energy_df)
    E_pred = pred_energy_df[energy_cols].values

    members = metadata_df["ensemble_member"].values.astype(int)
    sample_indices = metadata_df["sample_idx"].values.astype(int)

    if len(members) != E_pred.shape[0]:
        raise ValueError("metadata_df and pred_energy_df do not have the same number of rows.")

    member_values = analytical_energy["ensemble_member"].values.astype(int)
    member_to_index = {m: i for i, m in enumerate(member_values)}

    E = analytical_energy.sel(ensemble_member=50).values
    print((np.nanmax(E) - np.nanmin(E)) / abs(E[0]))
    E_true_all = analytical_energy.values

    ut_order = int(analytical_energy.attrs.get("ut_order", 4))
    cut = ut_order // 2

    E_true = np.full_like(E_pred, np.nan, dtype=float)

    for row in range(E_pred.shape[0]):
        member = members[row]
        sample_idx = sample_indices[row]

        if member not in member_to_index:
            continue

        member_index = member_to_index[member]

        for col, rollout in enumerate(energy_rollouts):
            original_time_index = sample_idx + int(rollout * dt_scale)

            # analytical_energy_sem starts at original time index = cut
            energy_time_index = original_time_index - cut

            if 0 <= energy_time_index < E_true_all.shape[1]:
                E_true[row, col] = E_true_all[member_index, energy_time_index]

    eps = 1e-12
    rel_error = np.abs(E_pred - E_true) / (np.abs(E_true) + eps)

    rel_error_df = pd.DataFrame(rel_error, columns=energy_cols)
    true_energy_df = pd.DataFrame(E_true, columns=energy_cols)
    pred_energy_df = pd.DataFrame(E_pred, columns=energy_cols)

    return rel_error_df, true_energy_df, energy_rollouts, pred_energy_df 
 
def compute_persistence_rmse_curve_from_nc_new2(u, metadata_df, rollouts, dt_scale):
    members = metadata_df["ensemble_member"].values.astype(int)
    sample_indices = metadata_df["sample_idx"].values.astype(int)

    n_time = u.sizes["time"]
    y_persistence = []

    for rollout in rollouts:
        horizon = int(rollout * dt_scale)
        rmse_values = []

        for member, sample_idx in zip(members, sample_indices):
            input_idx = sample_idx
            future_idx = sample_idx + horizon

            if future_idx >= n_time:
                continue

            u_pers = u.sel(ensemble_member=member).isel(time=input_idx).values
            u_true = u.sel(ensemble_member=member).isel(time=future_idx).values

            rmse = np.sqrt(np.mean((u_true - u_pers) ** 2))
            rmse_values.append(rmse)

        y_persistence.append(np.mean(rmse_values))

    return np.array(y_persistence)

def compute_persistence_rmse_curve_from_nc(u, rollouts, dt_scale):
    n_time = u.sizes["time"]

    max_rollout = int(np.max(rollouts))

    # t0 is the first input.
    # second input is t0 + dt_scale.
    # last target needed is t0 + (max_rollout + 1) * dt_scale.
    max_target_horizon = int((max_rollout + 1) * dt_scale)

    start_indices = np.arange(
        0,
        n_time - max_target_horizon,
        TEST_TIME_STRIDE,
        dtype=int,
    )

    member_indices = np.arange(TEST_MEMBER_START, TEST_MEMBER_END)

    y_persistence = []

    for rollout in rollouts:
        # Persistence prediction is the second input
        input_idx_offset = int(dt_scale)

        # rollout_1 target is two steps after t0: t0 + 2*dt_scale
        target_idx_offset = int((rollout + 1) * dt_scale)

        rmse_values = []

        for member in member_indices:
            u_member = u.isel(ensemble_member=member).values

            u_pers = u_member[start_indices + input_idx_offset, :]
            u_true = u_member[start_indices + target_idx_offset, :]

            rmse_per_start = np.sqrt(np.mean((u_true - u_pers) ** 2, axis=1))
            rmse_values.append(rmse_per_start)

        rmse_values = np.concatenate(rmse_values)
        y_persistence.append(np.mean(rmse_values))

    return np.array(y_persistence)

def compute_persistence_rmse_curve_from_nc_new(u, metadata_df, rollouts, dt_scale):
    members = metadata_df["ensemble_member"].values.astype(int)
    sample_indices = metadata_df["sample_idx"].values.astype(int)

    n_time = u.sizes["time"]
    y_persistence = []

    for rollout in rollouts:
        horizon = int(rollout * dt_scale)
        rmse_values = []

        for member, sample_idx in zip(members, sample_indices):
            future_idx = sample_idx + horizon

            if future_idx >= n_time:
                continue

            u0 = u.sel(ensemble_member=member).isel(time=sample_idx).values
            uh = u.sel(ensemble_member=member).isel(time=future_idx).values

            rmse = np.sqrt(np.mean((uh - u0) ** 2))
            rmse_values.append(rmse)

        if len(rmse_values) == 0:
            y_persistence.append(np.nan)
        else:
            y_persistence.append(np.mean(rmse_values))

    return np.array(y_persistence)

def compute_persistence_energy_error_curve_from_sem_nc(
    metadata_df,
    analytical_energy,
    rollouts,
    dt_scale,
):
    members = metadata_df["ensemble_member"].values.astype(int)
    sample_indices = metadata_df["sample_idx"].values.astype(int)

    member_values = analytical_energy["ensemble_member"].values.astype(int)
    member_to_index = {m: i for i, m in enumerate(member_values)}

    E_true_all = analytical_energy.values

    ut_order = int(analytical_energy.attrs.get("ut_order", 4))
    cut = ut_order // 2

    ree_values = []

    for rollout in rollouts:
        errors = []

        for row in range(len(metadata_df)):
            member = members[row]
            sample_idx = sample_indices[row]

            if member not in member_to_index:
                continue

            member_index = member_to_index[member]

            # Persistence energy = energy of initial state
            initial_time_index = sample_idx
            future_time_index = sample_idx + int(rollout * dt_scale)

            initial_energy_index = initial_time_index - cut
            future_energy_index = future_time_index - cut

            if (
                0 <= initial_energy_index < E_true_all.shape[1]
                and 0 <= future_energy_index < E_true_all.shape[1]
            ):
                E_pers = E_true_all[member_index, initial_energy_index]
                E_true = E_true_all[member_index, future_energy_index]

                error = abs(E_pers - E_true) / (abs(E_true) + 1e-12)
                errors.append(error)

        ree_values.append(np.mean(errors))

    return np.array(ree_values)

def plot_rollout_rmse_energy():
    data_rmse = load_metric("test_rmse_per_sample.csv")
    data_energy_pred = load_metric("test_energy_pred_per_sample.csv")
    data_energy_true = load_metric("test_energy_target_per_sample.csv")

    analytical_energy = load_analytical_energy()
    u,ds = load_true_u()

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    for dt_key, cfg in RUN_DIRS.items():
        dt_scale = cfg["dt_scale"]
        #label = cfg["label"]

    
        rmse_df = data_rmse[dt_key]
        rmse_cols, rmse_rollouts = get_rollout_cols(rmse_df)

        x_rmse = rmse_rollouts * dt_scale * 0.0155
        y_rmse = rmse_df[rmse_cols].mean(axis=0).values


        if dt_key == "100dt":
            x_rmse = x_rmse[:-1]
            y_rmse = y_rmse[:-1]

        # Persistence RMSE baseline
        if dt_key == "40dt":
            y_pers_rmse = compute_persistence_rmse_curve_from_nc(
                u=u,
                rollouts=rmse_rollouts,
                dt_scale=dt_scale,
            )
            # metadata_df = load_metadata(dt_key)

            # y_pers_rmse = compute_persistence_rmse_curve_from_nc(
            #     u=u,
            #     metadata_df=metadata_df,
            #     rollouts=rmse_rollouts,
            #     dt_scale=dt_scale,
            # )

            x_pers_rmse = rmse_rollouts * dt_scale * 0.0155

            if dt_key == "100dt":
                x_pers_rmse = x_pers_rmse[:-1]
                y_pers_rmse = y_pers_rmse[:-1]

            axes[0].loglog(
                x_pers_rmse,
                y_pers_rmse,
                linestyle="--",
                color="black",
                alpha=0.7,
                label="Persistence",
            )
        axes[0].loglog(
            x_rmse,
            y_rmse,
            marker="o",
            linestyle="-",
            label="RMSE",
        )


        # Relative energy error
        pred_energy_df = data_energy_pred[dt_key]
        metadata_df = load_metadata(dt_key)

        rel_energy_df, true_energy_df, energy_rollouts, pred_energy_df = (
            compute_relative_energy_error_from_sem_nc(
                pred_energy_df=pred_energy_df,
                metadata_df=metadata_df,
                analytical_energy=analytical_energy,
                dt_scale=dt_scale,
            )
        )

        # Save recomputed energy files for this model
        model_save_dir = BASE_DIR / cfg["result_dir"]

        rel_energy_df.to_csv(
            model_save_dir / "test_energy_rel_error_sem_reference_per_sample.csv",
            index=False,
        )

        true_energy_df.to_csv(
            model_save_dir / "test_energy_target_sem_reference_per_sample.csv",
            index=False,
        )

        energy_cols, _ = get_rollout_cols(rel_energy_df)

        x_energy = energy_rollouts * dt_scale * 0.0155
        y_energy = rel_energy_df[energy_cols].mean(axis=0).values

        if dt_key == "40dt":
            idx_max = np.argmax(y_energy)
            time_peak = x_energy[idx_max]
            print("Timepeak:",time_peak)
            #axes[1].axvline(time_peak,linestyle="--",color="black",label=f"t={time_peak}")
            #axes[0].axvline(time_peak,linestyle="--",color="black",label=f"t={time_peak}")


        if dt_key == "100dt":
            x_energy = x_energy[:-1]
            y_energy = y_energy[:-1]

        axes[1].loglog(
            x_energy,
            y_energy,
            marker="o",
            linestyle="-",
            label="Relative energy error",
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
    #axes[1].axvline(np.pi,linestyle="--",color="black")
    #axes[0].axvline(np.pi,linestyle="--",color="black")

    fig.tight_layout()

    out_path = RESULTS_DIR / "rmse_energy_sem_reference.png"
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

    

