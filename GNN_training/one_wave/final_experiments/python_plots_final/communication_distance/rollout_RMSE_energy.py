from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


BASE_DIR = Path("GNN_training/one_wave/different_mesh_size/final_results")
RESULTS_DIR = Path(
    "GNN_training/one_wave/different_mesh_size/final_results_plots/communicationdist/cont/other"
)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NC_FILE = Path(
    "GNN_training/one_wave/nc_files/"
    "wave_200_dtsmall_min10_g4_sigamin6_sigmax12_correctT_plus160dt.nc"
)

ANALYTICAL_ENERGY_FILE = Path(
    "GNN_training/one_wave/energy/analytical_energy_sem.nc"
)

DT_BASE = 0.015515220223

RUN_DIRS = {
    "40dt": {
        "label": r"$40\Delta t$",
        "result_dir": "test_40dt_2",
        "dt_scale": 40,
        "line": "-",
        "color": "blue",
    },
    "40dt sub 1": {
        "label": r"$40\Delta t$ sub 1",
        "result_dir": "test_40dt_sub1",
        "dt_scale": 40,
        "line": "-",
        "color": "red",
    },
    "40dt sub 2 nn91": {
        "label": r"$40\Delta t$ sub2 nn91",
        "result_dir": "test_40dt_sub2_nn91",
        "dt_scale": 40,
        "line": "-",
        "color": "purple",
    },
    "40dt sub 2 nn91 nn9": {
        "label": r"$40\Delta t$ sub2 nn91 nn9",
        "result_dir": "test_40dt_sub2_nn91_nn9",
        "dt_scale": 40,
        "line": "-",
        "color": "green",
    },
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

    return pd.read_csv(metadata_path)


def load_analytical_energy():
    ds_E = xr.open_dataset(ANALYTICAL_ENERGY_FILE)

    if "analytical_energy_sem" not in ds_E:
        raise KeyError("Variable 'analytical_energy_sem' was not found.")

    analytical_energy = ds_E["analytical_energy_sem"].transpose(
        "ensemble_member", "time"
    )

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
        raise ValueError(
            "metadata_df and pred_energy_df do not have the same number of rows."
        )

    member_values = analytical_energy["ensemble_member"].values.astype(int)
    member_to_index = {m: i for i, m in enumerate(member_values)}

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
            energy_time_index = original_time_index - cut

            if 0 <= energy_time_index < E_true_all.shape[1]:
                E_true[row, col] = E_true_all[member_index, energy_time_index]

    rel_error = np.abs(E_pred - E_true) / (np.abs(E_true) + 1e-12)

    rel_error_df = pd.DataFrame(rel_error, columns=energy_cols)
    true_energy_df = pd.DataFrame(E_true, columns=energy_cols)

    return rel_error_df, true_energy_df, energy_rollouts


def recompute_and_save_sem_energy_errors():
    data_energy_pred = load_metric("test_energy_pred_per_sample.csv")
    analytical_energy = load_analytical_energy()

    data_energy_rel_sem = {}

    for dt_key, cfg in RUN_DIRS.items():
        pred_energy_df = data_energy_pred[dt_key]
        metadata_df = load_metadata(dt_key)

        rel_energy_df, true_energy_df, _ = compute_relative_energy_error_from_sem_nc(
            pred_energy_df=pred_energy_df,
            metadata_df=metadata_df,
            analytical_energy=analytical_energy,
            dt_scale=cfg["dt_scale"],
        )

        model_save_dir = BASE_DIR / cfg["result_dir"]

        rel_energy_df.to_csv(
            model_save_dir / "test_energy_rel_error_sem_reference_per_sample.csv",
            index=False,
        )

        true_energy_df.to_csv(
            model_save_dir / "test_energy_target_sem_reference_per_sample.csv",
            index=False,
        )

        data_energy_rel_sem[dt_key] = rel_energy_df

    return data_energy_rel_sem


def plot_rollout_rmse_energy():
    data_rmse = load_metric("test_rmse_per_sample.csv")
    data_energy_rel = recompute_and_save_sem_energy_errors()

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    for dt_key, cfg in RUN_DIRS.items():
        dt_scale = cfg["dt_scale"]
        label = cfg["label"]
        color = cfg["color"]
        linestyle = cfg["line"]


        rmse_df = data_rmse[dt_key]
        rmse_cols, rmse_rollouts = get_rollout_cols(rmse_df)

        x_rmse = rmse_rollouts * dt_scale * DT_BASE
        y_rmse = rmse_df[rmse_cols].mean(axis=0).values

        axes[0].loglog(
            x_rmse,
            y_rmse,
            marker="o",
            linestyle=linestyle,
            color=color,
            label=label,
        )

  
        energy_df = data_energy_rel[dt_key]
        energy_cols, energy_rollouts = get_rollout_cols(energy_df)

        x_energy = energy_rollouts * dt_scale * DT_BASE
        y_energy = energy_df[energy_cols].mean(axis=0).values

        axes[1].loglog(
            x_energy,
            y_energy,
            marker="o",
            linestyle=linestyle,
            color=color,
            label=label,
        )

    axes[0].set_ylabel("RMSE", fontsize=20)
    axes[0].legend(fontsize=16)
    axes[0].grid(True, which="both", alpha=0.4)
    axes[0].tick_params(axis="both", labelsize=18)

    axes[1].set_ylabel("Relative energy error", fontsize=20)
    axes[1].set_xlabel("Physical time (s)", fontsize=20)
    axes[1].legend(fontsize=16)
    axes[1].grid(True, which="both", alpha=0.4)
    axes[1].tick_params(axis="both", labelsize=18)

    fig.tight_layout()

    out_path = RESULTS_DIR / "rmse_energy_sem_reference_other.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    plot_rollout_rmse_energy()