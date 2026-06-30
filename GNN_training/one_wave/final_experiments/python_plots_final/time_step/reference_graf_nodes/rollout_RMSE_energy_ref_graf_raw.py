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
    "1dt": {"label": r"$1\Delta t$", "result_dir": "test_1dt", "dt_scale": 1},
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
    return u

def load_analytical_energy():
    ds_E = xr.open_dataset(ANALYTICAL_ENERGY_FILE)

    if "analytical_energy_sem" not in ds_E:
        raise KeyError("Variable 'analytical_energy_sem' was not found.")

    analytical_energy = ds_E["analytical_energy_sem"].transpose("ensemble_member", "time")

    # Keep dataset attrs on the DataArray
    analytical_energy.attrs.update(ds_E.attrs)

    return analytical_energy


def plot_rollout_rmse_energy():
    data_rmse = load_metric("test_rmse_per_sample.csv")
    data_energy_pred = load_metric("test_energy_pred_per_sample.csv")
    data_energy_true = load_metric("test_energy_target_per_sample.csv")
    data_energy_rel = load_metric("test_energy_rel_error_per_sample.csv")

    analytical_energy = load_analytical_energy()
    u = load_true_u()

    fig, axes = plt.subplots(1, 1, figsize=(16, 10), sharex=True)

    for dt_key, cfg in RUN_DIRS.items():
        dt_scale = cfg["dt_scale"]
        label = cfg["label"]


        # -----------------------------
        # Relative energy error
        # -----------------------------
        pred_energy_df = data_energy_pred[dt_key]
        true_energy_df = data_energy_true[dt_key]
        rel_energy_df = data_energy_rel[dt_key]


        energy_cols, energy_rollouts  = get_rollout_cols(rel_energy_df)

        x_energy = energy_rollouts * dt_scale * 0.0155
        y_energy = rel_energy_df[energy_cols].mean(axis=0).values

        if dt_key == "100dt":
            x_energy = x_energy[:-1]
            y_energy = y_energy[:-1]

        y_energy_pred = pred_energy_df[energy_cols].mean(axis=0).values
        y_energy_true = true_energy_df[energy_cols].mean(axis=0).values
        if dt_scale == 100:
            y_energy_true = y_energy_true[:2]
            y_energy_pred = y_energy_pred[:2]

        axes.loglog(
           x_energy,
            y_energy_pred,
            marker="o",
            linestyle="-",
            label = label
        )
        axes.loglog(
            x_energy,
            y_energy_true,
            #marker="o",
            linestyle="--",
            color="black"
        )   
        idx_max = np.argmax(y_energy_pred)
        print("idx:",idx_max)

    axes.loglog(
    x_energy,
    y_energy_true,
    #marker="o",
    linestyle="--",
    label="Analytical energy",
    color="black"
    )

    axes.set_ylabel("Energy", fontsize=20)
    axes.set_xlabel(r"Physical time (s)", fontsize=20)
    axes.legend(fontsize=20)
    axes.grid(True, which="both", alpha=0.4)
    axes.tick_params(axis="x", labelsize=18)
    axes.tick_params(axis="y", labelsize=18)

    fig.tight_layout()

    out_path = RESULTS_DIR / "rmse_energy_graf_reference_raw_energy.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    plot_rollout_rmse_energy()