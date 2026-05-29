"""
Plotting script for message-passing-depth experiments.

This version:
- Compares different message-passing steps instead of training sizes.
- Loads test_metadata.csv for each MP-depth result folder.
- Uses true ensemble_member from metadata.
- Does NOT infer wave_id from row order.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

from matplotlib.colors import LogNorm, SymLogNorm
from scipy.stats import pearsonr


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

BASE_DIR = Path("GNN_training/one_wave/different_mp")
RESULTS_DIR = Path("GNN_training/one_wave/different_mp/results")
DATASET_PATH = Path("GNN_training/one_wave/nc_files/wave_200_ts_600_g4_sigmamin_6.nc")

# Message-passing step: result folder name
MP_RUNS = {
    1: "test_mp1_results_new",
    2: "test_mp2_results_new",
    3: "test_mp3_results_new",
}

MP_STEPS = list(MP_RUNS.keys())
MAIN_MP_STEPS = MP_STEPS

RMSE_NORM = LogNorm(vmin=1e-4, vmax=1e-1)
REL_ENERGY_NORM = LogNorm(vmin=1e-4, vmax=1e-1)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


METRIC_FILENAMES = {
    "pred_energy": "test_energy_pred_per_sample.csv",
    "target_energy": "test_energy_target_per_sample.csv",
    "rel_error": "test_energy_rel_error_per_sample.csv",
    "abs_error": "test_energy_abs_error_per_sample.csv",
    "mse": "test_mse_per_sample.csv",
    "rmse": "test_rmse_per_sample.csv",
}


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_test_metadata(result_dir):
    path = Path(result_dir) / "test_metadata.csv"
    return pd.read_csv(path)


def load_mp_results(base_dir, mp_runs):
    results = {}

    for mp_step, folder_name in mp_runs.items():
        result_dir = Path(base_dir) / folder_name

        if not result_dir.exists():
            print(f"Skipping MP step {mp_step}: folder does not exist: {result_dir}")
            continue

        results[mp_step] = {}

        for metric_name, filename in METRIC_FILENAMES.items():
            path = result_dir / filename
            results[mp_step][metric_name] = pd.read_csv(path)

        results[mp_step]["metadata"] = load_test_metadata(result_dir)

    return results


def load_initial_condition_metadata(dataset_path):
    ds = xr.open_dataset(dataset_path)

    metadata = {
        "centers": ds["center"].values,
        "sigmas": ds["sigma_deg"].values,
        "amplitudes": ds["A"].values,
    }

    return metadata


# ---------------------------------------------------------------------
# Per-wave utilities
# ---------------------------------------------------------------------

def get_rollout_columns(df_metric):
    return [col for col in df_metric.columns if col.startswith("rollout_")]


def attach_metadata(df_metric, df_meta):
    if len(df_metric) != len(df_meta):
        raise ValueError(
            f"Metric and metadata length mismatch: "
            f"{len(df_metric)=}, {len(df_meta)=}"
        )

    df = df_metric.copy()
    df["ensemble_member"] = df_meta["ensemble_member"].values
    df["sample_idx"] = df_meta["sample_idx"].values
    return df


def compute_per_wave_metric(df_metric, df_meta):
    df = attach_metadata(df_metric, df_meta)
    rollout_cols = get_rollout_columns(df)
    return df.groupby("ensemble_member")[rollout_cols].mean()


def get_wave_scores(df_metric, df_meta):
    per_wave = compute_per_wave_metric(df_metric, df_meta)
    return per_wave.mean(axis=1)


def get_wave_scores_at_rollout(df_metric, df_meta, rollout_idx):
    per_wave = compute_per_wave_metric(df_metric, df_meta)
    rollout_col = per_wave.columns[rollout_idx]
    return per_wave[rollout_col]


def build_wave_dataframe(score, metadata):
    ensemble_members = score.index.to_numpy(dtype=int)

    return pd.DataFrame({
        "ensemble_member": ensemble_members,
        "score": score.values,
        "sigma": metadata["sigmas"][ensemble_members],
        "A": metadata["amplitudes"][ensemble_members],
        "center_x": metadata["centers"][ensemble_members, 0],
        "center_y": metadata["centers"][ensemble_members, 1],
        "center_z": metadata["centers"][ensemble_members, 2],
    })


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------

def save_figure(fig, filename):
    path = RESULTS_DIR / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def mp_label(mp_step):
    return f"MP {mp_step}"



def add_initial_parameters_to_samples(df_metric, df_meta, metadata):
    df = attach_metadata(df_metric, df_meta)

    ensemble_members = df["ensemble_member"].to_numpy(dtype=int)

    df["sigma"] = metadata["sigmas"][ensemble_members]
    df["A"] = metadata["amplitudes"][ensemble_members]

    return df

def plot_worst_rmse_mp3_and_worst_energy_mp1(
    results,
    plot_mp_steps,
    n=5,
    filename="worst_rmse_mp3_worst_energy_mp1.png",
):
    selection_specs = [
        {
            "title": f"Worst RMSE trajectory from MP 1",
            "select_mp": 1,
            "metric_key": "rmse",
            "ylabel": "RMSE",
        },
        {
            "title": f"Worst relative energy error trajectory from MP 1",
            "select_mp": 1,
            "metric_key": "rel_error",
            "ylabel": "Relative energy error",
        },
    ]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5),
        sharey=False,
    )

    for ax, spec in zip(axes, selection_specs):
        select_mp = spec["select_mp"]
        metric_key = spec["metric_key"]

        scores = get_wave_scores(
            results[select_mp][metric_key],
            results[select_mp]["metadata"],
        )

        worst_members = scores.nlargest(n).index.to_numpy(dtype=int)

        for mp_step in plot_mp_steps:
            if mp_step not in results:
                print(f"Skipping MP {mp_step}: not available.")
                continue

            if metric_key not in results[mp_step]:
                print(f"Skipping MP {mp_step}: {metric_key} not available.")
                continue

            df = attach_metadata(
                results[mp_step][metric_key],
                results[mp_step]["metadata"],
            )

            rollout_cols = get_rollout_columns(df)

            group = df[df["ensemble_member"].isin(worst_members)]

            if len(group) == 0:
                print(
                    f"No matching samples for {spec['title']} "
                    f"in MP {mp_step}"
                )
                continue

            mean_curve = group[rollout_cols].mean(axis=0)

            ax.loglog(
                np.arange(1, len(rollout_cols) + 1),
                mean_curve,
                marker="o",
                linestyle="--",
                label=f"MP {mp_step}",
            )

        ax.set_title(spec["title"])
        ax.set_xlabel("Rollout")
        ax.set_ylabel(spec["ylabel"])
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    fig.suptitle(
        "Performance over rollout for worst-case initial condition waves",
        fontsize=16,
    )

    fig.tight_layout()
    save_figure(fig, filename)


def plot_rmse_and_relative_error_over_time(results):
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    for mp_step in results:

        rmse = results[mp_step]["rmse"]
        rel_error = results[mp_step]["rel_error"]

        rollouts_rmse = np.arange(1, rmse.shape[1] + 1)
        rollouts_error = np.arange(1, rel_error.shape[1] + 1)

        mean_rmse = rmse.mean()
        mean_rel_error = rel_error.mean()

        axes[0].loglog(
            rollouts_rmse,
            mean_rmse,
            label=mp_label(mp_step),
            linestyle="--",
            marker="o",
        )

        axes[1].loglog(
            rollouts_error,
            mean_rel_error,
            label=mp_label(mp_step),
            linestyle="--",
            marker="o",
        )

    axes[0].set_ylabel("RMSE", fontsize=18)
    axes[0].legend(fontsize=14)
    axes[0].grid(True, which="both")

    axes[1].set_ylabel("Relative energy error", fontsize=18)
    axes[1].set_xlabel("Rollout", fontsize=18)
    axes[1].legend(fontsize=14)
    axes[1].grid(True, which="both")

    fig.suptitle(
        "RMSE and relative energy error over rollout time",
        fontsize=20,
    )

    fig.tight_layout(pad=3.0)

    save_figure(
        fig,
        "rmse_and_relative_energy_error_over_time_mp.png",
    )

def plot_best_worst_metric_directional_slices_compare_mp(
    results,
    ds_data,
    mp_runs,
    base_dir,
    metric_key="rmse",
    reference_mp=None,
    rollout_indices=(0, 9, 19),
    slice_width=0.04,
    filename=None,
    sample_position_within_wave=0,
):
    mp_steps = [mp for mp in mp_runs.keys() if mp in results]

    if len(mp_steps) == 0:
        print("Skipping best/worst directional slice plot: no MP results available.")
        return

    if reference_mp is None:
        reference_mp = mp_steps[-1]

    if reference_mp not in results:
        raise ValueError(f"reference_mp={reference_mp} is not available.")

    metric_scores = get_wave_scores(
        results[reference_mp][metric_key],
        results[reference_mp]["metadata"],
    )

    best_member = int(metric_scores.idxmin())
    worst_member = int(metric_scores.idxmax())

    metric_label = {
        "rmse": "RMSE",
        "rel_error": "relative energy error",
    }.get(metric_key, metric_key)


    selected_waves = {
        f"Best": best_member,
        f"Worst": worst_member,
    }

    ds_preds = {}
    for mp_step in mp_steps:
        pred_path = Path(base_dir) / f"test_mp{mp_step}_results_new.zarr"

        if not pred_path.exists():
            print(f"Skipping MP {mp_step}: prediction Zarr not found: {pred_path}")
            continue

        ds_preds[mp_step] = xr.open_zarr(pred_path)

    available_mp_steps = list(ds_preds.keys())

    if len(available_mp_steps) == 0:
        print("No prediction Zarr files found.")
        return

    xyz = np.stack(
        [
            ds_data["x_static"].values,
            ds_data["y_static"].values,
            ds_data["z_static"].values,
        ],
        axis=1,
    )

    n_rows = len(selected_waves) * len(rollout_indices)
    n_cols = len(available_mp_steps)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.2 * n_cols, 3.3 * n_rows),
        sharex=True,
        sharey=True,
    )

    if n_cols == 1:
        axes = axes[:, None]

    row_idx = 0

    for wave_label, ensemble_member in selected_waves.items():

        center = ds_data["center"].isel(
            ensemble_member=ensemble_member
        ).values
        center = center / np.linalg.norm(center)

        reference = np.array([0.0, 0.0, 1.0])
        if np.abs(np.dot(center, reference)) > 0.95:
            reference = np.array([0.0, 1.0, 0.0])

        direction = reference - np.dot(reference, center) * center
        direction = direction / np.linalg.norm(direction)

        normal = np.cross(center, direction)
        normal = normal / np.linalg.norm(normal)

        distance_to_plane = np.abs(xyz @ normal)
        mask = distance_to_plane < slice_width

        if np.sum(mask) < 10:
            print(
                f"Warning: only {np.sum(mask)} nodes in slice for wave {ensemble_member}. "
                "Consider increasing slice_width."
            )

        xyz_slice = xyz[mask]

        signed_angle = np.arctan2(
            xyz_slice @ direction,
            xyz_slice @ center,
        )

        order = np.argsort(signed_angle)
        signed_angle = signed_angle[order]
        signed_angle = np.rad2deg(signed_angle)
        slice_node_indices = np.where(mask)[0][order]

        for rollout_idx in rollout_indices:
            for col, mp_step in enumerate(available_mp_steps):
                ax = axes[row_idx, col]

                df_meta = results[mp_step]["metadata"]

                matching_samples = np.where(
                    df_meta["ensemble_member"].to_numpy() == ensemble_member
                )[0]

                if len(matching_samples) == 0:
                    ax.set_title(f"MP {mp_step}: missing wave")
                    ax.axis("off")
                    continue

                if sample_position_within_wave >= len(matching_samples):
                    sample_idx = int(matching_samples[-1])
                else:
                    sample_idx = int(matching_samples[sample_position_within_wave])
                ds_pred = ds_preds[mp_step]

                target = ds_pred["target"].isel(
                    sample=sample_idx,
                    rollout_step=rollout_idx,
                    state_feature=0,
                ).values

                pred = ds_pred["prediction"].isel(
                    sample=sample_idx,
                    rollout_step=rollout_idx,
                    state_feature=0,
                ).values

                target_slice = target[slice_node_indices]
                pred_slice = pred[slice_node_indices]

                ax.plot(
                    signed_angle,
                    target_slice,
                    linewidth=2.0,
                    label="$u_t$",
                )

                ax.plot(
                    signed_angle,
                    pred_slice,
                    linewidth=2.0,
                    linestyle="--",
                    label="$\hat u_t$",
                )

                ax.axvline(0.0, color="black", linewidth=1, alpha=0.5)
                ax.grid(True)

                if row_idx == 0:
                    ax.set_title(f"MP {mp_step}",fontsize=20)

                if col == 0:
                    ax.set_ylabel(
                        f"{wave_label}\n"
                        f"rollout {rollout_idx + 1}\n" + r"$u$",fontsize=20
                    )

                if row_idx == n_rows - 1:
                    ax.set_xlabel("Angle from the center [deg]",fontsize=20)

            row_idx += 1

    axes[0, 0].legend(fontsize=20)

    fig.suptitle(
        f"Best and worst {metric_label} wave slices across message-passing depth"
        f" (based on MP {reference_mp})",
        fontsize=22,
    )

    fig.tight_layout(pad=2.0)

    if filename is None:
        filename = (
            f"best_worst_{metric_key}_directional_slices_compare_mp"
            f"_refmp{reference_mp}_rolloutstart{sample_position_within_wave}.png"
        )

    save_figure(fig, filename)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    results = load_mp_results(BASE_DIR, MP_RUNS)

    if len(results) == 0:
        raise RuntimeError("No MP result folders were loaded. Check BASE_DIR and MP_RUNS.")

    last_mp = list(results.keys())[-1]


    ds_data = xr.open_dataset(DATASET_PATH)

    plot_best_worst_metric_directional_slices_compare_mp(
        results=results,
        ds_data=ds_data,
        mp_runs=MP_RUNS,
        base_dir=BASE_DIR,
        metric_key="rmse",
        reference_mp=last_mp,
        rollout_indices=(0, 19),
        slice_width=0.04,
        sample_position_within_wave=0
    )

    plot_rmse_and_relative_error_over_time(results)

if __name__ == "__main__":
    main()
