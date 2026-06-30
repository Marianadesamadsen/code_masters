from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path("GNN_training/one_wave/different_mesh_size/final_results")
RESULTS_DIR = Path("GNN_training/one_wave/different_mesh_size/final_results_plots/AR/final")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DT_COLORS = {
    10: "tab:blue",
    20: "tab:orange",
    40: "tab:green",
}

RUN_PAIRS = {
    10: {
        "label": r"$10\Delta t$",
        "ar1_dir": "test_10dt_2",
        "ar2_dir": "test_10dt_AR_2",
    },
    20: {
        "label": r"$20\Delta t$",
        "ar1_dir": "test_20dt_2",
        "ar2_dir": "test_20dt_AR_2",
    },
    40: {
        "label": r"$40\Delta t$",
        "ar1_dir": "test_40dt_2",
        "ar2_dir": "test_40dt_AR_2",
    },
}


def get_rollout_cols(df):
    cols = [c for c in df.columns if c.startswith("rollout_")]
    cols = sorted(cols, key=lambda c: int(c.split("_")[-1]))
    rollouts = np.array([int(c.split("_")[-1]) for c in cols])
    return cols, rollouts


def load_metric(result_dir, filename):
    csv_path = BASE_DIR / result_dir / filename

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    return pd.read_csv(csv_path)


def compute_percentage_improvement(ar1_df, ar2_df):

    ar1_cols, ar1_rollouts = get_rollout_cols(ar1_df)
    ar2_cols, ar2_rollouts = get_rollout_cols(ar2_df)

    common_rollouts = np.intersect1d(ar1_rollouts, ar2_rollouts)

    x_rollouts = []
    improvement = []

    for rollout in common_rollouts:
        ar1_col = f"rollout_{rollout}"
        ar2_col = f"rollout_{rollout}"

        ar1_mean = ar1_df[ar1_col].mean()
        ar2_mean = ar2_df[ar2_col].mean()

        perc_improvement = 100.0 * (ar1_mean -ar2_mean ) / ar1_mean 

        x_rollouts.append(rollout)
        improvement.append(perc_improvement)

    return np.array(x_rollouts), np.array(improvement)


def plot_ar2_percentage_improvement():
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    for dt_scale, cfg in RUN_PAIRS.items():
        label = cfg["label"]
        color = DT_COLORS[dt_scale]

        # RMSE improvement
        ar1_rmse = load_metric(cfg["ar1_dir"], "test_rmse_per_sample.csv")
        ar2_rmse = load_metric(cfg["ar2_dir"], "test_rmse_per_sample.csv")

        rollouts, rmse_improvement = compute_percentage_improvement(
            ar1_df=ar1_rmse,
            ar2_df=ar2_rmse,
        )

        x_physical = rollouts * dt_scale *0.0155

        axes[0].plot(
            x_physical,
            rmse_improvement,
            marker="o",
            linewidth=2,
            color=color,
            label=label,
        )

        # Energy improvement
        ar1_energy = load_metric(cfg["ar1_dir"], "test_energy_rel_error_per_sample.csv")
        ar2_energy = load_metric(cfg["ar2_dir"], "test_energy_rel_error_per_sample.csv")

        rollouts, energy_improvement = compute_percentage_improvement(
            ar1_df=ar1_energy,
            ar2_df=ar2_energy,
        )

        x_physical = rollouts * dt_scale*0.0155

        axes[1].plot(
            x_physical,
            energy_improvement,
            marker="o",
            linewidth=2,
            color=color,
            label=label,
        )


    for ax in axes:
        ax.axhline(0.0, color="black", linewidth=1.5, linestyle="--")
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=18)
        ax.tick_params(axis="both", labelsize=18)

    axes[0].set_ylabel("RMSE improvement (%)", fontsize=20)
    axes[1].set_ylabel("Energy error improvement [%]", fontsize=20)
    axes[1].set_xlabel("Physical time (s)", fontsize=20)

    # fig.suptitle(
    #     "Percentage improvement from AR1 to AR2 in training",
    #     fontsize=22,
    # )

    fig.tight_layout()

    out_path = RESULTS_DIR / "ar2_percentage_improvement.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    plot_ar2_percentage_improvement()