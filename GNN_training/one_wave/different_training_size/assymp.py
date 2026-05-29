import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

energy_csv_path = (
    "GNN_training/one_wave/different_training_size/"
    "trainsize_validationlossenergy.csv"
)

rmse_csv_path = (
    "GNN_training/one_wave/different_training_size/"
    "trainsize_validationloss1.csv"
)

out_dir = (
    "GNN_training/one_wave/different_training_size/"
    "all_results_plot"
)

os.makedirs(out_dir, exist_ok=True)


train_sizes = [1, 10, 25, 50, 75]

energy_columns = [
    f"train_{i} - val_batch_energy_rel_error"
    for i in train_sizes
]

rmse_columns = [
    f"train_{i} - val_loss_unroll1_step"
    for i in train_sizes
]

samples_per_wave = 601

dataset_sizes = [
    size * samples_per_wave
    for size in train_sizes
]

window = 20000
tail_fraction = 0.15


def smooth_and_estimate_asymptote(df, columns, window, tail_fraction):
    df_smooth = (
        df[columns]
        .rolling(window=window, min_periods=1)
        .mean()
    )

    asymptotic_values = []

    for col in columns:
        y = df_smooth[col].dropna().values

        tail_start = int((1.0 - tail_fraction) * len(y))
        tail_values = y[tail_start:]

        asymptote = np.mean(tail_values)
        asymptotic_values.append(asymptote)

    return df_smooth, np.array(asymptotic_values)



df_energy = pd.read_csv(energy_csv_path)
df_rmse = pd.read_csv(rmse_csv_path)

num_steps = min(len(df_energy), len(df_rmse))
steps = np.arange(1, num_steps + 1)

df_energy = df_energy.iloc[:num_steps]
df_rmse = df_rmse.iloc[:num_steps]


energy_smooth, energy_asymptotes = smooth_and_estimate_asymptote(
    df=df_energy,
    columns=energy_columns,
    window=window,
    tail_fraction=tail_fraction,
)

rmse_smooth, rmse_asymptotes = smooth_and_estimate_asymptote(
    df=df_rmse,
    columns=rmse_columns,
    window=window,
    tail_fraction=tail_fraction,
)



fig, ax = plt.subplots(
    2,
    1,
    figsize=(12, 10),
    sharex=True,
)

# Relative energy error

for train_size, col, asymptote in zip(
    train_sizes,
    energy_columns,
    energy_asymptotes,
):
    ax[0].loglog(
        steps,
        energy_smooth[col],
        label=f"Train {train_size}",
    )

    ax[0].hlines(
        asymptote,
        xmin=steps[0],
        xmax=steps[-1],
        linestyles="--",
        linewidth=1,
        alpha=0.5,
    )

ax[0].set_title(
    f"Smoothed Relative Energy Error",
    fontsize=18,
)

ax[0].set_ylabel("Relative energy error", fontsize=16)
ax[0].grid(True, which="both", alpha=0.4)
ax[0].legend(fontsize=12)


for train_size, col, asymptote in zip(
    train_sizes,
    rmse_columns,
    rmse_asymptotes,
):
    ax[1].loglog(
        steps,
        rmse_smooth[col],
        label=f"Train {train_size}",
    )

    ax[1].hlines(
        asymptote,
        xmin=steps[0],
        xmax=steps[-1],
        linestyles="--",
        linewidth=1,
        alpha=0.5,
    )

ax[1].set_title(
    f"Smoothed RMSE / Validation Loss",
    fontsize=18,
)

ax[1].set_xlabel("Training step", fontsize=16)
ax[1].set_ylabel("RMSE / validation loss", fontsize=16)
ax[1].grid(True, which="both", alpha=0.4)
ax[1].legend(fontsize=12)

fig.suptitle(
    f"Smoothed validation curves with estimated asymptotes\n"
    f"Moving average window = {window}, tail fraction = {tail_fraction}",
    fontsize=20,
)

plt.tight_layout(rect=[0, 0, 1, 0.95])

out_path_1 = os.path.join(
    out_dir,
    "energy_rmse_loglog_smoothed_asymptotes.png",
)

plt.savefig(out_path_1, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved: {out_path_1}")


fig, ax = plt.subplots(
    2,
    1,
    figsize=(9, 10),
    sharex=True,
)


ax[0].loglog(
    dataset_sizes,
    energy_asymptotes,
    marker="o",
    linewidth=2,
)

for x, y, train_size in zip(dataset_sizes, energy_asymptotes, train_sizes):
    ax[0].annotate(
        f"{train_size} waves",
        xy=(x, y),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=11,
    )

ax[0].set_title(
    "Asymptotic relative energy error",
    fontsize=18,
)

ax[0].set_ylabel(
    "Estimated asymptotic relative energy error",
    fontsize=14,
)

ax[0].grid(True, which="both", alpha=0.4)

ax[1].loglog(
    dataset_sizes,
    rmse_asymptotes,
    marker="o",
    linewidth=2,
)

for x, y, train_size in zip(dataset_sizes, rmse_asymptotes, train_sizes):
    ax[1].annotate(
        f"{train_size} waves",
        xy=(x, y),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=11,
    )

ax[1].set_title(
    "Asymptotic RMSE loss",
    fontsize=18,
)

ax[1].set_xlabel("Number of training samples", fontsize=16)

ax[1].set_ylabel(
    "Estimated asymptotic RMSE loss",
    fontsize=14,
)

ax[1].grid(True, which="both", alpha=0.4)

plt.tight_layout()

out_path_2 = os.path.join(
    out_dir,
    "asymptotic_energy_rmse_vs_dataset_size.png",
)

plt.savefig(out_path_2, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved: {out_path_2}")


def fit_power_law(dataset_sizes, values, label):
    log_N = np.log(dataset_sizes)
    log_E = np.log(values)

    slope, intercept = np.polyfit(log_N, log_E, 1)

    alpha = -slope
    C = np.exp(intercept)

    print(f"\nEstimated power law for {label}:")
    print(f"{label} ≈ {C:.3e} * N^(-{alpha:.3f})")

    return C, alpha


energy_C, energy_alpha = fit_power_law(
    dataset_sizes,
    energy_asymptotes,
    "Relative energy error",
)

rmse_C, rmse_alpha = fit_power_law(
    dataset_sizes,
    rmse_asymptotes,
    "RMSE / validation loss",
)



summary_df = pd.DataFrame({
    "train_size_waves": train_sizes,
    "dataset_size_samples": dataset_sizes,
    "asymptotic_relative_energy_error": energy_asymptotes,
    "asymptotic_rmse_or_validation_loss": rmse_asymptotes,
})

print("\nAsymptotic values:")
print(summary_df)

summary_path = os.path.join(
    out_dir,
    "asymptotic_energy_rmse_summary.csv",
)

summary_df.to_csv(summary_path, index=False)

print(f"\nSaved summary: {summary_path}")