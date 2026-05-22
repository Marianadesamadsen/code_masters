import sys
sys.path.insert(0, "./")

from integrate_sphere.compute_energy import surface_mass_integration, compute_energy_over_time
import xarray as xr
import matplotlib.pyplot as plt

# Load data
ds = xr.open_dataset("GNN_training/one_wave/nc_files/wave_200_ts_600_g4_sigmamin_15.nc")

u_true = ds["u"].isel(ensemble_member=20).values
dt = ds.attrs["dt"]
print("dataset dt:", ds.attrs["dt"]) 

# Precompute geometry once
out = surface_mass_integration(N=6, generation=4, R=1)

fig, axs = plt.subplots(1, 2, figsize=(12, 4))

for order in [2,4,6]:
    E = compute_energy_over_time(
        u_true,
        dt=dt,
        out=out,
        c=1,
        ut_order=order
    )

    rel_drift = (E.max() - E.min()) / E[0]
    print(f"{order}: {rel_drift}")

    axs[0].plot(E, label=f"ut_order={order}")
    axs[1].plot(E / E[0], label=f"ut_order={order}")

# Raw energy plot
axs[0].set_xlabel("Time step")
axs[0].set_ylabel("Energy")
axs[0].set_title("Energy over time ")
axs[0].grid(True)
axs[0].legend()

# Normalized energy plot
axs[1].set_xlabel("Time step")
axs[1].set_ylabel("Energy / E(0)")
axs[1].set_title("Energy over time (normalized)")
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.savefig("energy_comparison_change_dt_order.png")
fig, axs = plt.subplots(1, 2, figsize=(12, 4))

ut_order = 4   # keep fixed while comparing N

for N in [2,4,6]:
    out = surface_mass_integration(N=N, generation=4, R=1)

    E = compute_energy_over_time(
        u_true,
        dt=dt,
        out=out,
        c=1,
        ut_order=ut_order
    )

    rel_drift = (E.max() - E.min()) / E[0]
    print(f"N={N}: {rel_drift}")

    axs[0].plot(E, label=f"N={N}")
    axs[1].plot(E / E[0], label=f"N={N}")

axs[0].set_xlabel("Time step")
axs[0].set_ylabel("Energy")
axs[0].set_title(f"Energy over time, ut order={ut_order}")
axs[0].grid(True)
axs[0].legend()

axs[1].set_xlabel("Time step")
axs[1].set_ylabel("Energy / E(0)")
axs[1].set_title(f"Energy over time (normalized), ut order={ut_order}")
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.savefig("energy_comparison_change_N.png")