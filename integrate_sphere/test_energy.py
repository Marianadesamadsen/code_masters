import sys
sys.path.insert(0, "./")
import numpy as np
from integrate_sphere.compute_energy import surface_mass_integration, compute_energy_over_time
import xarray as xr
import matplotlib.pyplot as plt
import data_generation_functions.SimulatorWaveEquation as simu


ds_correct = xr.open_dataset("GNN_training/one_wave/energy/analytical_energy_sem.nc")

energy_correct = ds_correct["analytical_energy_sem"].sel(ensemble_member=53).values

# Load data
ds = xr.open_dataset("GNN_training/one_wave/nc_files/wave_200_dtsmall_min10_g4_sigamin6_sigmax12_correctT_plus160dt.nc")
generations = 4
u_true = ds["u"].sel(ensemble_member=53).values
dt = ds.attrs["dt"]

# Precompute geometry once
out = surface_mass_integration(N=6, generation=generations, R=1)

fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

ut_order_fixed = 4

for N in [4, 6, 8]:
    out_N = surface_mass_integration(N=N, generation=generations, R=1)

    E = compute_energy_over_time(
        u_true,
        dt=dt,
        out=out_N,
        c=1,
        ut_order=ut_order_fixed
    )

    rel_drift = (E.max() - E.min()) / E[0]
    print(f"N={N}, ut_order={ut_order_fixed}: {rel_drift}")

    axs[0].plot(E, label=f"N={N}")

axs[0].plot(energy_correct,"--",color="black",label="SEM energy")
axs[0].set_xlabel("Time step")
axs[0].set_ylabel("Energy")
axs[0].set_title(rf"Varying SEM order $N$, $u_t$ order={ut_order_fixed}")
axs[0].grid(True)
axs[0].legend()


N_fixed = 6
out_fixed = surface_mass_integration(N=N_fixed, generation=generations, R=1)

for order in [2, 4, 6]:
    E = compute_energy_over_time(
        u_true,
        dt=dt,
        out=out_fixed,
        c=1,
        ut_order=order
    )

    rel_drift = (E.max() - E.min()) / E[0]
    print(f"N={N_fixed}, ut_order={order}: {rel_drift}")

    axs[1].plot(E, label=f"order={order}")

axs[1].plot(energy_correct,"--",color="black",label="SEM energy")
axs[1].set_xlabel("Time step")
axs[1].set_title(rf"Varying $u_t$ order, $N={N_fixed}$")
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.savefig("energy_comparison_N_and_ut_order.png", dpi=300)
plt.show()