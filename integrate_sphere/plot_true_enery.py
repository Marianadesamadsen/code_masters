import xarray as xr
import matplotlib.pyplot as plt

ds = xr.open_dataset("GNN_training/one_wave/energy/analytical_energy_sem.nc")

energy = ds["analytical_energy_sem"].sel(ensemble_member=53)

plt.figure(figsize=(6,4))
plt.plot(energy.time, energy.values)
plt.xlabel("Time")
plt.ylabel("Energy")
plt.tight_layout()
plt.savefig("true_energy.png", dpi=300)


