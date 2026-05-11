import sys
sys.path.insert(0, './')

import numpy as np
import matplotlib.pyplot as plt
import data_generation_functions.SimulatorWaveEquation as simu

R = 1
C = 1
Lmax = 20
tmax = 1
generations = [0, 1, 2, 3]

Ps = []
Tris = []

for i in generations:
    sim = simu.SimulatorWaveEquation(
        R=R,
        C=C,
        Lmax=Lmax,
        tmax=tmax,
        f_handle=lambda x, y, z: 0 * x,
        g_handle=lambda x, y, z: 0 * x,
        generations=i,
    )
    Ps.append(sim.P)
    Tris.append(sim.tri)

fig, axs = plt.subplots(1, 4, figsize=(16, 4))

for i, (P, tri) in enumerate(zip(Ps, Tris)):
    x, y, z = P[0], P[1], P[2]

    # Convert to lon/lat
    lon = np.arctan2(y, x)
    lat = np.arcsin(np.clip(z / R, -1.0, 1.0))

    # Convert radians to degrees
    lon_deg = np.degrees(lon)
    lat_deg = np.degrees(lat)

    ax = axs[i]
    ax.triplot(lon_deg, lat_deg, tri, color="blue", linewidth=0.6)
    ax.set_title(f"Subdivision {generations[i]}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim([-180, 180])
    ax.set_ylim([-90, 90])
    ax.set_aspect("equal")

plt.tight_layout()
plt.savefig("figures/icosahedral_mesh_subdivisions_lon_lat.png")