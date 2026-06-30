import sys
sys.path.insert(0, './')

import matplotlib.pyplot as plt
import data_generation_functions.SimulatorWaveEquation as simu


R = 1
C = 1
Lmax = 20
tmax = 1

coarse_gen = 3
fine_gen = 4

sim_coarse = simu.SimulatorWaveEquation(
    R=R,
    C=C,
    Lmax=Lmax,
    tmax=tmax,
    f_handle=lambda x, y, z: 0*x,
    g_handle=lambda x, y, z: 0*x,
    generations=coarse_gen,
)

sim_fine = simu.SimulatorWaveEquation(
    R=R,
    C=C,
    Lmax=Lmax,
    tmax=tmax,
    f_handle=lambda x, y, z: 0*x,
    g_handle=lambda x, y, z: 0*x,
    generations=fine_gen,
)

P_coarse = sim_coarse.P
tri_coarse = sim_coarse.tri

P_fine = sim_fine.P
tri_fine = sim_fine.tri

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Fine mesh
ax.plot_trisurf(
    P_fine[0],
    P_fine[1],
    P_fine[2],
    triangles=tri_fine,
    #color="lightblue",
    edgecolor="gray",
    linewidth=0.2,
    alpha=0.15,
)

# Coarse mesh on top
ax.plot_trisurf(
    P_coarse[0],
    P_coarse[1],
    P_coarse[2],
    triangles=tri_coarse,
    color="lightblue",
    edgecolor="red",
    linewidth=1.5,
    alpha=0.5,
)

ax.set_title(
    f"Icosahedral mesh overlay\n"
    f"coarse = subdivision {coarse_gen}, "
    f"fine = subdivision {fine_gen}"
)

ax.set_box_aspect([1, 1, 1])

ax.set_xlim([-R, R])
ax.set_ylim([-R, R])
ax.set_zlim([-R, R])

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

plt.tight_layout()

plt.savefig(f"figures/mesh_overlay_gen{coarse_gen}_gen{fine_gen}.png")


