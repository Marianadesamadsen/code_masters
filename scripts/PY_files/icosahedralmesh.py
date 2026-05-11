import sys
sys.path.insert(0, './')

import data_generation_functions.SimulatorWaveEquation as simu
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

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

fig = plt.figure(figsize=(14, 4))

for i, (P, tri) in enumerate(zip(Ps, Tris)):
    ax = fig.add_subplot(1, 4, i + 1, projection="3d")

    x, y, z = P[0], P[1], P[2]

    ax.plot_trisurf(
        x, y, z,
        triangles=tri,
        edgecolor="blue",
        linewidth=0.5,
        color="lightblue",
        alpha=0.8
    )

    ax.set_title(f"Subdivision {generations[i]}")
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim([-R, R])
    ax.set_ylim([-R, R])
    ax.set_zlim([-R, R])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

plt.suptitle("Icosahedral mesh subdivisions", fontsize=16)
plt.tight_layout(pad=2.0)
plt.savefig("figures/icosahedral_mesh_subdivisions.png")
