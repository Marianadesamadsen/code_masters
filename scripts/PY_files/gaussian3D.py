import numpy as np
import matplotlib.pyplot as plt
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


rng = np.random.default_rng(42)
N = 200
v = rng.normal(size=(N, 3))
norms = np.linalg.norm(v, axis=1, keepdims=True)
v_normalized = v / norms


mesh = trimesh.creation.icosphere(
    subdivisions=3,
    radius=1.0,
)

vertices = mesh.vertices
faces = mesh.faces


fig = plt.figure(figsize=(14, 6))
ax1 = fig.add_subplot(121, projection="3d")

ax1.scatter(
    v[:, 0],
    v[:, 1],
    v[:, 2],
    s=10,
    color="red",
    alpha=0.9,
)

ax1.set_title("3D Gaussian samples for the centers")

ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("z")

max_range = np.max(np.abs(v))

ax1.set_xlim(-max_range, max_range)
ax1.set_ylim(-max_range, max_range)
ax1.set_zlim(-max_range, max_range)


ax2 = fig.add_subplot(122, projection="3d")
triangles = vertices[faces]
mesh_collection = Poly3DCollection(
    triangles,
    alpha=0.25,
    edgecolor="white",
    linewidth=0.001,
)

# Light blue color
mesh_collection.set_facecolor("cornflowerblue")

ax2.add_collection3d(mesh_collection)

ax2.scatter(
    v_normalized[:, 0],
    v_normalized[:, 1],
    v_normalized[:, 2],
    s=10,
    color="red",
    alpha=0.9,
)
ax2.set_title("Projected centers onto sphere")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("z")

ax2.set_xlim(-1, 1)
ax2.set_ylim(-1, 1)
ax2.set_zlim(-1, 1)

# Equal aspect ratio
ax2.set_box_aspect([1, 1, 1])

plt.tight_layout()

plt.savefig("Gaussian3d_with_colored_icosahedral_mesh.png",)

plt.show()