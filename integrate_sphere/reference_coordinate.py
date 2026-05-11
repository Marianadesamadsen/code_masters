import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create reference triangle grid
def reference_triangle(N=20):
    r_list = []
    s_list = []

    for j in range(N + 1):
        for i in range(N + 1 - j):
            r = -1 + 2 * i / N
            s = -1 + 2 * j / N
            r_list.append(r)
            s_list.append(s)

    return np.array(r_list), np.array(s_list)


# Pick one triangle on sphere
def get_triangle_vertices():
    # Simple triangle on unit sphere
    v1 = np.array([0, 0, 1])
    v2 = np.array([1, 0, 0])
    v3 = np.array([0, 1, 0])
    return v1, v2, v3


# Mapping from (r,s) -> (x,y,z)
def map_to_sphere(r, s, v1, v2, v3):
    x = 0.5 * (-(r + s) * v1[0] + (1 + r) * v2[0] + (1 + s) * v3[0])
    y = 0.5 * (-(r + s) * v1[1] + (1 + r) * v2[1] + (1 + s) * v3[1])
    z = 0.5 * (-(r + s) * v1[2] + (1 + r) * v2[2] + (1 + s) * v3[2])

    # project to sphere
    norm = np.sqrt(x**2 + y**2 + z**2)
    return x / norm, y / norm, z / norm


# Generate data
r, s = reference_triangle(N=25)
v1, v2, v3 = get_triangle_vertices()

x, y, z = map_to_sphere(r, s, v1, v2, v3)

# --- Plot ---
fig = plt.figure(figsize=(12, 5))

# Reference triangle
ax1 = fig.add_subplot(121)
ax1.scatter(r, s, s=5)
ax1.set_title("Reference triangle (r,s)")
ax1.set_xlabel("r")
ax1.set_ylabel("s")
ax1.set_aspect("equal")

# Sphere triangle
ax2 = fig.add_subplot(122, projection="3d")
ax2.scatter(x, y, z, s=5)

# draw sphere wireframe
u = np.linspace(0, 2*np.pi, 50)
v = np.linspace(0, np.pi, 25)
xs = np.outer(np.cos(u), np.sin(v))
ys = np.outer(np.sin(u), np.sin(v))
zs = np.outer(np.ones_like(u), np.cos(v))
ax2.plot_wireframe(xs, ys, zs, color='lightgray', alpha=0.3)

ax2.set_title("Mapped triangle on sphere (x,y,z)")

plt.savefig("reference_coordinate.png")

