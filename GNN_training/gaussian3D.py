import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Generate random 3D Gaussian vectors
# -------------------------------------------------
rng = np.random.default_rng(42)

N = 1000

# Sample from 3D standard normal
v = rng.normal(size=(N, 3))

# -------------------------------------------------
# Normalize each vector to unit length
# -------------------------------------------------
norms = np.linalg.norm(v, axis=1, keepdims=True)

v_normalized = v / norms

# -------------------------------------------------
# Plot
# -------------------------------------------------
fig = plt.figure(figsize=(14, 6))

# -----------------------------
# Original Gaussian cloud
# -----------------------------
ax1 = fig.add_subplot(121, projection="3d")

ax1.scatter(
    v[:, 0],
    v[:, 1],
    v[:, 2],
    s=2,
    alpha=0.3,
)

ax1.set_title("3D Gaussian samples")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("z")

# Make axes equal
max_range = np.max(np.abs(v))
ax1.set_xlim(-max_range, max_range)
ax1.set_ylim(-max_range, max_range)
ax1.set_zlim(-max_range, max_range)

# -----------------------------
# Normalized vectors
# -----------------------------
ax2 = fig.add_subplot(122, projection="3d")

ax2.scatter(
    v_normalized[:, 0],
    v_normalized[:, 1],
    v_normalized[:, 2],
    s=2,
    alpha=0.5,
)

ax2.set_title("Normalized 3D Gaussian samples")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("z")

# Unit sphere limits
ax2.set_xlim(-1, 1)
ax2.set_ylim(-1, 1)
ax2.set_zlim(-1, 1)

plt.tight_layout()
plt.show()