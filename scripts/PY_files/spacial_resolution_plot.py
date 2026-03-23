import numpy as np
import matplotlib.pyplot as plt

# Grid spacing
dx =  0.04140430782935302#0.010361252408621272#0.0175  # radians

# Gaussian widths
sigma6 = np.deg2rad(15.0)
sigma10 = np.deg2rad(20.0)

# Angular distance axis
alpha = np.linspace(-0.5, 0.5, 800)

# Gaussian profiles
g6 = np.exp(-(alpha**2) / (2 * sigma6**2))
g10 = np.exp(-(alpha**2) / (2 * sigma10**2))

# Plot
plt.figure(figsize=(8, 5))
plt.plot(alpha, g6, label=r"$\sigma = 6^\circ$")
plt.plot(alpha, g10, label=r"$\sigma = 10^\circ$")

# Draw vertical lines for the grid spacing
for k in range(-25, 25):
    plt.axvline(k * dx, linestyle="--", linewidth=0.8)

plt.xlabel("Angular distance (radians)")
plt.ylabel("Amplitude")
plt.title(f"Inspection of spatial resolution with dx={dx}")
plt.legend()
plt.tight_layout()
plt.show()