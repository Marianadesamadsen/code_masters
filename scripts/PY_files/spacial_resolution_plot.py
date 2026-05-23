import numpy as np
import matplotlib.pyplot as plt

# Grid spacing
dx =  0.020718449365429062  # radians
dt = 0.01232234018801084

# Gaussian widths
sigma6 = np.deg2rad(4) #8
sigma10 = np.deg2rad(30.0) #30

# Angular distance axis
alpha = np.linspace(-0.5, 0.5, 800)

# Gaussian profiles
g6 = np.exp(-(alpha**2) / (2 * sigma6**2))
g10 = np.exp(-(alpha**2) / (2 * sigma10**2))

# Plot
plt.figure(figsize=(8, 5))
plt.plot(alpha, g6, label=r"$\sigma = 8^\circ$")
plt.plot(alpha, g10, label=r"$\sigma = 30^\circ$")

# Draw vertical lines for the grid spacing
for k in range(-25, 25):
    plt.axvline(k * dx, linestyle="--", linewidth=0.8)

plt.xlabel("Angular distance (radians)")
plt.ylabel("Amplitude")
plt.title(f"Inspection of spatial resolution with dx={dx}")
plt.legend()
plt.tight_layout()
plt.savefig("spatial_resolution_plot.png")


# Time axis
t = np.linspace(0, 10, 1000)

# Wave speed
c = 1.0

plt.figure(figsize=(8,6))

# Initial grid points
plt.plot( + c*t, t, linestyle="--", linewidth=0.8)
plt.plot( - c*t, t, linestyle="--", linewidth=0.8)
for k in range(-25, 26):
    x0 = k * dx
    plt.axvline(x0, color="blue", linestyle=":", linewidth=0.5)
    # Characteristic lines

# Horizontal dt lines
for n in range(0, 81):
    plt.axhline(n * dt, color="red", linestyle=":", linewidth=0.5)

# Labels
plt.xlabel("x")
plt.ylabel("t")
plt.title("Characteristic lines")

plt.xlim(-0.2, 0.2)
plt.ylim(0, 0.2)

plt.tight_layout()
plt.savefig("DtdX.png")
plt.show()