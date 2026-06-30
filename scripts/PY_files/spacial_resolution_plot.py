import numpy as np
import matplotlib.pyplot as plt

# Grid spacing
dx =  0.08#134019969865452  # radians
dt =  0.0155

# Gaussian widths
sigma6 = np.deg2rad(6) #8
sigma10 = np.deg2rad(12.0) #30

# Angular distance axis
alpha = np.linspace(-0.5, 0.5, 800)

# Gaussian profiles
g6 = np.exp(-(alpha**2) / (2 * sigma6**2))
g10 = np.exp(-(alpha**2) / (2 * sigma10**2))

# Plot
fig, axes = plt.subplots(1,2,figsize=(14, 5))

for i,dx in enumerate([0.08,0.02]):
    ax = axes[i]
    ax.plot(alpha, g6, label=r"$\sigma_{\min} = 6^\circ$")
    ax.plot(alpha, g10, label=r"$\sigma_{\max} = 12^\circ$")
    for k in range(-25, 25):
        ax.axvline(k * dx, linestyle="--", linewidth=0.8)

    ax.set_xlabel("Angular distance (radians)")
    axes[0].set_ylabel("Amplitude")
    ax.set_xlim(-0.5,0.5)
    ax.set_title(fr"$dx\approx{np.round(dx,3)}$")
    ax.legend()
plt.suptitle("Inspection of spatial resolution with different grid spacings", fontsize=16)
plt.savefig("spatial_resolution_plot_both.png")

plt.close()
plt.figure(figsize=(8, 5))
plt.plot(alpha, g6, label=r"$\sigma_{\min} = 6^\circ$")
plt.plot(alpha, g10, label=r"$\sigma_{\max} = 12^\circ$")

# Draw vertical lines for the grid spacing
for k in range(-25, 25):
    plt.axvline(k * dx, linestyle="--", linewidth=0.8)
plt.xlim(-0.5,0.5)

plt.xlabel("Angular distance (radians)")
plt.ylabel("Amplitude")
plt.title(fr"Inspection of spatial resolution with $dx\approx{np.round(dx,3)}$")
plt.legend()
plt.tight_layout()
plt.savefig("spatial_resolution_plot.png")
plt.close()

###################################################

dt = 0.0155
alpha = np.linspace(-np.pi, np.pi, 800)

fig, axes = plt.subplots(1, 5, figsize=(30, 8))
for idx,dt_scale in enumerate([1,10,20,40,80]):

    ax = axes[idx]
    
    c = 1.0
    def gaussian(x, sigma):
        return np.exp(-(x**2) / (2 * sigma**2))

    u0 = gaussian(alpha, sigma6)

    u1 = 0.5 * (
        gaussian(alpha - c*dt*dt_scale, sigma6)
        + gaussian(alpha + c*dt*dt_scale, sigma6)
    )

    u2 = 0.5 * (
        gaussian(alpha - c*2*dt*dt_scale, sigma6)
        + gaussian(alpha + c*2*dt*dt_scale, sigma6)
    )

    ax.plot(alpha, u0, label=r"$t=0$")
    ax.plot(alpha, u1, label=r"$t=\Delta t$")
    ax.plot(alpha, u2, label=r"$t=2\Delta t$")
    # for k in range(-25, 25):
    #     ax.axvline(k * dx, linestyle="--", linewidth=0.8)
    ax.set_xlim(-np.pi,np.pi)
    ax.set_xlabel("Angular distance (radians)",fontsize=18)
    ax.set_title(r"$\Delta t$="+f"{dt_scale}dt={dt*dt_scale}",fontsize=20)
    ax.set_ylabel("Amplitude",fontsize=18)
    ax.grid(True)
    ax.legend(fontsize=14)

plt.tight_layout(pad=4)
plt.suptitle("Wave propgation for different dt",fontsize=22, y=0.998)
plt.savefig("wave_propagation_in_time_10dt.png")

#####################################################

# Draw vertical lines for the grid spacing
for k in range(-25, 25):
    plt.axvline(k * dx, linestyle="--", linewidth=0.8)
plt.xlim(-0.5,0.5)

plt.xlabel("Angular distance (radians)")
plt.ylabel("Amplitude")
plt.title(fr"Inspection of spatial resolution with $dx\approx{np.round(dx,3)}$")
plt.legend()
plt.tight_layout()
plt.savefig("spatial_resolution_plot_time.png")


# Time axis
t = np.linspace(0, 10, 1000)

# Wave speed
c = 1.0

plt.figure(figsize=(8,6))

plt.plot( + c*t, t, linestyle="--", linewidth=0.8)
plt.plot( - c*t, t, linestyle="--", linewidth=0.8)
for k in range(-25, 26):
    x0 = k * dx
    plt.axvline(x0, color="blue", linestyle=":", linewidth=0.5)
    # Characteristic lines

for n in range(0, 81):
    plt.axhline(n * dt, color="red", linestyle=":", linewidth=0.5)

plt.xlabel("x")
plt.ylabel("t")
plt.title("Characteristic lines")

plt.xlim(-0.2, 0.2)
plt.ylim(0, 0.2)

plt.tight_layout()
plt.savefig("DtdX.png")
plt.show()