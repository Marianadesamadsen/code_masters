import numpy as np
import sys 
sys.path.insert(0, './')
import data_generation_functions.SimulatorWaveEquation as simu
import time
import data_generation_functions.DataPlotter as dp 
import os 

R = 1.0
C = 1.0
Lmax = 20
A = 1
generations = 3

omega_max = (C / R) * np.sqrt(Lmax * (Lmax + 1))
T_min = 2 * np.pi / omega_max
print("dt:",T_min / 20)
dt = T_min / 20  # 0.010361252408621268/3 # 
print("used dt:",dt)
tmax = dt*1000
print("tmax:",tmax)
rng = np.random.default_rng(42)

def make_g_handle():
    return lambda x, y, z: 0 * x

center = np.array([0, -R, R])
center = R * center / np.linalg.norm(center)
x0, y0, z0 = center 
def f_handle(x, y, z):
    # dot product with center
    dot = x*x0 + y*y0 + z*z0
    
    cos_alpha = np.clip(dot / R**2, -1.0, 1.0)

    alpha = np.arccos(cos_alpha)
    sigma = np.deg2rad(15.0)
    return A*np.exp(-(alpha**2) / (2*sigma**2))

sim = simu.SimulatorWaveEquation(
    R=R,
    C=C,
    Lmax=Lmax,
    tmax=tmax,
    f_handle=f_handle,#lambda x, y, z: 0*x,   # not used by simulate_ensemble
    g_handle=lambda x, y, z: 0*x,   # not used by simulate_ensemble
    generations=generations,
    dt=dt,
)

# ds = sim.simulate(savedata=False)
# from matplotlib import cm, colors
# u_min = float(np.nanmin(ds["u"].values))
# u_max = float(np.nanmax(ds["u"].values))
# norm = colors.Normalize(vmin=u_min, vmax=u_max)
# plot_true = dp.DataPlotter(ds=ds)
# anim_true = plot_true.animate_sphere(norm=norm,out_path="coarse_data_test.gif", fps=15)
# plt.close()

print("dx elem",sim.dx_elem)
print("dx:",sim.dx)
print("dx Allan:",sim.dx_allan)

def degree_spectrum(ulm):
    Lmax = ulm.shape[0] - 1
    spec = np.zeros(Lmax + 1)
    for ell in range(Lmax + 1):
        # valid m for this ell are -ell..ell, which correspond to columns (Lmax-ell):(Lmax+ell+1)
        col0 = Lmax - ell
        col1 = Lmax + ell+1 
        spec[ell] = np.sum(np.abs(ulm[ell, col0:col1])**2)
    return spec

coeffs = sim.get_spectral_coeffs(t=0.0)
ulm = coeffs["ulm"]

E = degree_spectrum(ulm)
import matplotlib.pyplot as plt
plt.semilogy(E,'-o')   # avoid log(0)
plt.xlabel("$\ell$")
plt.ylabel("$\sum_m |u_{\ell m}|^2$") 

cum = np.cumsum(E) / np.sum(E)
plt.figure()
plt.plot(cum, '-o')
plt.xlabel(r'$\ell$')
plt.ylabel('cumulative energy')
plt.grid(True)
plt.show()
