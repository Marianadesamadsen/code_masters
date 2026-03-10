import numpy as np
import sys 
sys.path.insert(0, './')
import data_generation.SimulatorWaveEquation as simu

R = 1.0
C = 1.0
Lmax = 20

generations = 6

omega_max = (C / R) * np.sqrt(Lmax * (Lmax + 1))
T_min = 2 * np.pi / omega_max
dt = T_min / 20
N_members = 1
tmax = dt
rng = np.random.default_rng(42)

def sample_center_on_sphere(R=1.0, rng=None):
    v = rng.normal(size=3)
    v = v / np.linalg.norm(v)
    return R * v


def make_f_handle(center, sigma, A, R=1.0):
    x0, y0, z0 = center

    def f_handle(x, y, z):
        dot = x * x0 + y * y0 + z * z0
        cos_alpha = np.clip(dot / R**2, -1.0, 1.0)
        alpha = np.arccos(cos_alpha)
        return A * np.exp(-(alpha**2) / (2 * sigma**2))

    return f_handle


def make_g_handle():
    return lambda x, y, z: 0 * x


fg_list = []
centers = []
sigmas = []
amplitudes = []

for _ in range(N_members):
    center = sample_center_on_sphere(R=R, rng=rng)
    sigma_deg = rng.uniform(10.0, 15.0)
    sigma = np.deg2rad(sigma_deg)
    A = rng.uniform(0.5, 2.0)

    f_i = make_f_handle(center=center, sigma=sigma, A=A, R=R)
    g_i = make_g_handle()

    fg_list.append((f_i, g_i))
    centers.append(center)
    sigmas.append(sigma)
    amplitudes.append(A)

centers = np.asarray(centers, dtype=np.float64)   # (ensemble, 3)
sigmas = np.asarray(sigmas, dtype=np.float64)     # (ensemble,)
amplitudes = np.asarray(amplitudes, dtype=np.float64)  # (ensemble,)


sim = simu.SimulatorWaveEquation(
    R=R,
    C=C,
    Lmax=Lmax,
    tmax=tmax,
    f_handle=lambda x, y, z: 0*x,   # not used by simulate_ensemble
    g_handle=lambda x, y, z: 0*x,   # not used by simulate_ensemble
    generations=generations,
    dt=dt,
)

print(f"dt = {dt}")
print(f"dx = {sim.dx}")
print(f"clf: {sim.clf_value}")

ds = sim.simulate_ensemble(
    fg_list,
    title="wave_ensemble_100",
    savedata=True,
    centers=centers,
    sigmas=sigmas,
    amplitudes=amplitudes,
)

print(ds)
print(ds["u"].shape)       # (100, time, N)
print(ds["center"].shape)  # (100, 3)
print(ds["sigma"].shape)   # (100,)
print(ds["A"].shape)       # (100,)

ds["u"].isel(ensemble=7)
ds["center"].isel(ensemble=7)
ds["sigma"].isel(ensemble=7)
ds["A"].isel(ensemble=7)

