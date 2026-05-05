import cProfile
import pstats
import numpy as np
import sys 
sys.path.insert(0, './')
import data_generation_functions.SimulatorWaveEquation as simu
import time

def main():

    R = 1.0 
    C = 1
    Lmax = 25
    generations = 4
    omega_max = (C / R) * np.sqrt(Lmax * (Lmax + 1))
    T_min = 2 * np.pi / omega_max
    print("dt compute",T_min / 20 )
    dt = T_min / 20 # 0.010361252408621261/3 # 
    N_members = 140
    tmax = dt*600
    print("tmax",tmax) 
    title = "wave_140_ts_600_g4_sigmamin_15"
    nc_path = r"GNN_training\one_wave\nc_files"

    sigma_range=(15.0, 20.0)
    A_range=(1.0, 2.0)
    
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

    def draw_unique_parameters(N_members, R, rng, sigma_range=(15.0, 20.0), A_range=(0.5, 2.0), decimals=10):
        seen = set()
        centers = []
        sigmas = []
        amplitudes = []

        while len(centers) < N_members:
            center = sample_center_on_sphere(R=R, rng=rng)
            sigma_deg = rng.uniform(*sigma_range)
            A = rng.uniform(*A_range)

            key = (
                tuple(np.round(center, decimals=decimals)),
                round(sigma_deg, decimals),
                round(A, decimals),
            )

            if key in seen:
                continue

            seen.add(key)
            centers.append(center)
            sigmas.append(np.deg2rad(sigma_deg))
            amplitudes.append(A)

        return (
            np.asarray(centers, dtype=np.float64),
            np.asarray(sigmas, dtype=np.float64),
            np.asarray(amplitudes, dtype=np.float64),
        )  

    centers, sigmas, amplitudes = draw_unique_parameters(
        N_members=N_members,
        R=R,
        rng=rng,
        sigma_range=sigma_range,
        A_range=A_range,
        decimals=10,
    )

    fg_list = []
    for center, sigma, A in zip(centers, sigmas, amplitudes):
        f_i = make_f_handle(center=center, sigma=sigma, A=A, R=R)
        g_i = make_g_handle()
        fg_list.append((f_i, g_i))  

    t_start = time.perf_counter()
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
    print("time class creator:",time.perf_counter()-t_start)

    print(f"N = {sim.N}")
    print(f"dt = {dt}")
    print(f"dx = {sim.dx_true}")
    print(f"tmax = {tmax}")
    print(f"cfl: {sim.cfl_value}")

    t_start = time.perf_counter()
    ds,u = sim.simulate_ensemble(
        fg_list,
        title=title,
        savedata=False,
        centers=centers,
        sigmas=sigmas,
        amplitudes=amplitudes, 
    )
    print("time simulation: ", time.perf_counter() - t_start)

    t_start = time.perf_counter()
    sim.save_data(ds,nc_path =nc_path, title=title)
    print("time save",time.perf_counter()-t_start)

if __name__ == "__main__":
    #profiler = cProfile.Profile()
    #profiler.enable()

    main()

    #profiler.disable() 
    #stats = pstats.Stats(profiler)
    #stats.strip_dirs().sort_stats("cumtime").print_stats(30)
    #print("\nOnly my file:\n")
    #stats.print_stats("PY_09_03_26_data_generate_ensemble")



