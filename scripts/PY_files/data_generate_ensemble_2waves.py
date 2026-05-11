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
    dt = T_min / 20

    print("dt compute", dt)

    N_members = 28
    tmax = dt * 600
    print("tmax", tmax)

    rng = np.random.default_rng(42)

    def sample_center_on_sphere(R=1.0, rng=None):
        v = rng.normal(size=3)
        v = v / np.linalg.norm(v)
        return R * v

    def angular_distance(c1, c2, R=1.0):
        cos_angle = np.dot(c1, c2) / (R**2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.arccos(cos_angle)

    def sample_two_centers_min_distance(R=1.0, rng=None, min_angle_deg=30.0):
        min_angle = np.deg2rad(min_angle_deg)
        while True:
            c1 = sample_center_on_sphere(R=R, rng=rng)
            c2 = sample_center_on_sphere(R=R, rng=rng)
            if angular_distance(c1, c2, R=R) >= min_angle:
                return c1, c2

    def make_f_handle_two_waves(center1, sigma1, A1,
                                center2, sigma2, A2,
                                R=1.0):
        x1, y1, z1 = center1
        x2, y2, z2 = center2

        def f_handle(x, y, z):
            # Wave 1
            dot1 = x * x1 + y * y1 + z * z1
            cos_alpha1 = np.clip(dot1 / R**2, -1.0, 1.0)
            alpha1 = np.arccos(cos_alpha1)
            wave1 = A1 * np.exp(-(alpha1**2) / (2 * sigma1**2))

            # Wave 2
            dot2 = x * x2 + y * y2 + z * z2
            cos_alpha2 = np.clip(dot2 / R**2, -1.0, 1.0)
            alpha2 = np.arccos(cos_alpha2)
            wave2 = A2 * np.exp(-(alpha2**2) / (2 * sigma2**2))

            return wave1 + wave2

        return f_handle

    def make_g_handle():
        return lambda x, y, z: 0 * x

    def draw_unique_two_wave_parameters(
        N_members,
        R,
        rng,
        sigma_range=(15.0, 20.0),
        A_range=(1.0, 2.0),
        min_angle_deg=30.0,
        decimals=10,
    ):
        seen = set()

        centers1 = []
        centers2 = []
        sigmas1 = []
        sigmas2 = []
        amplitudes1 = []
        amplitudes2 = []

        while len(centers1) < N_members:
            center1, center2 = sample_two_centers_min_distance(
                R=R,
                rng=rng,
                min_angle_deg=min_angle_deg,
            )

            sigma1_deg = rng.uniform(*sigma_range)
            sigma2_deg = rng.uniform(*sigma_range)
            A1 = rng.uniform(*A_range)
            A2 = rng.uniform(*A_range)

            key = (
                tuple(np.round(center1, decimals=decimals)),
                tuple(np.round(center2, decimals=decimals)),
                round(sigma1_deg, decimals),
                round(sigma2_deg, decimals),
                round(A1, decimals),
                round(A2, decimals),
            )

            if key in seen:
                continue

            seen.add(key)

            centers1.append(center1)
            centers2.append(center2)
            sigmas1.append(np.deg2rad(sigma1_deg))
            sigmas2.append(np.deg2rad(sigma2_deg))
            amplitudes1.append(A1)
            amplitudes2.append(A2)

        return (
            np.asarray(centers1, dtype=np.float64),
            np.asarray(centers2, dtype=np.float64),
            np.asarray(sigmas1, dtype=np.float64),
            np.asarray(sigmas2, dtype=np.float64),
            np.asarray(amplitudes1, dtype=np.float64),
            np.asarray(amplitudes2, dtype=np.float64),
        )

    # Draw parameters for 2-wave initial conditions
    centers1, centers2, sigmas1, sigmas2, amplitudes1, amplitudes2 = draw_unique_two_wave_parameters(
        N_members=N_members,
        R=R,
        rng=rng,
        sigma_range=(15.0, 20.0),
        A_range=(1.0, 2.0),
        min_angle_deg=30.0,
        decimals=10,
    )

    fg_list = []
    for center1, center2, sigma1, sigma2, A1, A2 in zip(
        centers1, centers2, sigmas1, sigmas2, amplitudes1, amplitudes2
    ):
        f_i = make_f_handle_two_waves(
            center1=center1,
            sigma1=sigma1,
            A1=A1,
            center2=center2,
            sigma2=sigma2,
            A2=A2,
            R=R,
        )
        g_i = make_g_handle()
        fg_list.append((f_i, g_i))

    t_start = time.perf_counter()
    sim = simu.SimulatorWaveEquation(
        R=R,
        C=C,
        Lmax=Lmax,
        tmax=tmax,
        f_handle=lambda x, y, z: 0 * x,   # not used by simulate_ensemble
        g_handle=lambda x, y, z: 0 * x,   # not used by simulate_ensemble
        generations=generations,
        dt=dt,
    )
    print("time class creator:", time.perf_counter() - t_start)

    print(f"N = {sim.N}")
    print(f"dt = {dt}")
    print(f"dx = {sim.dx_true}")
    print(f"tmax = {tmax}")
    print(f"cfl: {sim.cfl_value}")

    t_start = time.perf_counter()
    ds, u = sim.simulate_ensemble(
        fg_list,
        title="wave2_ensemble_28_coarse_600_timesteps_sub4",
        savedata=False,
    )
    print("time simulation: ", time.perf_counter() - t_start)

    t_start = time.perf_counter()
    sim.save_data(ds, title="wave2_ensemble_28_waves_600_timesteps_sub4")
    print("time save", time.perf_counter() - t_start)


if __name__ == "__main__":
    # profiler = cProfile.Profile()
    # profiler.enable()

    main()

    # profiler.disable()
    # stats = pstats.Stats(profiler)
    # stats.strip_dirs().sort_stats("cumtime").print_stats(30)
    # print("\nOnly my file:\n")
    # stats.print_stats("PY_09_03_26_data_generate_ensemble")