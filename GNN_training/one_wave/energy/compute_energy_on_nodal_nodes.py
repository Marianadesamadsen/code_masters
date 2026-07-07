import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_generation_functions import wave_sphere_exact_split as exact
from integrate_sphere.compute_energy import (
    surface_mass_integration,
    compute_ut_and_mid,
    compute_wave_energy_batch,
)


def make_gaussian_initial_condition(center, sigma, A, R=1.0):
    center = np.asarray(center, dtype=np.float64)
    center = center / np.linalg.norm(center)

    def f(x, y, z):
        pts = np.stack([x, y, z], axis=-1)
        pts = pts / np.linalg.norm(pts, axis=-1, keepdims=True)

        cos_alpha = np.clip(np.sum(pts * center, axis=-1), -1, 1)
        alpha = np.arccos(cos_alpha)

        return A * np.exp(-(alpha**2) / (2 * sigma**2))

    def g(x, y, z):
        return np.zeros_like(x)

    return f, g


def slice_eval_data(eval_data, start, end, n_points):
    eval_data_batch = {}

    for key, value in eval_data.items():
        value = np.asarray(value)

        if value.ndim > 0 and value.shape[0] == n_points:
            eval_data_batch[key] = value[start:end]
        else:
            eval_data_batch[key] = value

    return eval_data_batch


def construct_all_times_reuse_basis(
    flm,
    glm,
    is_real,
    eval_data,
    times,
    Lmax,
    C,
    R,
    batch_size=5000,
):

    n_points = len(next(iter(eval_data.values())))
    n_times = len(times)

    u_all = np.empty((n_times, n_points), dtype=np.float64)

    for start in range(0, n_points, batch_size):
        end = min(start + batch_size, n_points)

        eval_data_batch = slice_eval_data(
            eval_data,
            start=start,
            end=end,
            n_points=n_points,
        )

        Y_basis_batch = np.ascontiguousarray(
            np.hstack(exact.precompute_Ylm_basis(eval_data_batch, Lmax))
        )

        for j, t in enumerate(times):
            ulm = exact.evolve_modal_coefficients(
                flm,
                glm,
                float(t),
                Lmax,
                C,
                R,
            )

            u_batch = exact.synthesize_solution(
                ulm,
                is_real,
                Y_basis_batch,
            )

            if np.iscomplexobj(u_batch):
                u_batch = np.real(u_batch)

            u_all[j, start:end] = u_batch

    return u_all


def compute_sem_analytical_energy_from_nc(
    nc_path,
    output_path=None,
    N_sem=6,
    mesh_generation=4,
    ut_order=4,
    member_start=50,
    member_end=99,
    batch_size=5000,
    Lmax_override=None,
):
    ds = xr.open_dataset(nc_path)

    ds = ds.sel(ensemble_member=slice(member_start, member_end))#,time=slice(0.0,1.5))

    R = float(ds.attrs["R"])
    C = float(ds.attrs["C"])

    if Lmax_override is None:
        Lmax = int(ds.attrs["Lmax"])
    else:
        Lmax = int(Lmax_override)

    dt = float(ds.attrs["dt"])

    times = ds["time"].values.astype(float)
    members = ds["ensemble_member"].values

    out = surface_mass_integration(
        N=N_sem,
        generation=mesh_generation,
        R=R,
    )

    x_sem = out["x3D"]
    y_sem = out["y3D"]
    z_sem = out["z3D"]

    Np, K = x_sem.shape

    sem_xyz = np.stack(
        [x_sem.ravel(), y_sem.ravel(), z_sem.ravel()],
        axis=1,
    )

    quad = exact.setup_quadrature(Lmax, R)
    eval_data = exact.prepare_evaluation_points(sem_xyz, Lmax, R)

    E_members = []

    for i, member in enumerate(members):

        center = ds["center"].sel(ensemble_member=member).values
        sigma = float(ds["sigma"].sel(ensemble_member=member).values)
        A = float(ds["A"].sel(ensemble_member=member).values)

        f_handle, g_handle = make_gaussian_initial_condition(
            center=center,
            sigma=sigma,
            A=A,
            R=R,
        )

        fq, gq = exact.sample_initial_data_on_quadrature(
            quad,
            f_handle,
            g_handle,
        )

        flm, glm, _ = exact.compute_modal_coefficients(
            fq,
            gq,
            quad,
            Lmax,
        )

        is_real = np.isrealobj(fq) and np.isrealobj(gq)

        u_all_flat = construct_all_times_reuse_basis(
            flm=flm,
            glm=glm,
            is_real=is_real,
            eval_data=eval_data,
            times=times,
            Lmax=Lmax,
            C=C,
            R=R,
            batch_size=batch_size,
        )

        u_sem_time = u_all_flat.reshape(len(times), Np, K)

        ut_sem, u_mid_sem = compute_ut_and_mid(
            u_sem_time,
            dt=dt,
            ut_order=ut_order,
        )

        E_member = compute_wave_energy_batch(
            u_mid_sem,
            ut_sem,
            out=out,
            c=C,
        )

        E_members.append(E_member)

    E = np.stack(E_members, axis=0)

    cut = ut_order // 2
    energy_times = times[cut:-cut]

    ds_energy = xr.Dataset(
        data_vars={
            "analytical_energy_sem": (
                ("ensemble_member", "time"),
                E,
            )
        },
        coords={
            "ensemble_member": members,
            "time": energy_times,
        },
        attrs={
            "source_nc": str(nc_path),
            "R": R,
            "C": C,
            "Lmax": Lmax,
            "dt": dt,
            "N_sem": N_sem,
            "mesh_generation": mesh_generation,
            "ut_order": ut_order,
            "member_start": member_start,
            "member_end": member_end,
            "batch_size": batch_size,
        },
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ds_energy.to_netcdf(output_path)

    return ds_energy


if __name__ == "__main__":
    ds_E = compute_sem_analytical_energy_from_nc(
        nc_path="GNN_training/one_wave/nc_files/wave_200_dtsmall_min10_g4_sigamin6_sigmax12_correctT_plus160dt.nc",
        output_path="GNN_training/one_wave/energy/analytical_energy_sem.nc",
        N_sem=6,
        mesh_generation=4,
        ut_order=4,
        member_start=50,
        member_end=99,
        batch_size=1000,
        Lmax_override=40,  
    )
