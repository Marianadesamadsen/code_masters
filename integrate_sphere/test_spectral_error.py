import numpy as np
import matplotlib.pyplot as plt
from math import pi
from scipy.special import lpmv

from driver import surface_mass_integration


def cart2sph_angles(x, y, z):
    r = np.sqrt(x*x + y*y + z*z)
    theta = np.arccos(np.clip(z / r, -1.0, 1.0))
    phi = np.arctan2(y, x)
    phi = np.where(phi < 0, phi + 2*pi, phi)
    return theta, phi


def factratio(a, b):
    if a == b:
        return 1.0
    prod = 1.0
    for k in range(a + 1, b + 1):
        prod *= k
    return 1.0 / prod


def sph_harm_norm(ell, m):
    m = abs(m)
    return np.sqrt((2*ell + 1) / (4*pi) * factratio(ell - m, ell + m))


def Ylm(ell, m, theta, phi):
    if m >= 0:
        P = ((-1)**m) * lpmv(m, ell, np.cos(theta))
        return sph_harm_norm(ell, m) * P * np.exp(1j * m * phi)
    else:
        mp = -m
        Y_pos = Ylm(ell, mp, theta, phi)
        return ((-1)**mp) * np.conj(Y_pos)


def dg_integration_weights(out):
    MassMatrix = out["MassMatrix"]
    J2D = out["J2D"]

    Np, K = J2D.shape
    weights = np.zeros((Np, K))
    ones = np.ones(Np)

    for k in range(K):
        Mk = np.diag(J2D[:, k]) @ MassMatrix
        weights[:, k] = Mk @ ones

    return weights


def spherical_harmonic_coefficients(u, out, Lmax):
    x = out["x3D"].reshape(-1)
    y = out["y3D"].reshape(-1)
    z = out["z3D"].reshape(-1)

    u_flat = u.reshape(-1)
    w_flat = dg_integration_weights(out).reshape(-1)

    theta, phi = cart2sph_angles(x, y, z)

    coeffs = np.zeros((Lmax + 1, 2*Lmax + 1), dtype=complex)

    for ell in range(Lmax + 1):
        for m in range(-ell, ell + 1):
            Y = Ylm(ell, m, theta, phi)
            coeffs[ell, m + Lmax] = np.sum(w_flat * u_flat * np.conj(Y))

    return coeffs


def spectrum_vs_ell(coeffs, Lmax):
    A = np.zeros(Lmax + 1)

    for ell in range(Lmax + 1):
        for m in range(-ell, ell + 1):
            A[ell] += np.abs(coeffs[ell, m + Lmax])**2

    return A


def spectral_error_vs_ell(u_pred, u_true, out, Lmax):
    coeff_pred = spherical_harmonic_coefficients(u_pred, out, Lmax)
    coeff_true = spherical_harmonic_coefficients(u_true, out, Lmax)

    err = np.zeros(Lmax + 1)

    for ell in range(Lmax + 1):
        for m in range(-ell, ell + 1):
            diff = coeff_pred[ell, m + Lmax] - coeff_true[ell, m + Lmax]
            err[ell] += np.abs(diff)**2

    return err, coeff_pred, coeff_true


def make_example_fields(out):
    """
    Create artificial u_true and u_pred.

    u_true contains:
        low-frequency mode ell=2
        high-frequency mode ell=8

    u_pred captures the ell=2 mode correctly,
    but underestimates the ell=8 mode.

    Therefore spectral error should peak around ell=8.
    """

    x = out["x3D"].reshape(-1)
    y = out["y3D"].reshape(-1)
    z = out["z3D"].reshape(-1)

    theta, phi = cart2sph_angles(x, y, z)

    low_mode = Ylm(2, 1, theta, phi).real
    high_mode = Ylm(8, 3, theta, phi).real

    u_true_flat = low_mode + 0.5 * high_mode
    u_pred_flat = low_mode + 0.5 * high_mode

    u_true = u_true_flat.reshape(out["x3D"].shape)
    u_pred = u_pred_flat.reshape(out["x3D"].shape)

    return u_pred, u_true


if __name__ == "__main__":
    out = surface_mass_integration(N=6, generation=3, R=1.0)

    u_pred, u_true = make_example_fields(out)

    Lmax = 12   
    coeff_true = spherical_harmonic_coefficients(u_true, out, Lmax)

    for ell in range(Lmax+1):
        for m in range(-ell, ell+1):
            val = coeff_true[ell, m + Lmax]
            if abs(val) > 1e-3:
                print(f"(ell={ell}, m={m}) → {val}")

    err_l, coeff_pred, coeff_true = spectral_error_vs_ell(
        u_pred=u_pred,
        u_true=u_true,
        out=out,
        Lmax=Lmax,
    )

    A_pred = spectrum_vs_ell(coeff_pred, Lmax)
    A_true = spectrum_vs_ell(coeff_true, Lmax)

    ells = np.arange(Lmax + 1)

    print("Spectral error by ell:")
    for ell, val in zip(ells, err_l):
        print(f"ell={ell:2d}, error={val:.6e}")

    plt.figure()
    plt.semilogy(ells, A_true + 1e-16, marker="o", label="true spectrum")
    plt.semilogy(ells, A_pred + 1e-16, marker="o", label="pred spectrum")
    plt.xlabel("spherical harmonic degree ell")
    plt.ylabel("sum_m |a_lm|^2")
    plt.title("Spectrum vs ell")
    plt.grid(True)
    plt.legend()
    plt.savefig("spectrum_vs_ell.png")

    plt.figure()
    plt.semilogy(ells, err_l + 1e-16, marker="o")
    plt.xlabel("spherical harmonic degree ell")
    plt.ylabel("spectral error")
    plt.title("Spectral error vs ell")
    plt.grid(True)
    plt.savefig("spectral_error_vs_ell.png")