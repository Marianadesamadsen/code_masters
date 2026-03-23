import numpy as np
from numpy.linalg import eigh
from scipy.special import lpmv
from math import pi 

def wave_sphere_exact(XYZ, t, f_handle, g_handle, Lmax, c, R, return_coeffs=False):
    """
    Exact spectral solution of u_tt = c^2 Δ_{S^2_R} u on a sphere of radius R.

    Parameters
    ----------
    XYZ : (N,3) array
        Points on (or near) the sphere of radius R.
    t : float
        Time at which to evaluate solution.
    f_handle : callable
        f_handle(x,y,z) initial displacement on the sphere.
    g_handle : callable
        g_handle(x,y,z) initial velocity on the sphere.
    Lmax : int
        Max spherical harmonic degree.
    c : float
        Wave speed along the surface.
    R : float
        Sphere radius.

    Returns
    -------
    u : (N,) array
        Solution u(XYZ_i, t). Real if f,g samples are real; complex otherwise.

    Notes
    -----
    Uses complex Condon–Shortley spherical harmonics with L2-normalization:
        Y_l^m(θ,φ) = N_lm P_l^m(cosθ) e^{i m φ},
        N_lm = sqrt( (2l+1)/(4π) * (l-m)!/(l+m)! )
    Quadrature:
        Gauss–Legendre in μ=cosθ with Nθ=Lmax+1, uniform φ with Nφ=2Lmax+1.
    """
    XYZ = np.asarray(XYZ, dtype=float)

    # ------------------ set up quadrature grid ------------------
    Ntheta = Lmax + 1
    Nphi = 2 * Lmax + 1

    mu, w_mu = gausslegendre(Ntheta)           # μ in [-1,1]
    theta_q = np.arccos(mu)
    phi_q = np.linspace(0.0, 2.0 * pi, Nphi, endpoint=False)

    TH, PH = np.meshgrid(theta_q, phi_q, indexing="ij")
    XYZq = sph2cartR(TH.ravel(), PH.ravel(), R)

    fq = np.asarray(f_handle(XYZq[:, 0], XYZq[:, 1], XYZq[:, 2])).reshape(Ntheta, Nphi)
    gq = np.asarray(g_handle(XYZq[:, 0], XYZq[:, 1], XYZq[:, 2])).reshape(Ntheta, Nphi)

    # ------------------ forward SHT: compute f_lm, g_lm ------------------
    dphi = 2.0 * pi / Nphi
    flm = np.zeros((Lmax + 1, 2 * Lmax + 1), dtype=complex)
    glm = np.zeros((Lmax + 1, 2 * Lmax + 1), dtype=complex)

    mvals = np.arange(-Lmax, Lmax + 1)

    # Ephi[j, k] = exp(-i * phi_j * m_k)
    Ephi = np.exp(-1j * phi_q[:, None] * mvals[None, :])  # (Nphi, 2L+1)

    # Fourier sums in phi for each theta row:
    # Fhat[i, m] = sum_j f(theta_i, phi_j) * exp(-i m phi_j) dphi
    Fhat = (fq @ Ephi) * dphi  # (Ntheta, 2L+1)
    Ghat = (gq @ Ephi) * dphi

    for ell in range(Lmax + 1):
        # P_m(μ) for m=0..ell at all μ
        # lpmv(m, ell, x) includes Condon–Shortley phase
        # Shape: (ell+1, Ntheta)
        #P = np.vstack([lpmv(m, ell, mu) for m in range(ell + 1)])
        P = np.vstack([((-1)**m) * lpmv(m, ell, mu) for m in range(ell + 1)])

        for m in range(ell + 1):
            Nlm = np.sqrt((2 * ell + 1) / (4 * pi) * factratio(ell - m, ell + m))

            col_pos = m + Lmax
            col_neg = -m + Lmax

            plm_vec = P[m, :]  # (Ntheta,)
            # Weighted mu-integral: sum_i w_mu(i) * P_l^m(mu_i) * Fhat(i, m)
            flm_pos = Nlm * np.dot(w_mu, plm_vec * Fhat[:, col_pos])
            glm_pos = Nlm * np.dot(w_mu, plm_vec * Ghat[:, col_pos])

            flm[ell, col_pos] = flm_pos
            glm[ell, col_pos] = glm_pos

            if m > 0:
                flm[ell, col_neg] = ((-1) ** m) * np.conj(flm_pos)
                glm[ell, col_neg] = ((-1) ** m) * np.conj(glm_pos)
     

    non_zero_f = np.sum(np.abs(flm) > 1e-10)
    non_zero_g = np.sum(np.abs(glm) > 1e-10)

    # ------------------ exact modal time evolution ------------------
    ulm = np.zeros_like(flm)
    for ell in range(Lmax + 1):
        omega = (c / R) * np.sqrt(ell * (ell + 1))
        if ell == 0 or abs(omega) < 1e-15:
            ulm[ell, :] = flm[ell, :] + glm[ell, :] * t
        else:
            ulm[ell, :] = flm[ell, :] * np.cos(omega * t) + (glm[ell, :] / omega) * np.sin(omega * t)
    

    # ------------------ synthesize u at requested XYZ nodes ------------------
    XYZ = enforce_radius(XYZ, R)
    x, y, z = XYZ[:, 0], XYZ[:, 1], XYZ[:, 2]
    theta, phi = cart2sphAngles(x, y, z)

    N = XYZ.shape[0]
    u = np.zeros(N, dtype=complex)

    Em = np.exp(1j * phi[:, None] * mvals[None, :])  # (N, 2L+1)
    mu_pts = np.cos(theta)

    for ell in range(Lmax + 1):
        # P_m(mu_pts) for m=0..ell
        #P = np.vstack([lpmv(m, ell, mu_pts) for m in range(ell + 1)])  # (ell+1, N)
        P = np.vstack([((-1)**m) * lpmv(m, ell, mu_pts) for m in range(ell + 1)])

        Y_lm_all = np.zeros((N, 2 * Lmax + 1), dtype=complex)
        for m in range(ell + 1):
            Nlm = np.sqrt((2 * ell + 1) / (4 * pi) * factratio(ell - m, ell + m))
            Ypos = Nlm * P[m, :] * Em[:, m + Lmax]
            Y_lm_all[:, m + Lmax] = Ypos
            if m > 0:
                Y_lm_all[:, -m + Lmax] = ((-1) ** m) * np.conj(Ypos)

        u += Y_lm_all @ ulm[ell, :].T

    if np.isrealobj(fq) and np.isrealobj(gq):
        u = u.real

    if return_coeffs:
        # also return modal coefficients at time t
        # flm, glm, ulm shapes: (Lmax+1, 2*Lmax+1) where m index is shifted by +Lmax
        return u, {"flm": flm, "glm": glm, "ulm": ulm, "mvals": mvals}

    return u


# ======================= helpers =======================

def gausslegendre(n: int):
    """Gauss–Legendre nodes μ∈[-1,1] and weights (n-point), Golub–Welsch."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if n == 1:
        return np.array([0.0]), np.array([2.0])

    k = np.arange(1, n)
    beta = 0.5 / np.sqrt(1.0 - (2.0 * k) ** (-2))
    T = np.diag(beta, 1) + np.diag(beta, -1)
    D, V = eigh(T)  # eigenvalues ascending for eigh, but sort anyway for safety
    idx = np.argsort(D)
    mu = D[idx]
    V = V[:, idx]
    w = 2.0 * (V[0, :] ** 2)
    return mu, w


def factratio(a: int, b: int) -> float:
    """(a)!/(b)! for integers 0<=a<=b via product, stable for moderate b."""
    if a == b:
        return 1.0
    if a > b or a < 0:
        raise ValueError("Require 0 <= a <= b")
    prod = 1.0
    for k in range(a + 1, b + 1):
        prod *= k
    return 1.0 / prod


def sph2cartR(theta, phi, R):
    """Convert spherical angles (θ,φ) to (x,y,z) on radius R (θ in [0,π], φ in [0,2π))."""
    theta = np.asarray(theta)
    phi = np.asarray(phi)
    x = R * np.sin(theta) * np.cos(phi)
    y = R * np.sin(theta) * np.sin(phi)
    z = R * np.cos(theta)
    return np.column_stack([x.ravel(), y.ravel(), z.ravel()])


def cart2sphAngles(x, y, z):
    """Return θ∈[0,π], φ∈[0,2π) from Cartesian coords."""
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    phi = np.arctan2(y, x)
    phi = np.where(phi < 0, phi + 2 * pi, phi)
    r = np.sqrt(x * x + y * y + z * z)
    mu = np.clip(z / np.maximum(r, np.finfo(float).eps), -1.0, 1.0)
    theta = np.arccos(mu)
    return theta, phi


def enforce_radius(XYZ, R):
    """Project points to the sphere of radius R."""
    XYZ = np.asarray(XYZ, dtype=float)
    r = np.sqrt(np.sum(XYZ**2, axis=1))
    scale = R / np.maximum(r, np.finfo(float).eps)
    return XYZ * scale[:, None]
