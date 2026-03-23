import numpy as np
from numpy.linalg import eigh
from scipy.special import lpmv
from math import pi


def wave_sphere_exact(XYZ, t, f_handle, g_handle, Lmax, c, R, return_coeffs=False):
    """
    Exact spectral solution of u_tt = c^2 Δ_{S^2_R} u on a sphere of radius R.
    """
    XYZ = np.asarray(XYZ, dtype=float)

    quad = setup_quadrature(Lmax, R)
    fq, gq = sample_initial_data_on_quadrature(quad, f_handle, g_handle)
    flm, glm, mvals = compute_modal_coefficients(fq, gq, quad, Lmax)
    ulm = evolve_modal_coefficients(flm, glm, t, Lmax, c, R)
    eval_data = prepare_evaluation_points(XYZ, Lmax, R)
    Y_basis_big = np.hstack(precompute_Ylm_basis(eval_data, Lmax))
    is_real = np.isrealobj(fq) and np.isrealobj(gq)
    u = synthesize_solution(ulm, is_real, Y_basis_big)

    if return_coeffs:
        return u, {"flm": flm, "glm": glm, "ulm": ulm, "mvals": mvals}

    return u


# ===================== main stages =====================

def setup_quadrature(Lmax, R):
    """
    Set up Gauss-Legendre in mu and uniform quadrature in phi.
    """
    Ntheta = Lmax + 1
    Nphi = 2 * Lmax + 1

    mu, w_mu = gausslegendre(Ntheta)
    theta_q = np.arccos(mu)
    phi_q = np.linspace(0.0, 2.0 * pi, Nphi, endpoint=False)

    TH, PH = np.meshgrid(theta_q, phi_q, indexing="ij")
    XYZq = sph2cartR(TH.ravel(), PH.ravel(), R)

    dphi = 2.0 * pi / Nphi
    mvals = np.arange(-Lmax, Lmax + 1)
    Ephi = np.exp(-1j * phi_q[:, None] * mvals[None, :])

    return {
        "Ntheta": Ntheta,
        "Nphi": Nphi,
        "mu": mu,
        "w_mu": w_mu,
        "theta_q": theta_q,
        "phi_q": phi_q,
        "XYZq": XYZq,
        "dphi": dphi,
        "mvals": mvals,
        "Ephi": Ephi,
        "R": R,
    }


def sample_initial_data_on_quadrature(quad, f_handle, g_handle):
    """
    Sample f and g on the quadrature grid.
    """
    XYZq = quad["XYZq"]
    Ntheta = quad["Ntheta"]
    Nphi = quad["Nphi"]

    fq = np.asarray(f_handle(XYZq[:, 0], XYZq[:, 1], XYZq[:, 2])).reshape(Ntheta, Nphi)
    gq = np.asarray(g_handle(XYZq[:, 0], XYZq[:, 1], XYZq[:, 2])).reshape(Ntheta, Nphi)

    return fq, gq


def compute_modal_coefficients(fq, gq, quad, Lmax):
    """
    Compute spherical harmonic coefficients flm and glm from sampled data.
    """
    Ntheta = quad["Ntheta"]
    dphi = quad["dphi"]
    w_mu = quad["w_mu"]
    mu = quad["mu"]
    Ephi = quad["Ephi"]
    mvals = quad["mvals"]

    flm = np.zeros((Lmax + 1, 2 * Lmax + 1), dtype=complex)
    glm = np.zeros((Lmax + 1, 2 * Lmax + 1), dtype=complex)

    Fhat = (fq @ Ephi) * dphi
    Ghat = (gq @ Ephi) * dphi

    for ell in range(Lmax + 1):
        P = associated_legendre_block(ell, mu)

        for m in range(ell + 1):
            Nlm = spherical_harmonic_normalization(ell, m)

            col_pos = m + Lmax
            col_neg = -m + Lmax

            plm_vec = P[m, :]
            flm_pos = Nlm * np.dot(w_mu, plm_vec * Fhat[:, col_pos])
            glm_pos = Nlm * np.dot(w_mu, plm_vec * Ghat[:, col_pos])

            flm[ell, col_pos] = flm_pos
            glm[ell, col_pos] = glm_pos

            if m > 0:
                flm[ell, col_neg] = ((-1) ** m) * np.conj(flm_pos)
                glm[ell, col_neg] = ((-1) ** m) * np.conj(glm_pos)

    return flm, glm, mvals


def evolve_modal_coefficients(flm, glm, t, Lmax, c, R):
    """
    Evolve modal coefficients exactly in time.
    """
    ulm = np.zeros_like(flm)

    for ell in range(Lmax + 1):
        omega = (c / R) * np.sqrt(ell * (ell + 1))

        if ell == 0 or abs(omega) < 1e-15:
            ulm[ell, :] = flm[ell, :] + glm[ell, :] * t
        else:
            ulm[ell, :] = (
                flm[ell, :] * np.cos(omega * t)
                + (glm[ell, :] / omega) * np.sin(omega * t)
            )

    return ulm


def prepare_evaluation_points(XYZ, Lmax, R):
    """
    Prepare angles and azimuthal exponentials for evaluating the solution at XYZ.
    """
    XYZ = enforce_radius(XYZ, R)
    x, y, z = XYZ[:, 0], XYZ[:, 1], XYZ[:, 2]
    theta, phi = cart2sphAngles(x, y, z)

    mvals = np.arange(-Lmax, Lmax + 1)
    Em = np.exp(1j * phi[:, None] * mvals[None, :])
    mu_pts = np.cos(theta)

    return {
        "XYZ": XYZ,
        "theta": theta,
        "phi": phi,
        "mu_pts": mu_pts,
        "Em": Em,
        "mvals": mvals,
        "N": XYZ.shape[0],
        "R": R,
    }


def synthesize_solution(ulm, is_real, Y_basis_big):
    """
    Reconstruct u(XYZ, t) from modal coefficients ulm.
    """
    ulm_flat = np.ascontiguousarray(ulm.reshape(-1))
    u = Y_basis_big @ ulm_flat

    if is_real:
        u = u.real

    return u

def precompute_Ylm_basis(eval_data, Lmax):
    Y_basis = []
    mu_pts = eval_data["mu_pts"]
    Em = eval_data["Em"]

    for ell in range(Lmax + 1):
        Y_lm_all = build_Ylm_matrix_for_degree(ell, mu_pts, Em, Lmax)
        Y_basis.append(Y_lm_all)

    return Y_basis


# ===================== reusable math blocks =====================

def associated_legendre_block(ell, mu):
    """
    Return array P[m, i] = (-1)^m lpmv(m, ell, mu_i), for m=0..ell.
    Shape: (ell+1, len(mu))
    """
    return np.vstack([((-1) ** m) * lpmv(m, ell, mu) for m in range(ell + 1)])


def spherical_harmonic_normalization(ell, m):
    """
    L2 normalization factor for Y_l^m.
    """
    return np.sqrt((2 * ell + 1) / (4 * pi) * factratio(ell - m, ell + m))


def build_Ylm_matrix_for_degree(ell, mu_pts, Em, Lmax):
    """
    Build all Y_l^m values for fixed ell and all m=-Lmax..Lmax,
    stored in a matrix of shape (N, 2*Lmax+1).
    """
    N = mu_pts.shape[0]
    Y_lm_all = np.zeros((N, 2 * Lmax + 1), dtype=complex)

    P = associated_legendre_block(ell, mu_pts)

    for m in range(ell + 1):
        Nlm = np.sqrt((2 * ell + 1) / (4 * pi) * factratio(ell - m, ell + m))
        Ypos = Nlm * P[m, :] * Em[:, m + Lmax]
        Y_lm_all[:, m + Lmax] = Ypos

        if m > 0:
            Y_lm_all[:, -m + Lmax] = ((-1) ** m) * np.conj(Ypos)

    return Y_lm_all


# ======================= helpers =======================

def gausslegendre(n: int):
    """Gauss-Legendre nodes mu in [-1,1] and weights (n-point), Golub-Welsch."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if n == 1:
        return np.array([0.0]), np.array([2.0])

    k = np.arange(1, n)
    beta = 0.5 / np.sqrt(1.0 - (2.0 * k) ** (-2))
    T = np.diag(beta, 1) + np.diag(beta, -1)
    D, V = eigh(T)
    idx = np.argsort(D)
    mu = D[idx]
    V = V[:, idx]
    w = 2.0 * (V[0, :] ** 2)
    return mu, w


def factratio(a: int, b: int) -> float:
    """Return a!/b! for integers 0 <= a <= b."""
    if a == b:
        return 1.0
    if a > b or a < 0:
        raise ValueError("Require 0 <= a <= b")

    prod = 1.0
    for k in range(a + 1, b + 1):
        prod *= k
    return 1.0 / prod


def sph2cartR(theta, phi, R):
    """Convert spherical angles (theta, phi) to Cartesian coordinates on radius R."""
    theta = np.asarray(theta)
    phi = np.asarray(phi)
    x = R * np.sin(theta) * np.cos(phi)
    y = R * np.sin(theta) * np.sin(phi)
    z = R * np.cos(theta)
    return np.column_stack([x.ravel(), y.ravel(), z.ravel()])


def cart2sphAngles(x, y, z):
    """Return theta in [0, pi], phi in [0, 2pi) from Cartesian coords."""
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