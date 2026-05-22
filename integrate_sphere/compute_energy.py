import numpy as np
import trimesh
import torch
from scipy.special import eval_jacobi, gamma, roots_jacobi
from numpy.polynomial.legendre import legvander
from numpy.polynomial.legendre import Legendre

# Mesh utilities
def create_mesh(generation, R=1.0):
    mesh = trimesh.creation.icosphere(subdivisions=generation, radius=R)
    return mesh.vertices.T, mesh.faces.astype(int)

def check_orientation(P, tri):
    tri = tri.copy()

    v1 = P[:, tri[:, 0]].T
    v2 = P[:, tri[:, 1]].T
    v3 = P[:, tri[:, 2]].T

    n_tri = np.cross(v2 - v1, v3 - v1)
    centroid = (v1 + v2 + v3) / 3.0

    wrong = np.sum(n_tri * centroid, axis=1) < 0.0
    tri[wrong] = tri[wrong][:, [0, 2, 1]]

    return tri


# Jacobi polynomials 
def jacobi_p(x, alpha, beta, N):
    x = np.asarray(x, dtype=float)

    hN = (
        2.0 ** (alpha + beta + 1.0)
        / (2.0 * N + alpha + beta + 1.0)
        * gamma(N + alpha + 1.0)
        * gamma(N + beta + 1.0)
        / (gamma(N + 1.0) * gamma(N + alpha + beta + 1.0))
    )

    return eval_jacobi(N, alpha, beta, x) / np.sqrt(hN)


def grad_jacobi_p(x, alpha, beta, N):
    x = np.asarray(x, dtype=float)

    if N == 0:
        return np.zeros_like(x)

    return np.sqrt(N * (N + alpha + beta + 1.0)) * jacobi_p(
        x, alpha + 1.0, beta + 1.0, N - 1
    )


def jacobi_gl(alpha, beta, N):
    if N == 1:
        return np.array([-1.0, 1.0])

    x_int, _ = roots_jacobi(N - 1, alpha + 1.0, beta + 1.0)
    return np.concatenate(([-1.0], x_int, [1.0]))


def vandermonde_1d_matlab_directly(N, r):
    r = np.asarray(r, dtype=float).ravel()
    V1D = np.zeros((len(r), N + 1))

    for j in range(N + 1):
        V1D[:, j] = jacobi_p(r, 0.0, 0.0, j)

    return V1D

# Vectorized vandemonde using the legendre as alpha=beta=0
def vandermonde_1d(N, r):
    r = np.asarray(r, dtype=float).ravel()
    V = legvander(r, N)
    n = np.arange(N + 1)
    scale = np.sqrt((2 * n + 1) / 2)
    return V * scale

def grad_vandermonde_1d_matlab_directly(N, r):
    r = np.asarray(r, dtype=float).ravel()
    DVr = np.zeros((len(r), N + 1))

    for j in range(N + 1):
        DVr[:, j] = grad_jacobi_p(r, 0.0, 0.0, j)

    return DVr

# Vectorized gradient vandermonde using the legendre as alpha=beta=0
def grad_vandermonde_1d(N, r):
    r = np.asarray(r, dtype=float).ravel()

    DVr = np.column_stack([
        Legendre.basis(n).deriv()(r)
        for n in range(N + 1)
    ])

    scale = np.sqrt((2 * np.arange(N + 1) + 1) / 2)
    return DVr * scale

# Reference triangle nodes, mappings, and simplex basis
def warpfactor(N, rout):
    rout = np.asarray(rout, dtype=float).ravel()

    # Legendre-Gauss-Lobatto and equidistant nodes on [-1, 1]
    LGLr = jacobi_gl(0.0, 0.0, N)
    req = np.linspace(-1.0, 1.0, N + 1)

    # Compute Lagrange interpolation matrix from equidistant nodes to rout
    Veq = vandermonde_1d(N, req)
    Pmat = vandermonde_1d(N, rout).T
    Lmat = np.linalg.solve(Veq.T, Pmat).T

    warp = Lmat @ (LGLr - req)

    # Scale warp factor as in the book code
    zerof = np.abs(rout) < (1.0 - 1.0e-10)
    sf = 1.0 - (zerof.astype(float) * rout) ** 2
    warp = warp / sf + warp * (zerof.astype(float) - 1.0)

    return warp

def nodes_2d(N):
    alpopt = np.array([
        0.0000, 0.0000, 1.4152, 0.1001, 0.2751,
        0.9800, 1.0999, 1.2832, 1.3648, 1.4773,
        1.4959, 1.5743, 1.5770, 1.6223, 1.6258
    ])

    alpha = alpopt[N - 1] if N < 16 else 5.0 / 3.0

    i = np.arange(N + 1)
    j = np.arange(N + 1)
    I, J = np.meshgrid(i, j, indexing="ij")
    mask = I + J <= N

    L1 = I[mask] / N
    L3 = J[mask] / N
    L2 = 1.0 - L1 - L3

    x = -L2 + L3
    y = (-L2 - L3 + 2.0 * L1) / np.sqrt(3.0)

    blend1 = 4.0 * L2 * L3
    blend2 = 4.0 * L1 * L3
    blend3 = 4.0 * L1 * L2

    warpf1 = warpfactor(N, L3 - L2)
    warpf2 = warpfactor(N, L1 - L3)
    warpf3 = warpfactor(N, L2 - L1)

    warp1 = blend1 * warpf1 * (1.0 + (alpha * L1) ** 2)
    warp2 = blend2 * warpf2 * (1.0 + (alpha * L2) ** 2)
    warp3 = blend3 * warpf3 * (1.0 + (alpha * L3) ** 2)

    x = (
        x
        + warp1
        + np.cos(2.0 * np.pi / 3.0) * warp2
        + np.cos(4.0 * np.pi / 3.0) * warp3
    )

    y = (
        y
        + np.sin(2.0 * np.pi / 3.0) * warp2
        + np.sin(4.0 * np.pi / 3.0) * warp3
    )

    return x, y


def xytors(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    L1 = (np.sqrt(3.0) * y + 1.0) / 3.0
    L2 = (-3.0 * x - np.sqrt(3.0) * y + 2.0) / 6.0
    L3 = (3.0 * x - np.sqrt(3.0) * y + 2.0) / 6.0

    r = -L2 + L3 - L1
    s = -L2 - L3 + L1

    return r, s

def rstoab(r, s):
    r = np.asarray(r, dtype=float)
    s = np.asarray(s, dtype=float)

    a = np.empty_like(r)
    b = s.copy()

    # sets a = -1 at the singular point s = 1
    tol = 1.0e-12
    mask = np.abs(1.0 - s) > tol

    a[mask] = 2.0 * (1.0 + r[mask]) / (1.0 - s[mask]) - 1.0
    a[~mask] = -1.0

    return a, b

def simplex_2d_p(a, b, i, j):
    h1 = jacobi_p(a, 0.0, 0.0, i)
    h2 = jacobi_p(b, 2.0 * i + 1.0, 0.0, j)

    return np.sqrt(2.0) * h1 * h2 * (1.0 - b) ** i


def grad_simplex_2d_p(a, b, i, j):
    fa = jacobi_p(a, 0.0, 0.0, i)
    dfa = grad_jacobi_p(a, 0.0, 0.0, i)

    gb = jacobi_p(b, 2.0 * i + 1.0, 0.0, j)
    dgb = grad_jacobi_p(b, 2.0 * i + 1.0, 0.0, j)

    # r-derivative
    # d/dr = da/dr d/da + db/dr d/db = (2/(1-s)) d/da = (2/(1-b)) d/da
    dmodedr = dfa * gb
    if i > 0:
        dmodedr = dmodedr * (0.5 * (1.0 - b)) ** (i - 1)
    dmodedr = 2.0 ** (i + 0.5) * dmodedr

    # s-derivative
    # d/ds = ((1+a)/2)/((1-b)/2) d/da + d/db
    dmodeds = dfa * (gb * 0.5 * (1.0 + a))
    if i > 0:
        dmodeds = dmodeds * (0.5 * (1.0 - b)) ** (i - 1)

    tmp = dgb * (0.5 * (1.0 - b)) ** i
    if i > 0:
        tmp = tmp - 0.5 * i * gb * (0.5 * (1.0 - b)) ** (i - 1)

    dmodeds = 2.0 ** (i + 0.5) * (dmodeds + fa * tmp)

    return dmodedr, dmodeds


def vandermonde_2d(N, r, s):
    r = np.asarray(r, dtype=float).ravel()
    s = np.asarray(s, dtype=float).ravel()

    a, b = rstoab(r, s)

    Np = (N + 1) * (N + 2) // 2
    V2D = np.zeros((len(r), Np))

    sk = 0
    for i in range(N + 1):
        for j in range(N + 1 - i):
            V2D[:, sk] = simplex_2d_p(a, b, i, j)
            sk += 1

    return V2D


def grad_vandermonde_2d(N, r, s):
    r = np.asarray(r, dtype=float).ravel()
    s = np.asarray(s, dtype=float).ravel()

    a, b = rstoab(r, s)

    Np = (N + 1) * (N + 2) // 2
    V2Dr = np.zeros((len(r), Np))
    V2Ds = np.zeros((len(r), Np))

    sk = 0
    for i in range(N + 1):
        for j in range(N + 1 - i):
            V2Dr[:, sk], V2Ds[:, sk] = grad_simplex_2d_p(a, b, i, j)
            sk += 1

    return V2Dr, V2Ds

def dmatrices_2d(N, r, s, V):
    Vr, Vs = grad_vandermonde_2d(N, r, s)
    invV = np.linalg.inv(V)

    Dr = Vr @ invV
    Ds = Vs @ invV

    return Dr, Ds

def get_reference_element(N):
    x, y = nodes_2d(N)
    r, s = xytors(x, y)

    V = vandermonde_2d(N, r, s)
    Dr, Ds = dmatrices_2d(N, r, s, V)

    invV = np.linalg.inv(V)
    MassMatrix = invV.T @ invV

    NODETOL = 1.0e-12
    fmask1 = np.where(np.abs(s + 1.0) < NODETOL)[0]
    fmask2 = np.where(np.abs(r + s) < NODETOL)[0]
    fmask3 = np.where(np.abs(r + 1.0) < NODETOL)[0]

    # Sort face nodes so that each face has a consistent 1D ordering.
    fmask1 = fmask1[np.argsort(r[fmask1])]
    fmask2 = fmask2[np.argsort(r[fmask2])]
    fmask3 = fmask3[np.argsort(s[fmask3])]

    expected = N + 1
    if not (len(fmask1) == len(fmask2) == len(fmask3) == expected):
        raise RuntimeError(
            f"Bad Fmask sizes: {len(fmask1)}, {len(fmask2)}, {len(fmask3)}. "
            f"Expected {expected}."
        )

    Fmask = np.column_stack([fmask1, fmask2, fmask3])

    return r, s, Dr, Ds, MassMatrix, Fmask, V


# Curved face surface Jacobian
def compute_sJ_curved_face_old(x3D, y3D, z3D, Fmask, Nfp):
    Nfaces = 3
    Nfp = Fmask.shape[0]
    K = x3D.shape[1]

    # Equivalent to MATLAB:
    # [r1D, w1D] = JacobiGQ(0, 0, Nfp-1)
    r1D, w1D = roots_jacobi(Nfp, 0.0, 0.0)

    V1D = vandermonde_1d(Nfp - 1, r1D)
    Vr1D = grad_vandermonde_1d(Nfp - 1, r1D)
    Dr1D = Vr1D @ np.linalg.inv(V1D)

    one = np.ones(Nfp)
    denom = np.sum(w1D)

    sJ = np.zeros((Nfp * Nfaces, K))

    for k in range(K):
        for f in range(Nfaces):
            ids = Fmask[:, f]

            xf = x3D[ids, k]
            yf = y3D[ids, k]
            zf = z3D[ids, k]

            dxf = Dr1D @ xf
            dyf = Dr1D @ yf
            dzf = Dr1D @ zf

            ds = np.sqrt(dxf**2 + dyf**2 + dzf**2)

            # Equivalent to:
            # L = sum(M1D * ds)
            L = np.sum(w1D * ds)

            # Equivalent to:
            # sj_face = (L / (one' * M1D * one)) * one
            sj_face = (L / denom) * one

            start = f * Nfp
            end = (f + 1) * Nfp
            sJ[start:end, k] = sj_face

    return sJ

def compute_sJ_curved_face(x3D, y3D, z3D, Fmask, Nfp):
    Nfaces = 3
    Nfp = Fmask.shape[0]
    K = x3D.shape[1]

    r1D, w1D = roots_jacobi(Nfp, 0.0, 0.0)

    V1D = vandermonde_1d(Nfp - 1, r1D)
    Vr1D = grad_vandermonde_1d(Nfp - 1, r1D)
    Dr1D = Vr1D @ np.linalg.inv(V1D)

    denom = np.sum(w1D)
    sJ = np.zeros((Nfp * Nfaces, K))

    for f in range(Nfaces):
        ids = Fmask[:, f]

        xf = x3D[ids, :]
        yf = y3D[ids, :]
        zf = z3D[ids, :]

        dxf = Dr1D @ xf
        dyf = Dr1D @ yf
        dzf = Dr1D @ zf

        ds = np.sqrt(dxf**2 + dyf**2 + dzf**2)

        L = w1D @ ds
        sj_face = np.ones((Nfp, 1)) * (L[None, :] / denom)

        start = f * Nfp
        end = (f + 1) * Nfp
        sJ[start:end, :] = sj_face

    return sJ

def surface_mass_integration(N=6, generation=4, R=1.0):

    P, tri = create_mesh(generation, R=R)
    tri = check_orientation(P, tri)

    K = tri.shape[0]

    r, s, Dr, Ds, MassMatrix, Fmask, V = get_reference_element(N)
    Np = len(r)
    Nfp = N + 1

    v1 = P[:, tri[:, 0]]
    v2 = P[:, tri[:, 1]]
    v3 = P[:, tri[:, 2]]

    r_col = r[:, None]
    s_col = s[:, None]

    # Map from reference triangle to planar triangle in 3D
    x = 0.5 * (
        -(r_col + s_col) * v1[0][None, :]
        + (1.0 + r_col) * v2[0][None, :]
        + (1.0 + s_col) * v3[0][None, :]
    )
    y = 0.5 * (
        -(r_col + s_col) * v1[1][None, :]
        + (1.0 + r_col) * v2[1][None, :]
        + (1.0 + s_col) * v3[1][None, :]
    )
    z = 0.5 * (
        -(r_col + s_col) * v1[2][None, :]
        + (1.0 + r_col) * v2[2][None, :]
        + (1.0 + s_col) * v3[2][None, :]
    )

    # Project nodes to sphere
    norm = np.sqrt(x**2 + y**2 + z**2)
    x3D = R * x / norm
    y3D = R * y / norm
    z3D = R * z / norm

    # Derivatives of the curved mapping
    xr = Dr @ x3D
    xs = Ds @ x3D

    yr = Dr @ y3D
    ys = Ds @ y3D

    zr = Dr @ z3D
    zs = Ds @ z3D

    xr_vec = np.stack([xr, yr, zr], axis=2)
    xs_vec = np.stack([xs, ys, zs], axis=2)

    n_vec = np.cross(xr_vec, xs_vec, axis=2)
    J2D = np.linalg.norm(n_vec, axis=2)
    n_unit = n_vec / J2D[:, :, None]

    # Surface contravariant metric factors
    ar = np.cross(xs_vec, n_unit, axis=2) / J2D[:, :, None]
    a_s = np.cross(n_unit, xr_vec, axis=2) / J2D[:, :, None]

    rx2D = ar[:, :, 0]
    ry2D = ar[:, :, 1]
    rz2D = ar[:, :, 2]

    sx2D = a_s[:, :, 0]
    sy2D = a_s[:, :, 1]
    sz2D = a_s[:, :, 2]

    total_area = np.sum(J2D * MassMatrix.sum(axis=1)[:, None])

    # Curved face Jacobians and Fscale
    sJ = sJ = compute_sJ_curved_face(x3D, y3D, z3D, Fmask, Nfp) #compute_sJ_curved_face(x3D, y3D, z3D, r, s, Fmask, N) #
    Fmask_flat = Fmask.T.reshape(-1)
    Fscale = sJ / J2D[Fmask_flat, :]

    # Precompute repeated objects for energy integration
    Mk = J2D[:, None, :] * MassMatrix[:, :, None]
    ones = np.ones(Np)

    # Barycentric coordinates on the reference triangle
    lam1 = -(r + s) / 2.0
    lam2 = (1.0 + r) / 2.0
    lam3 = (1.0 + s) / 2.0

    return {
        "P": P,
        "tri": tri,
        "r": r,
        "s": s,
        "Dr": Dr,
        "Ds": Ds,
        "MassMatrix": MassMatrix,
        "Fmask": Fmask,
        "V": V,
        "x3D": x3D,
        "y3D": y3D,
        "z3D": z3D,
        "J2D": J2D,
        "rx2D": rx2D,
        "ry2D": ry2D,
        "rz2D": rz2D,
        "sx2D": sx2D,
        "sy2D": sy2D,
        "sz2D": sz2D,
        "sJ": sJ,
        "Fscale": Fscale,
        "total_area": total_area,
        "exact_area": 4.0 * np.pi * R**2,
        "area_error": abs(total_area - 4.0 * np.pi * R**2),

        # Speed-up objects
        "Mk": Mk,
        "ones": ones,
        "lam1": lam1,
        "lam2": lam2,
        "lam3": lam3,
    }


# Mapping GNN vertex values to spectral element nodes
def gnn_nodes_to_fem_batch(u_node, out):
    tri = out["tri"]
    lam1 = out["lam1"]
    lam2 = out["lam2"]
    lam3 = out["lam3"]

    u_node = np.asarray(u_node)

    if u_node.ndim == 2:
        u1 = u_node[:, tri[:, 0]]
        u2 = u_node[:, tri[:, 1]]
        u3 = u_node[:, tri[:, 2]]

        return (
            lam1[None, :, None] * u1[:, None, :]
            + lam2[None, :, None] * u2[:, None, :]
            + lam3[None, :, None] * u3[:, None, :]
        )

    if u_node.ndim == 3:
        u1 = u_node[:, :, tri[:, 0]]
        u2 = u_node[:, :, tri[:, 1]]
        u3 = u_node[:, :, tri[:, 2]]

        return (
            lam1[None, None, :, None] * u1[:, :, None, :]
            + lam2[None, None, :, None] * u2[:, :, None, :]
            + lam3[None, None, :, None] * u3[:, :, None, :]
        )

    raise ValueError(
        f"Expected shape (T, N_nodes) or (B, T, N_nodes), got {u_node.shape}"
    )

# NumPy energy computation
def compute_ut_and_mid(u_fem, dt, ut_order):
    """
    u_fem shape:
        (T, Np, K) or (B, T, Np, K)
    """
    if u_fem.ndim == 3:
        if ut_order == 2:
            return (u_fem[2:] - u_fem[:-2]) / (2.0 * dt), u_fem[1:-1]

        if ut_order == 4:
            return (
                -u_fem[4:]
                + 8.0 * u_fem[3:-1]
                - 8.0 * u_fem[1:-3]
                + u_fem[:-4]
            ) / (12.0 * dt), u_fem[2:-2]

        if ut_order == 6:
            return (
                u_fem[:-6]
                - 9.0 * u_fem[1:-5]
                + 45.0 * u_fem[2:-4]
                - 45.0 * u_fem[4:-2]
                + 9.0 * u_fem[5:-1]
                - u_fem[6:]
            ) / (60.0 * dt), u_fem[3:-3]

        if ut_order == 8:
            return (
                3.0 * u_fem[:-8]
                - 32.0 * u_fem[1:-7]
                + 168.0 * u_fem[2:-6]
                - 672.0 * u_fem[3:-5]
                + 672.0 * u_fem[5:-3]
                - 168.0 * u_fem[6:-2]
                + 32.0 * u_fem[7:-1]
                - 3.0 * u_fem[8:]
            ) / (840.0 * dt), u_fem[4:-4]

    if u_fem.ndim == 4:
        if ut_order == 2:
            return (u_fem[:, 2:] - u_fem[:, :-2]) / (2.0 * dt), u_fem[:, 1:-1]

        if ut_order == 4:
            return (
                -u_fem[:, 4:]
                + 8.0 * u_fem[:, 3:-1]
                - 8.0 * u_fem[:, 1:-3]
                + u_fem[:, :-4]
            ) / (12.0 * dt), u_fem[:, 2:-2]

        if ut_order == 6:
            return (
                u_fem[:, :-6]
                - 9.0 * u_fem[:, 1:-5]
                + 45.0 * u_fem[:, 2:-4]
                - 45.0 * u_fem[:, 4:-2]
                + 9.0 * u_fem[:, 5:-1]
                - u_fem[:, 6:]
            ) / (60.0 * dt), u_fem[:, 3:-3]

        if ut_order == 8:
            return (
                3.0 * u_fem[:, :-8]
                - 32.0 * u_fem[:, 1:-7]
                + 168.0 * u_fem[:, 2:-6]
                - 672.0 * u_fem[:, 3:-5]
                + 672.0 * u_fem[:, 5:-3]
                - 168.0 * u_fem[:, 6:-2]
                + 32.0 * u_fem[:, 7:-1]
                - 3.0 * u_fem[:, 8:]
            ) / (840.0 * dt), u_fem[:, 4:-4]

    raise ValueError(
        f"ut_order must be 2, 4, 6, or 8, and u_fem must have ndim 3 or 4. "
        f"Got ut_order={ut_order}, u_fem.shape={u_fem.shape}"
    )


def compute_wave_energy_batch(u, ut, out, c=1.0):
    """
    Compute wave energy for a batch.

    u  shape: (B, Np, K)
    ut shape: (B, Np, K)

    Returns
    -------
    ndarray, shape (B,)
    """
    Dr = out["Dr"]
    Ds = out["Ds"]
    Mk = out["Mk"]
    ones = out["ones"]

    rx2D = out["rx2D"]
    ry2D = out["ry2D"]
    rz2D = out["rz2D"]

    sx2D = out["sx2D"]
    sy2D = out["sy2D"]
    sz2D = out["sz2D"]

    ur = np.einsum("bjk,ij->bik", u, Dr)
    us = np.einsum("bjk,ij->bik", u, Ds)

    ux = rx2D[None, :, :] * ur + sx2D[None, :, :] * us
    uy = ry2D[None, :, :] * ur + sy2D[None, :, :] * us
    uz = rz2D[None, :, :] * ur + sz2D[None, :, :] * us

    grad_u_sq = ux**2 + uy**2 + uz**2

    term1 = np.einsum("bik,ijk,bjk->bk", ut, Mk, ut)
    term2 = np.einsum("bik,ijk,j->bk", grad_u_sq, Mk, ones)

    return 0.5 * np.sum(term1 + c**2 * term2, axis=1)


def compute_energy_over_time(
    u_array,
    generation=4,
    R=1.0,
    c=1.0,
    N=6,
    dt=1.0,
    out=None,
    ut_order=4,
):
    """
    Accepts
    -------
    u_array : ndarray
        Shape (T, N_nodes) or (B, T, N_nodes)

    Returns
    -------
    ndarray
        Shape (T_reduced,) or (B, T_reduced)
    """
    if out is None:
        out = surface_mass_integration(N=N, generation=generation, R=R)

    u_array = np.asarray(u_array)
    u_fem = gnn_nodes_to_fem_batch(u_array, out)
    ut_array, u_mid = compute_ut_and_mid(u_fem, dt=dt, ut_order=ut_order)

    if u_array.ndim == 2:
        return compute_wave_energy_batch(u_mid, ut_array, out, c=c)

    if u_array.ndim == 3:
        B, Tm, Np, K = ut_array.shape
        u_flat = u_mid.reshape(B * Tm, Np, K)
        ut_flat = ut_array.reshape(B * Tm, Np, K)

        E_flat = compute_wave_energy_batch(u_flat, ut_flat, out, c=c)
        return E_flat.reshape(B, Tm)

    raise ValueError(
        f"Expected shape (T, N_nodes) or (B, T, N_nodes), got {u_array.shape}"
    )


# Torch energy computation
def energy_out_to_torch(out, device, dtype=torch.float32):
    """
    Move the reusable energy objects to torch.
    """
    torch_out = {}

    int_keys = ["tri"]
    float_keys = [
        "Dr", "Ds", "Mk", "rx2D", "ry2D", "rz2D",
        "sx2D", "sy2D", "sz2D", "lam1", "lam2", "lam3", "ones",
    ]

    for key in int_keys:
        torch_out[key] = torch.as_tensor(out[key], device=device, dtype=torch.long)

    for key in float_keys:
        torch_out[key] = torch.as_tensor(out[key], device=device, dtype=dtype)

    return torch_out


def gnn_nodes_to_fem_torch(u_node, out):
    """
    u_node shape: (B, T, N_nodes)
    returns:      (B, T, Np, K)
    """
    tri = out["tri"]

    u1 = u_node[:, :, tri[:, 0]]
    u2 = u_node[:, :, tri[:, 1]]
    u3 = u_node[:, :, tri[:, 2]]

    return (
        out["lam1"][None, None, :, None] * u1[:, :, None, :]
        + out["lam2"][None, None, :, None] * u2[:, :, None, :]
        + out["lam3"][None, None, :, None] * u3[:, :, None, :]
    )


def compute_ut_and_mid_torch(u_fem, dt, ut_order=4):
    """
    u_fem shape: (B, T, Np, K)
    """
    if ut_order == 2:
        ut = (u_fem[:, 2:] - u_fem[:, :-2]) / (2.0 * dt)
        u_mid = u_fem[:, 1:-1]

    elif ut_order == 4:
        ut = (
            -u_fem[:, 4:]
            + 8.0 * u_fem[:, 3:-1]
            - 8.0 * u_fem[:, 1:-3]
            + u_fem[:, :-4]
        ) / (12.0 * dt)
        u_mid = u_fem[:, 2:-2]

    elif ut_order == 6:
        ut = (
            u_fem[:, :-6]
            - 9.0 * u_fem[:, 1:-5]
            + 45.0 * u_fem[:, 2:-4]
            - 45.0 * u_fem[:, 4:-2]
            + 9.0 * u_fem[:, 5:-1]
            - u_fem[:, 6:]
        ) / (60.0 * dt)
        u_mid = u_fem[:, 3:-3]

    elif ut_order == 8:
        ut = (
            3.0 * u_fem[:, :-8]
            - 32.0 * u_fem[:, 1:-7]
            + 168.0 * u_fem[:, 2:-6]
            - 672.0 * u_fem[:, 3:-5]
            + 672.0 * u_fem[:, 5:-3]
            - 168.0 * u_fem[:, 6:-2]
            + 32.0 * u_fem[:, 7:-1]
            - 3.0 * u_fem[:, 8:]
        ) / (840.0 * dt)
        u_mid = u_fem[:, 4:-4]

    else:
        raise ValueError("ut_order must be 2, 4, 6, or 8")

    return ut, u_mid


def compute_wave_energy_torch(u, ut, out, c=1.0):
    """
    u  shape: (BT, Np, K)
    ut shape: (BT, Np, K)

    returns shape: (BT,)
    """
    Dr = out["Dr"]
    Ds = out["Ds"]
    Mk = out["Mk"]
    ones = out["ones"]

    ur = torch.einsum("bjk,ij->bik", u, Dr)
    us = torch.einsum("bjk,ij->bik", u, Ds)

    ux = out["rx2D"][None, :, :] * ur + out["sx2D"][None, :, :] * us
    uy = out["ry2D"][None, :, :] * ur + out["sy2D"][None, :, :] * us
    uz = out["rz2D"][None, :, :] * ur + out["sz2D"][None, :, :] * us

    grad_u_sq = ux**2 + uy**2 + uz**2

    term1 = torch.einsum("bik,ijk,bjk->bk", ut, Mk, ut)
    term2 = torch.einsum("bik,ijk,j->bk", grad_u_sq, Mk, ones)

    return 0.5 * torch.sum(term1 + c**2 * term2, dim=1)


def compute_energy_over_time_torch(u_node, out, dt, c=1.0, ut_order=4):
    """
    u_node shape: (B, T, N_nodes)
    returns:      (B, T_reduced)
    """
    u_fem = gnn_nodes_to_fem_torch(u_node, out)
    ut, u_mid = compute_ut_and_mid_torch(u_fem, dt=dt, ut_order=ut_order)

    B, Tm, Np, K = ut.shape

    u_flat = u_mid.reshape(B * Tm, Np, K)
    ut_flat = ut.reshape(B * Tm, Np, K)

    E_flat = compute_wave_energy_torch(u_flat, ut_flat, out, c=c)

    return E_flat.reshape(B, Tm)


# Quick check
if __name__ == "__main__":
    N = 6
    generation = 3
    R = 1.0

    out = surface_mass_integration(N=N, generation=generation, R=R)

    print("Check mass matrix and jacobians by integrating at the sphere surface")
    print(f"Exact area is 4*pi*R^2 : {out['exact_area']:.8f}")
    print(f"Computed total area    : {out['total_area']:.8f}")
    print(f"Absolute error         : {out['area_error']:.8e}")
