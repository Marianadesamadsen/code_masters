import numpy as np
import trimesh
from scipy.special import roots_legendre, roots_jacobi
import torch


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
    wrong = np.sum(n_tri * centroid, axis=1) < 0
    tri[wrong] = tri[wrong][:, [0, 2, 1]]
    return tri

def jacobi_gll(N):
    if N == 1:
        return np.array([-1.0, 1.0])
    x_int, _ = roots_jacobi(N - 1, 1, 1)
    return np.concatenate(([-1.0], x_int, [1.0]))

def vandermonde_1d(N, r):
    return np.vander(r, N + 1, increasing=True)

def warpfactor(N, rout):
    rout = np.asarray(rout, dtype=float)

    LGLr = jacobi_gll(N)
    req = np.linspace(-1.0, 1.0, N + 1)

    Veq = vandermonde_1d(N, req)
    Pmat = vandermonde_1d(N, rout).T

    Lmat = np.linalg.solve(Veq.T, Pmat)
    warp = Lmat.T @ (LGLr - req)

    zerof = np.abs(rout) < 1.0 - 1.0e-10
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
    x = np.asarray(x)
    y = np.asarray(y)

    L1 = (np.sqrt(3.0) * y + 1.0) / 3.0
    L2 = (-3.0 * x - np.sqrt(3.0) * y + 2.0) / 6.0
    L3 = (3.0 * x - np.sqrt(3.0) * y + 2.0) / 6.0

    r = -L2 + L3 - L1
    s = -L2 - L3 + L1
    return r, s

def reference_nodes_triangle(N):
    x, y = nodes_2d(N)
    return xytors(x, y)

def monomial_powers_triangle(N):
    powers = []
    for total in range(N + 1):
        a = np.arange(total + 1)
        b = total - a
        powers.append(np.stack([a, b], axis=1))
    return np.vstack(powers)

def vandermonde_triangle(r, s, powers):
    a, b = np.asarray(powers).T
    return (r[:, None] ** a) * (s[:, None] ** b)

def grad_vandermonde_triangle(r, s, powers):
    a, b = np.asarray(powers).T

    r_col = r[:, None]
    s_col = s[:, None]
    a_row = a[None, :]
    b_row = b[None, :]

    Vr = a_row * (r_col ** np.maximum(a_row - 1, 0)) * (s_col ** b_row)
    Vs = b_row * (r_col ** a_row) * (s_col ** np.maximum(b_row - 1, 0))

    return Vr, Vs

def triangle_quadrature(nq):
    xi, wi = roots_legendre(nq)
    eta, wj = roots_legendre(nq)

    X, E = np.meshgrid(xi, eta, indexing="xy")
    WX, WE = np.meshgrid(wi, wj, indexing="xy")

    s = E
    r_left = -1.0
    r_right = -s

    r = 0.5 * (r_right - r_left) * X + 0.5 * (r_right + r_left)
    w = WX * WE * 0.5 * (r_right - r_left)

    return r.ravel(), s.ravel(), w.ravel()

def get_reference_element(N):
    r, s = reference_nodes_triangle(N)
    powers = monomial_powers_triangle(N)

    V = vandermonde_triangle(r, s, powers)
    invV = np.linalg.inv(V)

    Vr, Vs = grad_vandermonde_triangle(r, s, powers)

    Dr = Vr @ invV
    Ds = Vs @ invV

    rq, sq, wq = triangle_quadrature(nq=N + 4)
    Vq = vandermonde_triangle(rq, sq, powers)
    Lq = Vq @ invV
    MassMatrix = Lq.T @ (wq[:, None] * Lq)

    NODETOL = 1e-12
    fmask1 = np.where(np.abs(s + 1.0) < NODETOL)[0]
    fmask2 = np.where(np.abs(r + s) < NODETOL)[0]
    fmask3 = np.where(np.abs(r + 1.0) < NODETOL)[0]

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


def derivative_matrix_1d(x):
    x = np.asarray(x)
    n = len(x)

    diff = x[:, None] - x[None, :]
    w = 1.0 / np.prod(diff + np.eye(n), axis=1)

    D = np.zeros((n, n))
    mask = ~np.eye(n, dtype=bool)
    D[mask] = (w[None, :] / w[:, None])[mask] / diff[mask]
    np.fill_diagonal(D, -np.sum(D, axis=1))

    return D


def compute_sJ_curved_face(x3D, y3D, z3D, Fmask):
    Nfp = Fmask.shape[0]
    t_nodes = np.linspace(-1.0, 1.0, Nfp)
    Dr1D = derivative_matrix_1d(t_nodes)

    _, wq = roots_legendre(Nfp)

    xf = x3D[Fmask, :]
    yf = y3D[Fmask, :]
    zf = z3D[Fmask, :]

    dxf = np.einsum("ij,jfk->ifk", Dr1D, xf)
    dyf = np.einsum("ij,jfk->ifk", Dr1D, yf)
    dzf = np.einsum("ij,jfk->ifk", Dr1D, zf)

    ds = np.sqrt(dxf**2 + dyf**2 + dzf**2)
    L = np.einsum("i,ifk->fk", wq, ds)

    sj_face = L / np.sum(wq)
    sJ_faces = np.ones((Nfp, 1, 1)) * sj_face[None, :, :]

    return sJ_faces.reshape(Nfp * 3, x3D.shape[1], order="F")


def surface_mass_integration(N=6, generation=4, R=1.0):
    P, tri = create_mesh(generation, R=R)
    tri = check_orientation(P, tri)

    K = tri.shape[0]

    r, s, Dr, Ds, MassMatrix, Fmask, V = get_reference_element(N)
    Np = len(r)

    v1 = P[:, tri[:, 0]]
    v2 = P[:, tri[:, 1]]
    v3 = P[:, tri[:, 2]]

    r_col = r[:, None]
    s_col = s[:, None]

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

    norm = np.sqrt(x**2 + y**2 + z**2)

    x3D = R * x / norm
    y3D = R * y / norm
    z3D = R * z / norm

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

    ar = np.cross(xs_vec, n_unit, axis=2) / J2D[:, :, None]
    a_s = np.cross(n_unit, xr_vec, axis=2) / J2D[:, :, None]

    rx2D = ar[:, :, 0]
    ry2D = ar[:, :, 1]
    rz2D = ar[:, :, 2]

    sx2D = a_s[:, :, 0]
    sy2D = a_s[:, :, 1]
    sz2D = a_s[:, :, 2]

    total_area = np.sum(J2D * MassMatrix.sum(axis=1)[:, None])

    sJ = compute_sJ_curved_face(x3D, y3D, z3D, Fmask)
    Fmask_flat = Fmask.T.reshape(-1)
    Fscale = sJ / J2D[Fmask_flat, :]

    # Precompute repeated objects for speed
    Mk = J2D[:, None, :] * MassMatrix[:, :, None]
    mass_row_sum = MassMatrix.sum(axis=1)
    ones = np.ones(Np)

    lam1 = -(r + s) / 2.0
    lam2 = (1.0 + r) / 2.0
    lam3 = (1.0 + s) / 2.0

    # exact_area = 4.0 * np.pi * R**2
    # print()
    # print("Check mass matrix and jacobians by integrating at the sphere surface")
    # print(f"Exact area is 4*pi*R^2 : {exact_area:.8f}")
    # print(f"Computed total area    : {total_area:.8f}")
    # print(f"Absolute error         : {abs(total_area - exact_area):.8e}")

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

        # Speed-up objects
        "Mk": Mk,
        "ones": ones,
        "mass_row_sum": mass_row_sum,
        "lam1": lam1,
        "lam2": lam2,
        "lam3": lam3,
    }


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

    # If computing on a whole batch
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


def compute_ut_and_mid(u_fem, dt, ut_order):
    """
    u_fem shape:
        (T, Np, K) or (B, T, Np, K)
    """

    # Single sample
    if u_fem.ndim == 3:
        if ut_order == 2:
            return (u_fem[2:] - u_fem[:-2]) / (2 * dt), u_fem[1:-1]

        elif ut_order == 4:
            return (
                -u_fem[4:]
                + 8 * u_fem[3:-1]
                - 8 * u_fem[1:-3]
                + u_fem[:-4]
            ) / (12 * dt), u_fem[2:-2]

        elif ut_order == 6:
            return (
                u_fem[:-6]
                - 9 * u_fem[1:-5]
                + 45 * u_fem[2:-4]
                - 45 * u_fem[4:-2]
                + 9 * u_fem[5:-1]
                - u_fem[6:]
            ) / (60 * dt), u_fem[3:-3]

        elif ut_order == 8:
            return (
                3 * u_fem[:-8]
                - 32 * u_fem[1:-7]
                + 168 * u_fem[2:-6]
                - 672 * u_fem[3:-5]
                + 672 * u_fem[5:-3]
                - 168 * u_fem[6:-2]
                + 32 * u_fem[7:-1]
                - 3 * u_fem[8:]
            ) / (840 * dt), u_fem[4:-4]

    # whople batch
    if u_fem.ndim == 4:
        if ut_order == 2:
            return (u_fem[:, 2:] - u_fem[:, :-2]) / (2 * dt), u_fem[:, 1:-1]

        elif ut_order == 4:
            return (
                -u_fem[:, 4:]
                + 8 * u_fem[:, 3:-1]
                - 8 * u_fem[:, 1:-3]
                + u_fem[:, :-4]
            ) / (12 * dt), u_fem[:, 2:-2]

        elif ut_order == 6:
            return (
                u_fem[:, :-6]
                - 9 * u_fem[:, 1:-5]
                + 45 * u_fem[:, 2:-4]
                - 45 * u_fem[:, 4:-2]
                + 9 * u_fem[:, 5:-1]
                - u_fem[:, 6:]
            ) / (60 * dt), u_fem[:, 3:-3]

        elif ut_order == 8:
            return (
                3 * u_fem[:, :-8]
                - 32 * u_fem[:, 1:-7]
                + 168 * u_fem[:, 2:-6]
                - 672 * u_fem[:, 3:-5]
                + 672 * u_fem[:, 5:-3]
                - 168 * u_fem[:, 6:-2]
                + 32 * u_fem[:, 7:-1]
                - 3 * u_fem[:, 8:]
            ) / (840 * dt), u_fem[:, 4:-4]

    raise ValueError(
        f"ut_order must be 2, 4, 6, or 8, and u_fem must have ndim 3 or 4. "
        f"Got ut_order={ut_order}, u_fem.shape={u_fem.shape}"
    )


def compute_wave_energy_batch(u, ut, out, c=1.0):
    """
    u  shape: (B, Np, K)
    ut shape: (B, Np, K)

    returns shape: (B,)
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

    kinetic = np.einsum("bik,ijk,bjk->bk", ut, Mk, ut)
    potential = np.einsum("bik,ijk,j->bk", grad_u_sq, Mk, ones)

    return 0.5 * np.sum(kinetic + c**2 * potential, axis=1)


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
    Accepts:
        u_array shape (T, N_nodes)
        or
        u_array shape (B, T, N_nodes)

    Returns:
        shape (T-reduction,) for single trajectory
        shape (B, T-reduction) for batch
    """
    if out is None:
        out = surface_mass_integration(N=N, generation=generation, R=R)

    u_array = np.asarray(u_array)
    u_fem = gnn_nodes_to_fem_batch(u_array, out)
    ut_array, u_mid = compute_ut_and_mid(u_fem, dt=dt, ut_order=ut_order)

    if u_array.ndim == 2:
        return compute_wave_energy_batch(u_mid, ut_array, out, c)

    if u_array.ndim == 3:
        B, Tm, Np, K = ut_array.shape
        u_flat = u_mid.reshape(B * Tm, Np, K)
        ut_flat = ut_array.reshape(B * Tm, Np, K)

        E_flat = compute_wave_energy_batch(u_flat, ut_flat, out, c)
        return E_flat.reshape(B, Tm)

    raise ValueError(
        f"Expected shape (T, N_nodes) or (B, T, N_nodes), got {u_array.shape}"
    )

# Using GPU instead of numpy
def energy_out_to_torch(out, device, dtype=torch.float32):
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
        ut = (u_fem[:, 2:] - u_fem[:, :-2]) / (2 * dt)
        u_mid = u_fem[:, 1:-1]

    elif ut_order == 4:
        ut = (
            -u_fem[:, 4:]
            + 8 * u_fem[:, 3:-1]
            - 8 * u_fem[:, 1:-3]
            + u_fem[:, :-4]
        ) / (12 * dt)
        u_mid = u_fem[:, 2:-2]

    elif ut_order == 6:
        ut = (
            u_fem[:, :-6]
            - 9 * u_fem[:, 1:-5]
            + 45 * u_fem[:, 2:-4]
            - 45 * u_fem[:, 4:-2]
            + 9 * u_fem[:, 5:-1]
            - u_fem[:, 6:]
        ) / (60 * dt)
        u_mid = u_fem[:, 3:-3]

    elif ut_order == 8:
        ut = (
            3 * u_fem[:, :-8]
            - 32 * u_fem[:, 1:-7]
            + 168 * u_fem[:, 2:-6]
            - 672 * u_fem[:, 3:-5]
            + 672 * u_fem[:, 5:-3]
            - 168 * u_fem[:, 6:-2]
            + 32 * u_fem[:, 7:-1]
            - 3 * u_fem[:, 8:]
        ) / (840 * dt)
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

    kinetic = torch.einsum("bik,ijk,bjk->bk", ut, Mk, ut)
    potential = torch.einsum("bik,ijk,j->bk", grad_u_sq, Mk, ones)

    return 0.5 * torch.sum(kinetic + c**2 * potential, dim=1)


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

if __name__ == "__main__":

    N = 6
    generation = 3
    R = 1.0
    c = 1.0

    dt = 0.01
    num_times = 200

    out = surface_mass_integration(N=N, generation=generation, R=R)
