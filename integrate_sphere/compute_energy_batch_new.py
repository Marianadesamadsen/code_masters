import numpy as np
import trimesh
from scipy.special import roots_legendre
from scipy.special import roots_jacobi


def create_mesh(generation, R=1.0):
    mesh = trimesh.creation.icosphere(subdivisions=generation, radius=R)
    P = mesh.vertices.T
    tri = mesh.faces.astype(int)
    return P, tri


def check_orientation(P, tri):
    tri = tri.copy()

    v1 = P[:, tri[:, 0]].T
    v2 = P[:, tri[:, 1]].T
    v3 = P[:, tri[:, 2]].T

    n_tri = np.cross(v2 - v1, v3 - v1)
    centroid = (v1 + v2 + v3) / 3.0

    wrong_orientation = np.sum(n_tri * centroid, axis=1) < 0
    tri[wrong_orientation, [1, 2]] = tri[wrong_orientation][:, [2, 1]]
    num_flipped = np.sum(wrong_orientation)

    return tri


def jacobi_gll(N):
    """
    Legendre-Gauss-Lobatto nodes on [-1, 1].
    """
    if N == 1:
        return np.array([-1.0, 1.0])

    x_int, _ = roots_jacobi(N - 1, 1, 1)
    return np.concatenate(([-1.0], x_int, [1.0]))


def vandermonde_1d(N, r):

    V = np.vander(r, N + 1, increasing=True)
    return V


def warpfactor(N, rout):
    """
    Python translation of MATLAB Warpfactor(N, rout).
    """
    rout = np.asarray(rout, dtype=float)

    # LGL and equidistant nodes
    LGLr = jacobi_gll(N)
    req = np.linspace(-1.0, 1.0, N + 1)

    # Vandermonde at equidistant nodes
    Veq = vandermonde_1d(N, req)

    # Evaluate basis at rout
    Pmat = vandermonde_1d(N, rout).T  # shape (N+1, Nr)

    # Equivalent to MATLAB: Lmat = Veq' \ Pmat
    Lmat = np.linalg.solve(Veq.T, Pmat)

    # Warp factor
    warp = Lmat.T @ (LGLr - req)

    # Scale factor, exactly like MATLAB
    zerof = np.abs(rout) < 1.0 - 1.0e-10
    sf = 1.0 - (zerof.astype(float) * rout) ** 2

    warp = warp / sf + warp * (zerof.astype(float) - 1.0)

    return warp


def nodes_2d(N):
    """
    Python translation of MATLAB Nodes2D(N).

    Returns
    -------
    x, y : arrays
        Warped nodes in the equilateral triangle.
    """
    alpopt = np.array([
        0.0000, 0.0000, 1.4152, 0.1001, 0.2751,
        0.9800, 1.0999, 1.2832, 1.3648, 1.4773,
        1.4959, 1.5743, 1.5770, 1.6223, 1.6258
    ])

    if N < 16:
        alpha = alpopt[N - 1]   # MATLAB indexing starts at 1
    else:
        alpha = 5.0 / 3.0

    Np = (N + 1) * (N + 2) // 2

    L1 = np.zeros(Np)
    L3 = np.zeros(Np)

    sk = 0
    for n in range(1, N + 2):
        for m in range(1, N + 3 - n):
            L1[sk] = (n - 1) / N
            L3[sk] = (m - 1) / N
            sk += 1

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
    """
    Python translation of MATLAB xytors(x,y).

    Converts equilateral triangle coordinates (x,y)
    to reference triangle coordinates (r,s).
    """
    x = np.asarray(x)
    y = np.asarray(y)

    L1 = (np.sqrt(3.0) * y + 1.0) / 3.0
    L2 = (-3.0 * x - np.sqrt(3.0) * y + 2.0) / 6.0
    L3 = (3.0 * x - np.sqrt(3.0) * y + 2.0) / 6.0

    r = -L2 + L3 - L1
    s = -L2 - L3 + L1

    return r, s


def reference_nodes_triangle_old(N):
    r_list = []
    s_list = []

    for j in range(N + 1):
        for i in range(N + 1 - j):
            r = -1.0 + 2.0 * i / N
            s = -1.0 + 2.0 * j / N
            r_list.append(r)
            s_list.append(s)

    return np.array(r_list), np.array(s_list)


def reference_nodes_triangle(N):
    x, y = nodes_2d(N)
    r, s = xytors(x, y)
    return r, s


def monomial_powers_triangle(N):
    powers = []
    for total in range(N + 1):
        for a in range(total + 1):
            b = total - a
            powers.append((a, b))
    return powers


def vandermonde_triangle(r, s, N):
    powers = monomial_powers_triangle(N)
    return vandermonde_triangle(r, s, powers)

def vandermonde_triangle(r, s, powers):
    V = np.zeros((len(r), len(powers)))

    for n, (a, b) in enumerate(powers):
        V[:, n] = r**a * s**b

    return V


def grad_vandermonde_triangle(r, s, powers):
    Vr = np.zeros((len(r), len(powers)))
    Vs = np.zeros((len(r), len(powers)))

    for n, (a, b) in enumerate(powers):
        if a > 0:
            Vr[:, n] = a * r ** (a - 1) * s**b
        if b > 0:
            Vs[:, n] = b * r**a * s ** (b - 1)

    return Vr, Vs


def triangle_quadrature(nq):
    """
    Tensor-product Gauss rule mapped to reference triangle:
        -1 <= s <= 1
        -1 <= r <= -s
    """
    xi, wi = roots_legendre(nq)
    eta, wj = roots_legendre(nq)

    rq = []
    sq = []
    wq = []

    for e, we in zip(eta, wj):
        s = e
        r_left = -1.0
        r_right = -s

        for x, wx in zip(xi, wi):
            r = 0.5 * (r_right - r_left) * x + 0.5 * (r_right + r_left)
            weight = wx * we * 0.5 * (r_right - r_left)

            rq.append(r)
            sq.append(s)
            wq.append(weight)

    return np.array(rq), np.array(sq), np.array(wq)


def get_reference_element(N):
    r, s = reference_nodes_triangle(N)
    powers = monomial_powers_triangle(N)

    V = vandermonde_triangle(r, s, powers)
    invV = np.linalg.inv(V)

    Vr, Vs = grad_vandermonde_triangle(r, s, powers)

    Dr = Vr @ invV
    Ds = Vs @ invV

    # Build nodal mass matrix by quadrature
    rq, sq, wq = triangle_quadrature(nq=N + 4)
    Vq = vandermonde_triangle(rq, sq, powers)

    # Values of nodal basis functions at quadrature points
    Lq = Vq @ invV

    MassMatrix = Lq.T @ (wq[:, None] * Lq)

    NODETOL = 1e-12

    fmask1 = np.where(np.abs(s + 1.0) < NODETOL)[0]
    fmask2 = np.where(np.abs(r + s) < NODETOL)[0]
    fmask3 = np.where(np.abs(r + 1.0) < NODETOL)[0]

    # Sort nodes along each face
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
    D = np.zeros((n, n))

    w = np.ones(n)
    for j in range(n):
        for k in range(n):
            if k != j:
                w[j] /= x[j] - x[k]

    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = w[j] / w[i] / (x[i] - x[j])

    D[np.diag_indices(n)] = -np.sum(D, axis=1)

    return D


def compute_sJ_curved_face(x3D, y3D, z3D, Fmask):
    Nfaces = 3
    Nfp = Fmask.shape[0]
    K = x3D.shape[1]

    t_nodes = np.linspace(-1.0, 1.0, Nfp)
    Dr1D = derivative_matrix_1d(t_nodes)

    tq, wq = roots_legendre(Nfp)
    M1D = np.diag(wq)
    one = np.ones(Nfp)

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

            L = np.sum(M1D @ ds)
            sj_face = (L / (one @ M1D @ one)) * one

            rows = slice(f * Nfp, (f + 1) * Nfp)
            sJ[rows, k] = sj_face

    return sJ


def surface_mass_integration(N=6, generation=3, R=1.0):
    P, tri = create_mesh(generation, R=R)
    tri = check_orientation(P, tri)

    K = tri.shape[0]

    r, s, Dr, Ds, MassMatrix, Fmask, V = get_reference_element(N)
    Np = len(r)

    rx2D = np.zeros((Np, K))
    ry2D = np.zeros((Np, K))
    rz2D = np.zeros((Np, K))

    sx2D = np.zeros((Np, K))
    sy2D = np.zeros((Np, K))
    sz2D = np.zeros((Np, K))

    J2D = np.zeros((Np, K))

    x3D = np.zeros((Np, K))
    y3D = np.zeros((Np, K))
    z3D = np.zeros((Np, K))

    total_area = 0.0

    for k in range(K):
        v1 = P[:, tri[k, 0]]
        v2 = P[:, tri[k, 1]]
        v3 = P[:, tri[k, 2]]

        x = 0.5 * (-(r + s) * v1[0] + (1.0 + r) * v2[0] + (1.0 + s) * v3[0])
        y = 0.5 * (-(r + s) * v1[1] + (1.0 + r) * v2[1] + (1.0 + s) * v3[1])
        z = 0.5 * (-(r + s) * v1[2] + (1.0 + r) * v2[2] + (1.0 + s) * v3[2])

        norm = np.sqrt(x**2 + y**2 + z**2)

        x = R * x / norm
        y = R * y / norm
        z = R * z / norm

        x3D[:, k] = x
        y3D[:, k] = y
        z3D[:, k] = z

        xr = Dr @ x
        xs = Ds @ x

        yr = Dr @ y
        ys = Ds @ y

        zr = Dr @ z
        zs = Ds @ z

        xr_vec = np.column_stack([xr, yr, zr])
        xs_vec = np.column_stack([xs, ys, zs])

        n_vec = np.cross(xr_vec, xs_vec, axis=1)
        J = np.sqrt(np.sum(n_vec**2, axis=1))

        n_unit = n_vec / J[:, None]

        ar = np.cross(xs_vec, n_unit, axis=1) / J[:, None]
        a_s = np.cross(n_unit, xr_vec, axis=1) / J[:, None]

        rx2D[:, k] = ar[:, 0]
        ry2D[:, k] = ar[:, 1]
        rz2D[:, k] = ar[:, 2]

        sx2D[:, k] = a_s[:, 0]
        sy2D[:, k] = a_s[:, 1]
        sz2D[:, k] = a_s[:, 2]

        J2D[:, k] = J

        M_k = np.diag(J) @ MassMatrix
        total_area += np.sum(M_k)

    sJ = compute_sJ_curved_face(x3D, y3D, z3D, Fmask)

    Fmask_flat = Fmask.T.reshape(-1)
    Fscale = sJ / J2D[Fmask_flat, :]

    exact_area = 4.0 * np.pi * R**2

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
    }

def compute_wave_energy_batch(u, ut, out, c=1.0):
    """
    Compute discrete wave energy for a batch.

    u  shape: (B, Np, K)
    ut shape: (B, Np, K)

    returns shape: (B,)
    """

    Dr = out["Dr"]
    Ds = out["Ds"]
    MassMatrix = out["MassMatrix"]
    J2D = out["J2D"]

    rx2D = out["rx2D"]
    ry2D = out["ry2D"]
    rz2D = out["rz2D"]

    sx2D = out["sx2D"]
    sy2D = out["sy2D"]
    sz2D = out["sz2D"]

    B, Np, K = u.shape
    energy = np.zeros(B, dtype=np.float64)

    for k in range(K):
        uk = u[:, :, k]      # (B, Np)
        utk = ut[:, :, k]    # (B, Np)

        # Reference derivatives
        ur = uk @ Dr.T       # (B, Np)
        us = uk @ Ds.T       # (B, Np)

        # Surface gradient
        ux = rx2D[:, k][None, :] * ur + sx2D[:, k][None, :] * us
        uy = ry2D[:, k][None, :] * ur + sy2D[:, k][None, :] * us
        uz = rz2D[:, k][None, :] * ur + sz2D[:, k][None, :] * us

        grad_u_sq = ux**2 + uy**2 + uz**2

        Mk = np.diag(J2D[:, k]) @ MassMatrix

        kinetic = np.einsum("bi,ij,bj->b", utk, Mk, utk)
        potential = np.einsum("bi,ij,j->b", grad_u_sq, Mk, np.ones(Np))

        energy += 0.5 * (kinetic + c**2 * potential)

    return energy


def gnn_nodes_to_fem_batch(u_node, out):
    """
    Convert GNN nodal values to FEM element values.

    u_node shape:
        (T, N_nodes) or (B, T, N_nodes)

    returns:
        (T, Np, K) or (B, T, Np, K)
    """

    tri = out["tri"]
    r = out["r"]
    s = out["s"]

    lam1 = -(r + s) / 2.0
    lam2 = (1.0 + r) / 2.0
    lam3 = (1.0 + s) / 2.0

    u_node = np.asarray(u_node)

    if u_node.ndim == 2:
        u1 = u_node[:, tri[:, 0]]  # (T, K)
        u2 = u_node[:, tri[:, 1]]
        u3 = u_node[:, tri[:, 2]]

        return (
            lam1[None, :, None] * u1[:, None, :]
            + lam2[None, :, None] * u2[:, None, :]
            + lam3[None, :, None] * u3[:, None, :]
        )

    if u_node.ndim == 3:
        u1 = u_node[:, :, tri[:, 0]]  # (B, T, K)
        u2 = u_node[:, :, tri[:, 1]]
        u3 = u_node[:, :, tri[:, 2]]

        return (
            lam1[None, None, :, None] * u1[:, :, None, :]
            + lam2[None, None, :, None] * u2[:, :, None, :]
            + lam3[None, None, :, None] * u3[:, :, None, :]
        )

    raise ValueError(f"Expected shape (T, N_nodes) or (B, T, N_nodes), got {u_node.shape}")


def compute_energy_over_time(u_array, generation=4, R=1, c=1, N=6, dt=1, out=None):
    """
    Compute wave energy over time.

    Accepts:
        u_array shape (T, N_nodes)
        or
        u_array shape (B, T, N_nodes)

    Returns:
        shape (T-2,) for single trajectory
        shape (B, T-2) for batch
    """

    if out is None:
        out = surface_mass_integration(N=N, generation=generation, R=R)

    u_array = np.asarray(u_array)
    u_fem = gnn_nodes_to_fem_batch(u_array, out)

    if u_array.ndim == 2:
        # u_fem shape: (T, Np, K)
        ut_array = (u_fem[2:] - u_fem[:-2]) / (2 * dt)

        E = []
        for t in range(ut_array.shape[0]):
            Et = compute_wave_energy_batch(
                u_fem[t + 1][None, :, :],
                ut_array[t][None, :, :],
                out,
                c,
            )[0]
            E.append(Et)

        return np.asarray(E)

    if u_array.ndim == 3:
        # u_fem shape: (B, T, Np, K)
        ut_array = (u_fem[:, 2:] - u_fem[:, :-2]) / (2 * dt)

        B, Tm2, Np, K = ut_array.shape
        E = np.zeros((B, Tm2), dtype=np.float64)

        for t in range(Tm2):
            E[:, t] = compute_wave_energy_batch(
                u_fem[:, t + 1],
                ut_array[:, t],
                out,
                c,
            )

        return E

    raise ValueError(f"Expected shape (T, N_nodes) or (B, T, N_nodes), got {u_array.shape}")