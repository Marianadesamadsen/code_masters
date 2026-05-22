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
    tri[wrong_orientation] = tri[wrong_orientation][:, [0, 2, 1]]
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

    i = np.arange(N + 1)
    j = np.arange(N + 1)

    I, J = np.meshgrid(i, j, indexing="ij")

    mask = (I + J) <= N

    L1 = I[mask] / N
    L3 = J[mask] / N
    L2 = 1.0 - L1 - L3

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


def reference_nodes_triangle(N):
    x, y = nodes_2d(N)
    r, s = xytors(x, y)
    return r, s


def monomial_powers_triangle(N):
    powers = []
    for total in range(N + 1):
        a = np.arange(total + 1)
        b = total - a
        powers.append(np.stack([a, b], axis=1))
    return np.vstack(powers)


def vandermonde_triangle(r, s, powers):
    a, b = powers.T
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
    """
    Tensor-product Gauss rule mapped to reference triangle:
        -1 <= s <= 1
        -1 <= r <= -s
    """
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

    diff = x[:, None] - x[None, :]

    # Compute barycentric weights
    w = 1.0 / np.prod(diff + np.eye(n), axis=1)

    # Initialize D
    D = np.zeros((n, n))

    # Only compute off-diagonal entries
    mask = ~np.eye(n, dtype=bool)

    D[mask] = (w[None, :] / w[:, None])[mask] / diff[mask]

    # Diagonal
    np.fill_diagonal(D, -np.sum(D, axis=1))

    return D


def compute_sJ_curved_face(x3D, y3D, z3D, Fmask):
    Nfaces = 3
    Nfp = Fmask.shape[0]
    K = x3D.shape[1]

    t_nodes = np.linspace(-1.0, 1.0, Nfp)
    Dr1D = derivative_matrix_1d(t_nodes)

    tq, wq = roots_legendre(Nfp)

    # Extract face nodes for all faces and all elements
    # shape: (Nfp, Nfaces, K)
    xf = x3D[Fmask, :]
    yf = y3D[Fmask, :]
    zf = z3D[Fmask, :]

    # Differentiate along face-node direction
    # Dr1D: (Nfp, Nfp)
    # xf:   (Nfp, Nfaces, K)
    dxf = np.einsum("ij,jfk->ifk", Dr1D, xf)
    dyf = np.einsum("ij,jfk->ifk", Dr1D, yf)
    dzf = np.einsum("ij,jfk->ifk", Dr1D, zf)

    # Arc-length density along each face
    # shape: (Nfp, Nfaces, K)
    ds = np.sqrt(dxf**2 + dyf**2 + dzf**2)

    # Integrate each face length using quadrature weights
    # shape: (Nfaces, K)
    L = np.einsum("i,ifk->fk", wq, ds)

    # denominator = one @ M1D @ one = sum(wq)
    denom = np.sum(wq)

    # Constant surface Jacobian per face
    # shape: (Nfaces, K)
    sj_face = L / denom

    # Repeat over face nodes
    # shape: (Nfp, Nfaces, K)
    sJ_faces = np.ones((Nfp, 1, 1)) * sj_face[None, :, :]

    # Reshape to original output shape: (Nfp * Nfaces, K)
    sJ = sJ_faces.reshape(Nfp * Nfaces, K, order="F")

    return sJ

def surface_mass_integration(N=6, generation=3, R=1.0):
    P, tri = create_mesh(generation, R=R)
    tri = check_orientation(P, tri)

    K = tri.shape[0]

    r, s, Dr, Ds, MassMatrix, Fmask, V = get_reference_element(N)
    Np = len(r)

    # Triangle vertices, shape (3, K)
    v1 = P[:, tri[:, 0]]
    v2 = P[:, tri[:, 1]]
    v3 = P[:, tri[:, 2]]

    # Reference coordinates as columns
    r_col = r[:, None]
    s_col = s[:, None]

    # Affine triangle mapping, shape (Np, K)
    x = 0.5 * (-(r_col + s_col) * v1[0][None, :]
               + (1.0 + r_col) * v2[0][None, :]
               + (1.0 + s_col) * v3[0][None, :])

    y = 0.5 * (-(r_col + s_col) * v1[1][None, :]
               + (1.0 + r_col) * v2[1][None, :]
               + (1.0 + s_col) * v3[1][None, :])

    z = 0.5 * (-(r_col + s_col) * v1[2][None, :]
               + (1.0 + r_col) * v2[2][None, :]
               + (1.0 + s_col) * v3[2][None, :])

    # Project to sphere
    norm = np.sqrt(x**2 + y**2 + z**2)

    x3D = R * x / norm
    y3D = R * y / norm
    z3D = R * z / norm

    # Derivatives, shape (Np, K)
    xr = Dr @ x3D
    xs = Ds @ x3D

    yr = Dr @ y3D
    ys = Ds @ y3D

    zr = Dr @ z3D
    zs = Ds @ z3D

    # Stack into vectors, shape (Np, K, 3)
    xr_vec = np.stack([xr, yr, zr], axis=2)
    xs_vec = np.stack([xs, ys, zs], axis=2)

    # Surface normal and Jacobian
    n_vec = np.cross(xr_vec, xs_vec, axis=2)
    J2D = np.linalg.norm(n_vec, axis=2)

    n_unit = n_vec / J2D[:, :, None]

    # Contravariant basis vectors
    ar = np.cross(xs_vec, n_unit, axis=2) / J2D[:, :, None]
    a_s = np.cross(n_unit, xr_vec, axis=2) / J2D[:, :, None]

    rx2D = ar[:, :, 0]
    ry2D = ar[:, :, 1]
    rz2D = ar[:, :, 2]

    sx2D = a_s[:, :, 0]
    sy2D = a_s[:, :, 1]
    sz2D = a_s[:, :, 2]

    # Total area
    total_area = np.sum(J2D * MassMatrix.sum(axis=1)[:, None])

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

    # Reference derivatives
    # u:  (B, Np, K)
    # Dr: (Np, Np)
    # ur: (B, Np, K)
    ur = np.einsum("bjk,ij->bik", u, Dr)
    us = np.einsum("bjk,ij->bik", u, Ds)

    # Surface gradient
    ux = rx2D[None, :, :] * ur + sx2D[None, :, :] * us
    uy = ry2D[None, :, :] * ur + sy2D[None, :, :] * us
    uz = rz2D[None, :, :] * ur + sz2D[None, :, :] * us

    grad_u_sq = ux**2 + uy**2 + uz**2  # (B, Np, K)

    # Element mass matrices:
    # Mk[i,j,k] = J2D[i,k] * MassMatrix[i,j]
    Mk = J2D[:, None, :] * MassMatrix[:, :, None]  # (Np, Np, K)

    # Kinetic energy over all elements
    kinetic = np.einsum("bik,ijk,bjk->bk", ut, Mk, ut)

    # Potential energy over all elements
    ones = np.ones(Np)
    potential = np.einsum("bik,ijk,j->bk", grad_u_sq, Mk, ones)

    # Sum over elements
    energy = 0.5 * np.sum(kinetic + c**2 * potential, axis=1)

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

def compute_energy_over_time(u_array, generation=4, R=1, c=1, N=6, dt=1, out=None, ut_order=4):
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

        if ut_order == 2:
            ut_array = (u_fem[2:] - u_fem[:-2]) / (2 * dt)
            u_mid = u_fem[1:-1]
        elif ut_order == 4:
            ut_array = (
                -u_fem[4:]
                + 8 * u_fem[3:-1]
                - 8 * u_fem[1:-3]
                + u_fem[:-4]
            ) / (12 * dt)
            u_mid = u_fem[2:-2]
        elif ut_order == 6:
            ut_array = (
                u_fem[:-6]
                - 9 * u_fem[1:-5]
                + 45 * u_fem[2:-4]
                - 45 * u_fem[4:-2]
                + 9 * u_fem[5:-1]
                - u_fem[6:]
            ) / (60 * dt)
            u_mid = u_fem[3:-3]
        elif ut_order == 8:
            ut_array = (
                -u_fem[:-8]
                + 8 * u_fem[ 1:-7]
                - 28 * u_fem[2:-6]
                + 56 * u_fem[3:-5]
                - 56 * u_fem[5:-3]
                + 28 * u_fem[6:-2]
                - 8 * u_fem[7:-1]
                + u_fem[ 8:]
            ) / (280 * dt)

            u_mid = u_fem[4:-4]

        # Treat time as batch
        E = compute_wave_energy_batch(
            u_mid,   # (T-2, Np, K)
            ut_array,      # (T-2, Np, K)
            out,
            c,
        )

        return E

    if u_array.ndim == 3:
   
        # u_fem shape: (B, T, Np, K)
        if ut_order == 2:
            ut_array = (u_fem[:,2:] - u_fem[:,:-2]) / (2 * dt)
            u_mid = u_fem[:,1:-1]
        elif ut_order == 4:
            ut_array = (
                -u_fem[:, 4:]
                + 8 * u_fem[:,3:-1]
                - 8 * u_fem[:,1:-3]
                + u_fem[:,:-4]
            ) / (12 * dt)
            u_mid = u_fem[:,2:-2]
        elif ut_order == 6:
            ut_array = (
                u_fem[:, :-6]
                - 9 * u_fem[:,1:-5]
                + 45 * u_fem[:,2:-4]
                - 45 * u_fem[:,4:-2]
                + 9 * u_fem[:,5:-1]
                - u_fem[:,6:]
            ) / (60 * dt)
            u_mid = u_fem[:,3:-3]
        elif ut_order == 8:
            ut_array = (
                -u_fem[:, :-8]
                + 8 * u_fem[:, 1:-7]
                - 28 * u_fem[:, 2:-6]
                + 56 * u_fem[:, 3:-5]
                - 56 * u_fem[:, 5:-3]
                + 28 * u_fem[:, 6:-2]
                - 8 * u_fem[:, 7:-1]
                + u_fem[:, 8:]
            ) / (280 * dt)

            u_mid = u_fem[:, 4:-4]

        B, Tm2, Np, K = ut_array.shape

        # Flatten batch and time into one axis
        u_flat = u_mid.reshape(B * Tm2, Np, K)
        ut_flat = ut_array.reshape(B * Tm2, Np, K)

        E_flat = compute_wave_energy_batch(
            u_flat,
            ut_flat,
            out,
            c,
        )

        return E_flat.reshape(B, Tm2)

    raise ValueError(
        f"Expected shape (T, N_nodes) or (B, T, N_nodes), got {u_array.shape}"
    )

if __name__ == "__main__":

    # Parameters
    N = 6
    generation = 3
    R = 1.0
    c = 1.0

    dt = 0.01
    num_times = 200

    # Build geometry once so we can generate test data
    out = surface_mass_integration(N=N, generation=generation, R=R)
