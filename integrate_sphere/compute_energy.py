import numpy as np
import trimesh
from scipy.special import roots_legendre


def create_mesh(generation, R=1.0):
    mesh = trimesh.creation.icosphere(subdivisions=generation, radius=R)
    P = mesh.vertices.T
    tri = mesh.faces.astype(int)
    return P, tri


def check_orientation(P, tri):
    tri = tri.copy()
    num_flipped = 0

    for k in range(tri.shape[0]):
        v1 = P[:, tri[k, 0]]
        v2 = P[:, tri[k, 1]]
        v3 = P[:, tri[k, 2]]

        n_tri = np.cross(v2 - v1, v3 - v1)
        centroid = (v1 + v2 + v3) / 3.0

        if np.dot(n_tri, centroid) < 0:
            tri[k, [1, 2]] = tri[k, [2, 1]]
            num_flipped += 1

    print(f"Corrected {num_flipped} triangles with incorrect orientation.")
    return tri


def reference_nodes_triangle(N):
    r_list = []
    s_list = []

    for j in range(N + 1):
        for i in range(N + 1 - j):
            r = -1.0 + 2.0 * i / N
            s = -1.0 + 2.0 * j / N
            r_list.append(r)
            s_list.append(s)

    return np.array(r_list), np.array(s_list)


def monomial_powers_triangle(N):
    powers = []
    for total in range(N + 1):
        for a in range(total + 1):
            b = total - a
            powers.append((a, b))
    return powers


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

    print()
    print("Check mass matrix and jacobians by integrating at the sphere surface")
    print(f"Exact area is 4*pi*R^2 : {exact_area:.8f}")
    print(f"Computed total area    : {total_area:.8f}")
    print(f"Absolute error         : {abs(total_area - exact_area):.8e}")

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

def compute_wave_energy(u, ut, out, c=1.0):
    """
    Compute discrete wave energy on the sphere.

    Parameters
    ----------
    u : ndarray
        Displacement values, shape (Np, K).
    ut : ndarray
        Time derivative values, shape (Np, K).
    out : dict
        Output from surface_mass_integration.
    c : float
        Wave speed.

    Returns
    -------
    energy : float
        Discrete wave energy.
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

    K = u.shape[1]

    energy = 0.0

    for k in range(K):
        uk = u[:, k]
        utk = ut[:, k]

        # Reference derivatives
        ur = Dr @ uk
        us = Ds @ uk

        # Surface gradient in Cartesian coordinates
        ux = rx2D[:, k] * ur + sx2D[:, k] * us
        uy = ry2D[:, k] * ur + sy2D[:, k] * us
        uz = rz2D[:, k] * ur + sz2D[:, k] * us

        grad_u_sq = ux**2 + uy**2 + uz**2

        # Curved-element mass matrix
        Mk = np.diag(J2D[:, k]) @ MassMatrix

        kinetic = utk.T @ Mk @ utk
        potential = grad_u_sq.T @ Mk @ np.ones_like(grad_u_sq)

        energy += 0.5 * (kinetic + c**2 * potential)

    return energy

def gnn_nodes_to_fem(u_node, out):
    """
    Convert GNN nodal values on out["P"].T to FEM element values.

    u_node: shape (N_nodes,)
    returns: shape (Np, K)
    """
    tri = out["tri"]
    r = out["r"]
    s = out["s"]

    # These are the interpolation weights from the triangle vertices
    lam1 = -(r + s) / 2.0
    lam2 = (1.0 + r) / 2.0
    lam3 = (1.0 + s) / 2.0

    u1 = u_node[tri[:, 0]]  # values at vertex 1 of each triangle, shape (K,)
    u2 = u_node[tri[:, 1]]
    u3 = u_node[tri[:, 2]]

    u_fem = (
        lam1[:, None] * u1[None, :]
        + lam2[:, None] * u2[None, :]
        + lam3[:, None] * u3[None, :]
    )

    return u_fem

def gnn_series_to_fem(u_array_node, out):
    """
    u_array_node: shape (time, N_nodes)
    returns: shape (time, Np, K)
    """
    return np.stack([gnn_nodes_to_fem(u_t, out) for u_t in u_array_node], axis=0)

def compute_energy_over_time(u_array,generation,R=1,c=1,N=6,dt=1):

    out = surface_mass_integration(N=N, generation=generation, R=R)
    u_fem = gnn_series_to_fem(u_array, out)
    ut_array = (u_fem[2:] - u_fem[:-2]) / (2 * dt)

    E = []
    for idx in range(len(ut_array)):
        Et = compute_wave_energy(u_fem[idx+1], ut_array[idx],out,c)
        E.append(Et)
    
    return np.array(E)

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

    x = out["x3D"]

    omega = c * np.sqrt(2.0) / R

    times = np.arange(num_times) * dt

    u_true_list = []
    u_pred_list = []

    for t in times:
        # Exact l=1 wave: u = x cos(omega t)
        u_true_t = x * np.cos(omega * t)

        # Fake prediction: slightly damped and slightly noisy
        damping = 1.0 - 0.0005 * t
        noise = 0.001 * np.random.randn(*u_true_t.shape)

        u_pred_t = damping * u_true_t + noise

        u_true_list.append(u_true_t)
        u_pred_list.append(u_pred_t)

    u_true = np.array(u_true_list)  # shape (time, Np, K)
    u_pred = np.array(u_pred_list)  # shape (time, Np, K)

    print("u_true shape:", u_true.shape)
    print("u_pred shape:", u_pred.shape)

    # Compute energies
    E_true = compute_energy_over_time(
        u_true,
        generation=generation,
        R=R,
        c=c,
        N=N,
        dt=dt,
    )

    E_pred = compute_energy_over_time(
        u_pred,
        generation=generation,
        R=R,
        c=c,
        N=N,
        dt=dt,
    )

    # Exact continuous energy for l=1 wave
    E_exact = 4.0 * np.pi * c**2 * R**2 / 3.0

    print()
    print("Exact continuous energy:", E_exact)

    print()
    print("True energy:")
    print("min:", E_true.min())
    print("max:", E_true.max())
    print("mean:", E_true.mean())
    print("relative variation:", (E_true.max() - E_true.min()) / E_true.mean())
    print("relative mean error:", abs(E_true.mean() - E_exact) / E_exact)

    print()
    print("Pred energy:")
    print("min:", E_pred.min())
    print("max:", E_pred.max())
    print("mean:", E_pred.mean())
    print("relative variation:", (E_pred.max() - E_pred.min()) / E_pred.mean())
    print("relative mean error:", abs(E_pred.mean() - E_exact) / E_exact)

    print()
    print("Energy error pred - true:")
    energy_error = E_pred - E_true
    print("mean error:", energy_error.mean())
    print("max abs error:", np.max(np.abs(energy_error)))