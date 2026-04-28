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


def test_field(out, t=0.0, omega=1.0):
    x = out["x3D"]
    y = out["y3D"]

    u = x * np.cos(omega * t) + y * np.sin(omega * t)
    ut = -omega * x * np.sin(omega * t) + omega * y * np.cos(omega * t)

    return u, ut

def exact_l1_wave(out, t, c=1.0, R=1.0):
    x = out["x3D"]

    omega = c * np.sqrt(2.0) / R

    u = x * np.cos(omega * t)
    ut = -omega * x * np.sin(omega * t)

    return u, ut


def exact_l1_energy(c=1.0, R=1.0):
    return 4.0 * np.pi * c**2 * R**2 / 3.0

if __name__ == "__main__":
    out = surface_mass_integration(N=6, generation=3, R=1.0)

    u = out["x3D"] * 0.0
    ut = out["x3D"] * 0.0

    # Example field
    u = out["x3D"]   # just a test function
    ut = 0.0 * u

    E = compute_wave_energy(u, ut, out, c=1.0)

    print(E)
    out = surface_mass_integration(N=6, generation=3, R=1.0)

    times = np.linspace(0, 2*np.pi, 20)
    energies = []

    for t in times:
        u, ut = test_field(out, t)
        E = compute_wave_energy(u, ut, out, c=1.0)
        energies.append(E)

    energies = np.array(energies)

    print("Energy min:", energies.min())
    print("Energy max:", energies.max())
    print("Relative variation:", (energies.max() - energies.min()) / energies.mean())

    out = surface_mass_integration(N=6, generation=3, R=1.0)

    c = 1.0
    R = 1.0

    E_exact = exact_l1_energy(c=c, R=R)

    times = np.linspace(0, 2*np.pi, 20)
    E_num = []

    for t in times:
        u, ut = exact_l1_wave(out, t, c=c, R=R)
        E_num.append(compute_wave_energy(u, ut, out, c=c))

    E_num = np.array(E_num)

    print("Exact energy:", E_exact)
    print("Numerical min:", E_num.min())
    print("Numerical max:", E_num.max())
    print("Mean numerical:", E_num.mean())
    print("Relative error:", abs(E_num.mean() - E_exact) / E_exact)
    print("Relative variation:", (E_num.max() - E_num.min()) / E_num.mean())
