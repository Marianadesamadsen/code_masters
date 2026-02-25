
import numpy as np
import matplotlib.pyplot as plt

def randomly_rotate(points):
    """
    Apply a random rotation to a set of 3D points.
    """
    random_rotation_matrix = np.linalg.qr(np.random.randn(3, 3))[0]
    return np.dot(random_rotation_matrix, points)

def get_icosahedral_mesh():
    """
    Generate an icosahedral mesh.
    Returns:
        P (numpy.ndarray): 3x12 array of vertex positions.
        tri (numpy.ndarray): 20x3 array of triangle indices.
    """
    phi = (1 + np.sqrt(5)) / 2
    P = np.array([
        [-1,  phi,  0],
        [ 1,  phi,  0],
        [-1, -phi,  0],
        [ 1, -phi,  0],
        [ 0, -1,  phi],
        [ 0,  1,  phi], 
        [ 0, -1, -phi],
        [ 0,  1, -phi],
        [ phi,  0, -1],
        [ phi,  0,  1],
        [-phi,  0, -1],
        [-phi,  0,  1],
    ]).T

    # Normalize vertices to lie on the unit sphere
    P = P / np.linalg.norm(P, axis=0)

    # Center the vertices around the origin
    P = P - np.mean(P, axis=1, keepdims=True)

    # Apply a random rotation
    P = randomly_rotate(P)

    tri = np.array([
        [1, 12, 6],
        [1, 6, 2],
        [1, 2, 8],
        [1, 8, 11],
        [1, 11, 12],
        [2, 6, 10],
        [6, 12, 5],
        [12, 11, 3],
        [11, 8, 7],
        [8, 2, 9],
        [4, 10, 5],
        [4, 5, 3],
        [4, 3, 7],
        [4, 7, 9],
        [4, 9, 10],
        [5, 10, 6],
        [3, 5, 12],
        [7, 3, 11],
        [9, 7, 8],
        [10, 9, 2]
    ]) - 1  # Convert 1-based indexing to 0-based

    return P, tri


def plot_icosahedral_mesh(P, tri):
    """
    Plot an icosahedral mesh using Matplotlib.
    Args:
        P (numpy.ndarray): 3x12 array of vertex positions.
        tri (numpy.ndarray): 20x3 array of triangle indices.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create the trisurf plot
    ax.plot_trisurf(P[0, :], P[1, :], P[2, :], triangles=tri, edgecolor='k', linewidth=0.5, alpha=0.8)

    # Set the aspect ratio to be equal
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio for all axes

    # Set axis limits for better visualization
    limit = 1.2
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])
    ax.set_zlim([-limit, limit])

    # Turn off the grid and axis for a cleaner look (optional)
    ax.grid(False)
    ax.axis('off')

    plt.show()


def refine_mesh(Pin, triin):
    """
    Refine a triangular mesh by subdividing each triangle into smaller triangles.
    Args:
        Pin (numpy.ndarray): 3xN array of vertex positions.
        triin (numpy.ndarray): Mx3 array of triangle indices.

    Returns:
        P (numpy.ndarray): Refined 3xN array of vertex positions.
        tri (numpy.ndarray): Refined Mx3 array of triangle indices.
    """
    N = Pin.shape[1]  # Number of vertices in the original mesh

    # Step 1: Generate all edges for each triangle
    edges_1 = triin[:, [0, 1]]  # Edges between vertices 0 and 1 
    edges_2 = triin[:, [1, 2]]  # Edges between vertices 1 and 2
    edges_3 = triin[:, [2, 0]]  # Edges between vertices 2 and 0

    # Combine all edges and sort each edge to ensure consistent order
    edges = np.vstack([edges_1, edges_2, edges_3])
    edges_sorted = np.sort(edges, axis=1)

    # Step 2: Find unique edges and their indices
    edges_unique, unique_indices = np.unique(edges_sorted, axis=0, return_inverse=True)

    # Step 3: Create new vertices at the midpoints of unique edges
    Pnew = (Pin[:, edges_unique[:, 0]] + Pin[:, edges_unique[:, 1]]) / 2
    Pnew = Pnew / np.linalg.norm(Pnew, axis=0, keepdims=True)  # Normalize to unit sphere

    # Step 4: Map triangle edges to new vertices
    idx1 = unique_indices[:len(triin)]              # Indices for edges_1
    idx2 = unique_indices[len(triin):2*len(triin)]  # Indices for edges_2
    idx3 = unique_indices[2*len(triin):]            # Indices for edges_3

    # Step 5: Generate new triangles
    tri = np.vstack([
        np.column_stack([triin[:, 0], N + idx1, N + idx3]),
        np.column_stack([triin[:, 1], N + idx2, N + idx1]),
        np.column_stack([triin[:, 2], N + idx3, N + idx2]),
        np.column_stack([N + idx1, N + idx2, N + idx3])
    ])

    # Step 6: Combine original and new vertices 
    P = np.hstack([Pin, Pnew])

    return P, tri 
