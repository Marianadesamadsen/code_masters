import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm, colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import torch

def triangles_from_m2m_edge_index(edge_index: np.ndarray) -> np.ndarray:
    """
    Reconstruct triangles from an undirected graph by finding 3-cliques.

    edge_index: (2, E) array of directed edges (likely contains both directions).
    Returns: (T, 3) int array of triangle vertex indices.
    """
    if edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, E)")

    src = edge_index[0]
    dst = edge_index[1]

    # Build undirected adjacency sets
    N = int(max(src.max(), dst.max())) + 1
    nbrs = [set() for _ in range(N)]
    for u, v in zip(src, dst):
        if u == v:
            continue
        nbrs[int(u)].add(int(v))
        nbrs[int(v)].add(int(u))

    # Find triangles u < v < w with edges (u,v), (u,w), (v,w)
    triangles = []
    for u in range(N):
        nu = sorted(nbrs[u])
        # Intersect neighbor sets to find common neighbors
        for i, v in enumerate(nu):
            if v <= u:
                continue
            common = nbrs[u].intersection(nbrs[v])
            for w in common:
                if w > v:  # enforce ordering u < v < w
                    triangles.append((u, v, w))

    if not triangles:
        raise RuntimeError(
            "No triangles found from the graph edges. "
            "If this is not a triangular mesh graph, clique-triangle reconstruction won't work."
        )

    return np.asarray(triangles, dtype=np.int64)


class DataPlotter:
    def __init__(self, nc_path: str, graph_dir: str):
        self.data = xr.open_dataset(nc_path)

        # positions from dataset
        self.P = np.column_stack([self.data["x"].values,
                                  self.data["y"].values,
                                  self.data["z"].values]).astype(np.float64)  # (N,3)

        # load graph m2m edges
        m2m_path = os.path.join(graph_dir, "m2m_edge_index.pt")
        m2m = torch.load(m2m_path, map_location="cpu")

        # neural-lam stores m2m_edge_index.pt as a list for multi-level graphs; handle both
        if isinstance(m2m, list):
            edge_index = m2m[0].detach().cpu().numpy()
        else:
            edge_index = m2m.detach().cpu().numpy()

        self.tri = triangles_from_m2m_edge_index(edge_index)  # (T,3)

    def animate_sphere(self,
                       R=None,
                       out_path=None,
                       fps=10,
                       interval=100,
                       cmap_name="viridis"):

        P = self.P  # (N,3)
        tri = np.asarray(self.tri, dtype=int)
        u_data = self.data["u"].values
        time_steps = self.data["time"].values

        if R is None:
            # infer from max norm of points
            R = float(np.nanmax(np.linalg.norm(P, axis=1)))

        # stable colormap range
        u_min = float(np.nanmin(u_data))
        u_max = float(np.nanmax(u_data))
        norm = colors.Normalize(vmin=u_min, vmax=u_max)
        cmap = cm.get_cmap(cmap_name)

        # Build triangle vertex coordinates (T,3,3)
        tri_vertices = P[tri]  # each row is (v0,v1,v2) each v is (x,y,z)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        poly = Poly3DCollection(
            tri_vertices,
            edgecolor="k",
            linewidths=0.2,
            alpha=0.95
        )
        ax.add_collection3d(poly)

        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.1, shrink=0.8)
        cbar.set_label("u")

        ax.set_xlim(-R, R)
        ax.set_ylim(-R, R)
        ax.set_zlim(-R, R)
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlabel("Distance x (m)")
        ax.set_ylabel("Distance y (m)")
        ax.set_zlabel("Distance z (m)")

        def update(frame):
            u = u_data[frame]              # (N,)
            u_face = u[tri].mean(axis=1)   # (T,)
            poly.set_facecolor(cmap(norm(u_face)))
            t = time_steps[frame]
            t_sec = t #/ np.timedelta64(1, "s")
            ax.set_title(f"Wave equation on sphere (t = {float(t_sec):.6f} s)")
            return [poly]

        anim = animation.FuncAnimation(
            fig,
            update,
            frames=len(time_steps),
            interval=interval,
            blit=False
        )

        if out_path:
            anim.save(out_path, writer="pillow", fps=fps)
            plt.close(fig) 

        return anim