"""Visualisation utilities for triangle meshes and convergence plots."""

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3


def _mesh_figure(vertices, triangles, colours, figsize, title):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    collection = a3.art3d.Poly3DCollection(
        vertices[triangles], facecolors=colours, edgecolors='#273641', linewidths=0.3)
    ax.add_collection3d(collection)
    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    center = (lower + upper) / 2
    radius = 0.52 * np.max(upper - lower) or 1.0
    ax.set(xlim=(center[0]-radius, center[0]+radius),
           ylim=(center[1]-radius, center[1]+radius),
           zlim=(center[2]-radius, center[2]+radius))
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    return fig, ax


def _finish_plot(fig, path, show):
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_mesh(vertices, faces=None, *, path=None, show=False, title=None,
              cmap='cividis', figsize=(8, 6)):
    """Plot a 3D triangle mesh.

    Args:
        vertices: (N, 3) array, or a TriangleMesh instance.
        faces: list of index tuples (required if vertices is an array).
        path: if given, save figure to this path.
        show: if True, display interactively.
        title: optional plot title.
        cmap: matplotlib colourmap name.
        figsize: figure size.
    """
    if hasattr(vertices, 'faces'):
        faces = getattr(vertices, 'triangles', vertices.faces)
        vertices = vertices.vertices
    verts = np.asarray(vertices)
    triangles = faces if isinstance(faces, np.ndarray) else np.asarray([list(f) for f in faces], dtype=np.intp)
    normals = np.cross(verts[triangles[:, 1]] - verts[triangles[:, 0]],
                       verts[triangles[:, 2]] - verts[triangles[:, 0]])
    lengths = np.linalg.norm(normals, axis=1)
    light = np.abs(normals @ np.array([0.36, -0.48, 0.8]))
    shade = np.divide(light, lengths, out=np.zeros_like(light), where=lengths > 0)
    fig, _ = _mesh_figure(verts, triangles, plt.get_cmap(cmap)(0.25 + 0.65*shade), figsize, title)
    return _finish_plot(fig, path, show)


def plot_curvature(mesh, curvature=None, *, vmin=None, vmax=None,
                   cmap='bwr', path=None, show=False, title=None, figsize=(8, 6)):
    """Plot mesh coloured by per-vertex Gaussian curvature.

    Args:
        mesh: TriangleMesh (or object with .vertices, .faces).
        curvature: (N,) array. If None, computed from mesh geometry.
        vmin, vmax: colour scale limits.
        cmap: colourmap name.
        path: save path.
        show: display interactively.
    """
    from .metric import DiscreteMetric

    if curvature is None:
        curvature = DiscreteMetric(mesh, mesh.edge_lengths).gaussian_curvature
    curvature = np.asarray(curvature, dtype=float)
    if curvature.shape != (len(mesh.vertices),):
        raise ValueError("curvature must contain one value per vertex")
    norm = plt.Normalize(vmin=curvature.min() if vmin is None else vmin,
                         vmax=curvature.max() if vmax is None else vmax)
    triangles = mesh.triangles if hasattr(mesh, 'triangles') else np.asarray([list(f) for f in mesh.faces])
    colours = plt.get_cmap(cmap)(norm(curvature[triangles].mean(axis=1)))
    fig, ax = _mesh_figure(np.asarray(mesh.vertices), triangles, colours, figsize, title)
    fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax,
                 shrink=0.5, label='Angle defect (rad)')
    return _finish_plot(fig, path, show)


def plot_convergence(history, *, path=None, show=False, title=None, figsize=(8, 4)):
    """Plot convergence history on a log scale.

    Args:
        history: list of loss values per iteration.
        path: save path.
        show: display interactively.
        title: plot title.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.semilogy(history)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title(title or 'Convergence')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
    return fig


def plot_comparison(values_list, labels=None, *, kind='violin',
                    path=None, show=False, title=None, figsize=(8, 4)):
    """Compare distributions via violin or box plot.

    Args:
        values_list: list of arrays to compare.
        labels: list of labels for each distribution.
        kind: 'violin' or 'box'.
        path: save path.
        show: display interactively.
    """
    fig, ax = plt.subplots(figsize=figsize)
    if kind == 'violin':
        ax.violinplot(values_list, showmedians=True)
    else:
        ax.boxplot(values_list, labels=labels)
    if labels and kind == 'violin':
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels)
    if title:
        ax.set_title(title)
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
    return fig
