"""Visualisation utilities for triangle meshes and convergence plots."""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3


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
    # Accept TriangleMesh directly
    if hasattr(vertices, 'vertices') and hasattr(vertices, 'faces'):
        mesh = vertices
        verts = mesh.vertices
        face_list = [sorted(f) for f in mesh.faces]
    else:
        verts = np.asarray(vertices)
        face_list = [list(f) for f in faces]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    m, M = np.min(verts), np.max(verts)
    margin = 0.2 * (M - m)
    ax.set_xlim([m - margin, M + margin])
    ax.set_ylim([m - margin, M + margin])
    ax.set_zlim([m - margin, M + margin])
    ax.set_axis_off()

    for f in face_list:
        tri = [verts[f[0]], verts[f[1]], verts[f[2]]]
        face_poly = a3.art3d.Poly3DCollection([tri])
        face_poly.set_edgecolor('k')
        face_poly.set_linewidth(0.3)
        face_poly.set_alpha(0.9)
        ax.add_collection3d(face_poly)

    if title:
        ax.set_title(title)
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
    return fig


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

    verts = mesh.vertices
    face_list = [sorted(f) for f in mesh.faces]

    if curvature is None:
        g = DiscreteMetric(mesh, mesh.edge_lengths)
        curvature = g.gaussian_curvature

    if vmin is None:
        vmin = curvature.min()
    if vmax is None:
        vmax = curvature.max()

    cm = plt.get_cmap(cmap)
    if vmax - vmin < 1e-15:
        colours = np.full((len(verts), 4), 0.5)
    else:
        colours = cm((curvature - vmin) / (vmax - vmin))

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    m, M = np.min(verts), np.max(verts)
    margin = 0.2 * (M - m)
    ax.set_xlim([m - margin, M + margin])
    ax.set_ylim([m - margin, M + margin])
    ax.set_zlim([m - margin, M + margin])
    ax.set_axis_off()

    for f in face_list:
        tri = [verts[f[0]], verts[f[1]], verts[f[2]]]
        avg_c = np.mean([colours[f[0]], colours[f[1]], colours[f[2]]], axis=0)
        face_poly = a3.art3d.Poly3DCollection([tri])
        face_poly.set_facecolor(avg_c)
        face_poly.set_edgecolor('k')
        face_poly.set_linewidth(0.2)
        ax.add_collection3d(face_poly)

    sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.5, label='Gaussian curvature')

    if title:
        ax.set_title(title)
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=200, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()
    return fig


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
    import seaborn as sns

    fig, ax = plt.subplots(figsize=figsize)
    if kind == 'violin':
        parts = ax.violinplot(values_list, showmedians=True)
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
