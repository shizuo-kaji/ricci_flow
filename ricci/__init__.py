"""Discrete Ricci flow for metric deformation on triangle meshes.

A package for designing surfaces with prescribed Gaussian curvature
via circle packing Ricci flow and metric embedding.

Typical usage::

    from ricci import TriangleMesh, ricci_flow, embed

    mesh = TriangleMesh.from_ply('dome.ply')
    result = ricci_flow(mesh, target_curvature=0.1, fix_boundary=True)
    embedded = embed(mesh, result.edge_lengths)
    embedded.mesh.save_ply('dome_result.ply')
"""

from .mesh import TriangleMesh
from .metric import DiscreteMetric, CirclePackingMetric
from .ricci import ricci_flow, RicciFlowResult
from .embed import embed, EmbedResult
from .curvature_flow import curvature_flow, CurvatureFlowResult
from .viz import plot_mesh, plot_curvature, plot_convergence, plot_comparison

__all__ = [
    'TriangleMesh',
    'DiscreteMetric',
    'CirclePackingMetric',
    'ricci_flow',
    'RicciFlowResult',
    'embed',
    'EmbedResult',
    'curvature_flow',
    'CurvatureFlowResult',
    'plot_mesh',
    'plot_curvature',
    'plot_convergence',
    'plot_comparison',
]
