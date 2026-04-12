"""Ricci flow on discrete surfaces.

High-level functional API that takes a mesh and target curvature
specification and returns optimised edge lengths.
"""

import os
import numpy as np
from dataclasses import dataclass, field
from typing import Union, Optional

from .mesh import TriangleMesh
from .metric import DiscreteMetric, CirclePackingMetric, _ricci_energy


@dataclass
class RicciFlowResult:
    """Result of a Ricci flow computation.

    Attributes:
        edge_lengths: dict mapping frozenset edge -> optimised length.
        curvature: (N,) realised Gaussian curvature per vertex.
        target_curvature: (N,) target curvature that was specified.
        scale_factor: global scaling factor from circle packing.
        history: convergence history (list of error values).
        circle_packing: the CirclePackingMetric after optimisation.
        mesh: the input mesh.
    """
    edge_lengths: dict
    curvature: np.ndarray
    target_curvature: np.ndarray
    scale_factor: float
    history: list = field(default_factory=list)
    circle_packing: CirclePackingMetric = None
    mesh: TriangleMesh = None


def _parse_target_curvature(mesh, target, fix_boundary):
    """Parse target curvature specification into a per-vertex array.

    Args:
        mesh: TriangleMesh
        target: float, ndarray, or path to CSV file.
            - float: uniform target for interior vertices; boundary inferred.
            - ndarray of length N: per-vertex target.
            - str: path to CSV with rows (vertex_id, curvature).
              Vertices with K > 2*pi are "free" (determined by Gauss-Bonnet).
        fix_boundary: if True, boundary curvature stays unchanged.

    Returns:
        (K, specified_verts, free_verts) where K is (N,) target array,
        specified_verts are vertices with specified curvature, and
        free_verts are vertices whose conformal factor can vary.
    """
    n = mesh.n_vertices
    K = np.full(n, 4 * np.pi)  # sentinel: > 2*pi means "unspecified"

    if isinstance(target, (int, float)):
        K[mesh.interior_vertices] = target
    elif isinstance(target, np.ndarray) and target.ndim == 1 and len(target) == n:
        K = target.copy()
    elif isinstance(target, str) and os.path.isfile(target):
        tc = np.loadtxt(target, delimiter=",")
        K[tc[:, 0].astype(int)] = tc[:, 1]
    else:
        raise ValueError(f"Cannot parse target_curvature: {target}")

    # Magic: K == 100 means keep initial curvature (handled later)
    # Unspecified (K > 2*pi): fill uniformly via Gauss-Bonnet
    free_K = K > 2 * np.pi
    if free_K.sum() > 0:
        specified_sum = K[~free_K].sum()
        uniform_K = (2 * mesh.euler_characteristic * np.pi - specified_sum) / free_K.sum()
        K[free_K] = uniform_K

    # Which vertices have target curvature specified
    if fix_boundary:
        specified_verts = mesh.interior_vertices
        free_verts = mesh.interior_vertices
    else:
        specified_verts = list(range(n))
        free_verts = list(range(n))

    # Override with CSV-specified vertices if available
    if isinstance(target, str):
        specified_verts = list(np.where(K < 2 * np.pi)[0])

    return K, specified_verts, free_verts


def ricci_flow(
    mesh: TriangleMesh,
    target_curvature: Union[float, np.ndarray, str] = 0.0,
    *,
    method: str = 'trf',
    scheme: str = 'combinatorial',
    fix_boundary: bool = False,
    boundary_constraint: str = 'edge',
    boundary_weight: float = 0.01,
    alpha: float = -1,
    lr: float = 0.1,
    gtol: float = 1e-6,
    verbose: int = 1,
) -> RicciFlowResult:
    """Run discrete Ricci flow on a triangle mesh.

    Finds edge lengths that produce the desired Gaussian curvature at each
    vertex, preserving the conformal class defined by the circle packing scheme.

    Args:
        mesh: input TriangleMesh.
        target_curvature: target Gaussian curvature specification.
            - float: uniform curvature for interior vertices.
            - ndarray: per-vertex curvature (values > 2*pi are "free").
            - str: path to CSV file with (vertex_id, curvature) rows.
        method: optimisation method.
            'sgd': gradient descent with line search.
            'newton': Newton's method with Hessian.
            'lm': Levenberg-Marquardt (scipy).
            'trf': Trust Region Reflective (scipy, default).
        scheme: circle packing scheme.
            'inversive', 'thurston', 'thurston2', 'combinatorial',
            a float value for constant eta, or a CSV file path.
        fix_boundary: if True, keep boundary radii unchanged.
        boundary_constraint: 'edge' to preserve boundary edge lengths,
            'radius' to preserve boundary radii.
        boundary_weight: weight for boundary constraints.
        alpha: Thurston scaling factor (negative for auto).
        lr: learning rate for sgd/newton methods.
        gtol: gradient tolerance for convergence.
        verbose: 0=silent, 1=summary, 2=progress.

    Returns:
        RicciFlowResult with optimised edge lengths, curvature, etc.
    """
    # Build initial metric
    g = DiscreteMetric(mesh, mesh.edge_lengths)

    # Parse target curvature
    K, specified_verts, free_verts = _parse_target_curvature(
        mesh, target_curvature, fix_boundary)

    # Handle K == 100 (magic number: keep initial curvature)
    keep_mask = (K == 100)
    K[keep_mask] = g.gaussian_curvature[keep_mask]

    if verbose > 0:
        chi = mesh.euler_characteristic
        print(f"V={mesh.n_vertices}, E={mesh.n_edges}, F={mesh.n_faces}, "
              f"boundary_V={len(mesh.boundary_vertices)}, chi={chi}")
        print(f"specified_verts={len(specified_verts)}, free_verts={len(free_verts)}")
        print(f"target total_K = {K.sum()/np.pi:.4f} pi")

    # Build circle packing metric
    cp = CirclePackingMetric.from_metric(g, scheme=scheme, alpha=alpha)
    if verbose > 0:
        print(f"scale_factor = {cp.scale_factor:.6f}")

    # Rescale initial boundary lengths
    init_boundary_len = np.array([mesh.edge_lengths[e] for e in mesh.boundary_edges])
    init_boundary_len /= cp.scale_factor
    init_u = cp.u.copy()

    # Run optimisation
    if method == 'sgd':
        history = cp.run_gradient_descent(
            K, free_verts, dt=lr, gtol=gtol,
            use_hessian=False, verbose=verbose)
    elif method == 'newton':
        history = cp.run_gradient_descent(
            K, free_verts, dt=lr, gtol=gtol,
            use_hessian=True, verbose=verbose)
    else:
        no_jac = (method == 'lm')
        target_l = init_boundary_len if boundary_constraint == 'edge' else None
        target_u = init_u if boundary_constraint == 'radius' else None
        cp.run_least_squares(
            K, specified_verts, free_verts,
            target_u=target_u, target_lengths=target_l,
            boundary_weight=boundary_weight, opt_target='conformal_factor',
            method=method, gtol=gtol, no_jacobian=no_jac, verbose=verbose)
        history = []

    # Collect results
    edge_lengths = {}
    for edge in mesh.edges:
        i, j = edge
        edge_lengths[edge] = cp.scale_factor * cp._l[i, j]

    curvature_mae = np.abs(cp.gaussian_curvature[specified_verts] - K[specified_verts]).mean()
    if verbose > 0:
        print(f"curvature MAE = {curvature_mae:.6e}")

    return RicciFlowResult(
        edge_lengths=edge_lengths,
        curvature=cp.gaussian_curvature.copy(),
        target_curvature=K,
        scale_factor=cp.scale_factor,
        history=history,
        circle_packing=cp,
        mesh=mesh,
    )
