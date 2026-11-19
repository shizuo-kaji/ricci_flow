"""Ricci flow on discrete surfaces.

High-level functional API that takes a mesh and target curvature
specification and returns optimised edge lengths.
"""

import os
import numpy as np
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Callable, Optional, Union

from .mesh import TriangleMesh
from .metric import DiscreteMetric, CirclePackingMetric
from .shape_from_metric import shape_from_metric
from ._embed_common import EmbedResult


@dataclass
class RicciFlowPreview:
    """One reconstructed metric at an accepted flow iteration."""

    iteration: int
    curvature_error: float
    edge_lengths: dict
    shape: EmbedResult
    curvature: np.ndarray


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
        previews: retained snapshots when previewing without a callback.
        skipped_preview_iterations: intermediate states without valid triangles.
    """
    edge_lengths: dict
    curvature: np.ndarray
    target_curvature: np.ndarray
    scale_factor: float
    history: list = field(default_factory=list)
    circle_packing: CirclePackingMetric = None
    mesh: TriangleMesh = None
    previews: list = field(default_factory=list)
    skipped_preview_iterations: list = field(default_factory=list)


def _parse_target_curvature(mesh, target, fix_boundary, initial_curvature=None):
    """Expand scalar, vertex-array or CSV targets and fill free angle defects."""
    n = mesh.n_vertices
    K = np.full(n, 4 * np.pi)
    if isinstance(target, Real):
        K[mesh.interior_vertices] = target
    elif isinstance(target, (str, os.PathLike)):
        rows = np.loadtxt(target, delimiter=",", ndmin=2)
        if rows.shape[1] != 2:
            raise ValueError("curvature CSV rows must contain vertex_id,curvature")
        ids = rows[:, 0]
        if not np.all(np.isfinite(ids) & (ids == np.floor(ids)) & (ids >= 0) & (ids < n)):
            raise ValueError("curvature CSV contains an invalid vertex_id")
        K[ids.astype(np.intp)] = rows[:, 1]
    else:
        K = np.array(target, dtype=np.float64, copy=True)
        if K.shape != (n,):
            raise ValueError(f"target_curvature must contain {n} values")
    if not np.isfinite(K).all():
        raise ValueError("target_curvature must be finite")
    keep = K == 100
    if np.any(keep):
        if initial_curvature is None:
            initial_curvature = DiscreteMetric(mesh, mesh.edge_lengths).gaussian_curvature
        K[keep] = initial_curvature[keep]
    free = K > 2 * np.pi
    if np.any(free):
        # Each connected component has its own Gauss-Bonnet constraint.
        if initial_curvature is None:
            initial_curvature = DiscreteMetric(mesh, mesh.edge_lengths).gaussian_curvature
        for indices in mesh.connected_components:
            missing = free[indices]
            if np.any(missing):
                K[indices[missing]] = (
                    initial_curvature[indices].sum() - K[indices[~missing]].sum()
                ) / missing.sum()
    vertices = mesh.interior_vertices if fix_boundary else list(range(n))
    return K, vertices, vertices


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
    max_iter: Optional[int] = None,
    preview_every: int = 0,
    on_preview: Optional[Callable[[RicciFlowPreview], None]] = None,
    preview_options: Optional[dict] = None,
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
        fix_boundary: prescribe interior curvature only. SGD/Newton also
            hold boundary radii fixed. TRF/LM use soft boundary constraints.
        boundary_constraint: 'edge' to preserve boundary edge lengths,
            'radius' to preserve boundary radii.
        boundary_weight: weight for boundary constraints.
        alpha: Thurston scaling factor (negative for auto).
        lr: learning rate for sgd/newton methods.
        gtol: gradient tolerance for convergence.
        verbose: 0=silent, 1=summary, 2=progress.
        max_iter: iteration budget for gradient/Newton flow, or maximum SciPy
            function evaluations for TRF/LM. None keeps the solver default.
        preview_every: reconstruct every N accepted steps and at the final metric.
            Zero disables all shape reconstruction.
        on_preview: receive each RicciFlowPreview during the solve. Without a
            callback, enabled snapshots are collected in result.previews.
        preview_options: keyword arguments for shape_from_metric. Each preview
            after the first starts from the preceding reconstructed shape.

    Returns:
        RicciFlowResult with optimised edge lengths, curvature, etc.
    """
    if isinstance(preview_every, bool) or not isinstance(preview_every, Integral) or preview_every < 0:
        raise ValueError("preview_every must be a non-negative integer")
    if on_preview is not None and (not callable(on_preview) or not preview_every):
        raise ValueError("on_preview requires a callable and a positive preview_every")
    if max_iter is not None and (not isinstance(max_iter, Integral) or max_iter < 1):
        raise ValueError("max_iter must be a positive integer")
    if not np.isfinite(gtol) or gtol <= 0:
        raise ValueError("gtol must be positive and finite")
    if method not in ('sgd', 'newton', 'trf', 'lm'):
        raise ValueError(f"Unknown flow method: {method}")
    if method in ('sgd', 'newton') and (not np.isfinite(lr) or lr <= 0):
        raise ValueError("lr must be positive and finite")
    if boundary_constraint not in ('edge', 'radius') or not np.isfinite(boundary_weight) or boundary_weight < 0:
        raise ValueError("use an edge or radius boundary constraint with a non-negative finite weight")
    if not mesh.n_faces or any(not row for row in mesh.adjacency):
        raise ValueError("Ricci flow requires triangles incident to every vertex")

    g = DiscreteMetric(mesh, mesh.edge_lengths)
    if not g.is_valid:
        raise ValueError("the input mesh contains a degenerate triangle")
    K, specified_verts, free_verts = _parse_target_curvature(
        mesh, target_curvature, fix_boundary, g.gaussian_curvature)

    if verbose > 0:
        chi = mesh.euler_characteristic
        print(f"V={mesh.n_vertices}, E={mesh.n_edges}, F={mesh.n_faces}, "
              f"boundary_V={len(mesh.boundary_vertices)}, chi={chi}")
        print(f"specified_verts={len(specified_verts)}, free_verts={len(free_verts)}")
        print(f"target total_K = {K.sum()/np.pi:.4f} pi")

    # Build circle packing metric
    cp = CirclePackingMetric.from_metric(g, scheme=scheme, alpha=alpha)
    if not cp.is_valid:
        raise ValueError("the selected circle packing produces an invalid initial metric")
    if verbose > 0:
        print(f"scale_factor = {cp.scale_factor:.6f}")

    # Rescale initial boundary lengths
    init_boundary_len = np.array([mesh.edge_lengths[e] for e in mesh.boundary_edges])
    init_boundary_len /= cp.scale_factor
    init_u = cp.u.copy()

    history = [float(np.abs((cp.gaussian_curvature - K)[specified_verts]).sum())]
    previous_u = init_u.copy()
    preview_iteration = -1
    previews = []
    skipped_previews = []
    options = dict(preview_options or {})
    options.setdefault("verbose", 0)
    if options.get("fixed_vertices") is not None and options.get("fixed_positions") is None:
        options["fixed_positions"] = mesh.vertices[np.asarray(options["fixed_vertices"], dtype=np.intp)]

    def reconstruct(u):
        nonlocal preview_iteration
        lengths = cp.scaled_edge_lengths(u)
        metric = DiscreteMetric(mesh, lengths)
        iteration = len(history) - 1
        if not metric.is_valid:
            # The extended-angle least-squares objective allows a trial to
            # cross a degenerate triangle. It has no spatial realization.
            skipped_previews.append(iteration)
            return
        shape = shape_from_metric(mesh, lengths, **options)
        options["initial_positions"] = shape.vertices.copy()
        preview_iteration = iteration
        snapshot = RicciFlowPreview(iteration, history[-1], lengths, shape, metric.gaussian_curvature)
        if on_preview is None:
            previews.append(snapshot)
        else:
            on_preview(snapshot)

    def record(u, error):
        nonlocal previous_u
        if np.array_equal(u, previous_u):
            return
        history.append(error)
        previous_u = u.copy()
        if preview_every and (len(history) - 1) % preview_every == 0:
            reconstruct(u)

    if method in ('sgd', 'newton'):
        cp.run_gradient_descent(
            K, free_verts, dt=lr, gtol=gtol, use_hessian=(method == 'newton'),
            verbose=verbose, max_iter=max_iter or 100000, on_accept=record)
    else:
        cp.run_least_squares(
            K, specified_verts, free_verts,
            target_u=init_u if boundary_constraint == 'radius' else None,
            target_lengths=init_boundary_len if boundary_constraint == 'edge' else None,
            boundary_weight=boundary_weight, opt_target='conformal_factor',
            method=method, gtol=gtol, verbose=verbose, on_accept=record, max_nfev=max_iter)

    if not cp.is_valid:
        raise ValueError("the final metric violates the strict triangle inequalities")
    final_error = float(np.abs((cp.gaussian_curvature - K)[specified_verts]).sum())
    record(cp.u, final_error)
    if preview_every and preview_iteration != len(history) - 1:
        reconstruct(cp.u)
        if preview_iteration != len(history) - 1:
            raise ValueError("the final edge lengths in input units violate the triangle inequalities")
    edge_lengths = cp.scaled_edge_lengths()
    curvature_mae = final_error / max(1, len(specified_verts))
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
        previews=previews,
        skipped_preview_iterations=skipped_previews,
    )
