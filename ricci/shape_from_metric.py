"""Reconstruct vertex positions from a metric on a fixed triangle mesh.

The intrinsic-to-extrinsic problem follows Chern et al., Shape from Metric
(https://doi.org/10.1145/3197517.3201276). Here each face uses an independent
rigid frame and the initial vertex positions select a branch of the solution.
A local/global rigid fit provides a warm start for sparse least squares on the
squared edge lengths (Kaji and Zhang, https://arxiv.org/abs/2103.07584).
Planar stationary points are tested for lower-energy spatial realizations,
including a local trilateration step inspired by Kim and Santangelo
(https://arxiv.org/abs/2505.07696). The result depends on the initial shape.
"""

import numpy as np
from scipy import sparse
from scipy.optimize import least_squares
from scipy.sparse import csgraph, linalg as splinalg

from ._embed_common import (
    EmbedResult, edge_residual_cost, prepare_edge_data, prepare_edge_weights, relative_edge_rms,
)
from ._hinges import HingeBending
from .mesh import TriangleMesh


def _positions(name, values, count):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (count, 3) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a finite ({count}, 3) array")
    return values


def _constraints(reference, fixed_vertices, fixed_positions, vertex_weights, strength, n_edges):
    if fixed_vertices is None or strength == 0:
        return np.empty(0, dtype=int), np.empty((0, 3)), np.empty(0)
    ids = np.asarray(fixed_vertices)
    if ids.shape == (0,):
        return np.empty(0, dtype=int), np.empty((0, 3)), np.empty(0)
    if (ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer)
            or np.any(ids < 0) or np.any(ids >= len(reference))):
        raise ValueError("fixed_vertices must contain valid integer vertex indices")
    target = reference[ids] if fixed_positions is None else _positions(
        "fixed_positions", fixed_positions, len(ids))
    w = np.ones(len(ids)) if vertex_weights is None else np.asarray(vertex_weights, dtype=float)
    if w.shape != ids.shape or not np.isfinite(w).all() or np.any(w < 0) or not np.any(w > 0):
        raise ValueError("vertex_weights must match fixed_vertices, be finite, non-negative and not all zero")
    active = w > 0
    return ids[active], target[active], strength * max(n_edges, 1) / len(ids) * w[active]


def _face_charts(mesh, edges, lengths):
    """Intrinsic triangles and their three edge vectors, in canonical edge order."""
    faces = np.sort(mesh.triangles, axis=1)
    pairs = faces[:, [[0, 1], [1, 2], [0, 2]]]
    lookup = {tuple(edge): i for i, edge in enumerate(edges)}
    ids = np.asarray([lookup[tuple(pair)] for pair in pairs.reshape(-1, 2)], dtype=int).reshape(-1, 3)

    # Kahan's rearrangement of Heron's formula avoids cancellation on thin triangles.
    a, b, c = np.sort(lengths[ids], axis=1)[:, ::-1].T
    gap = c - (a - b)
    if np.any(gap <= 0):
        face = int(np.flatnonzero(gap <= 0)[0])
        raise ValueError(f"Target lengths violate the strict triangle inequality on face {face}")
    areas = 0.25 * np.sqrt((a + (b + c)) * gap * (c + (a - b)) * (a + (b - c)))
    base, opposite, diagonal = lengths[ids].T
    x = 0.5 * (base + (diagonal - opposite) * (diagonal + opposite) / base)
    y = 2 * areas / base
    chart = np.stack([np.column_stack((base, np.zeros_like(base))),
                      np.column_stack((x - base, y)), np.column_stack((x, y))], axis=1)
    return ids, chart, areas


def _components(incidence, edge_weights):
    graph = incidence[edge_weights > 0]
    count, labels = csgraph.connected_components((graph.T @ graph).tocsr(), directed=False)
    order = np.argsort(labels, kind="stable")
    components = np.split(order, np.cumsum(np.bincount(labels, minlength=count))[:-1])
    return labels, components


def _global_map(incidence, face_edges, chart, areas, weights, reference, labels,
                components, pins, pin_targets, pin_strength):
    """Factor the constant weighted graph Laplacian once for all XYZ right-hand sides."""
    strength = weights[face_edges] ** 2
    if len(areas):
        strength *= (areas / areas.mean())[:, None]
    edge_sum = np.bincount(face_edges.ravel(), strength.ravel(), minlength=incidence.shape[0])
    diagonal = np.bincount(pins, pin_strength, minlength=len(reference)).astype(float)
    anchor_rhs = np.zeros_like(reference)
    np.add.at(anchor_rhs, pins, pin_strength[:, None] * pin_targets)
    pinned = set(labels[pins])
    gauge_strength = max(float(edge_sum.max(initial=0)), 1.0)
    # Remove exactly one translation nullspace per unconstrained component.
    for component in components:
        first = component[0]
        if labels[first] not in pinned:
            diagonal[first] = gauge_strength
            anchor_rhs[first] = diagonal[first] * reference[first]
    matrix = incidence.T @ incidence.multiply(edge_sum[:, None]) + sparse.diags(diagonal)
    solve = splinalg.splu(matrix.tocsc()).solve
    weighted_chart = chart * strength[:, :, None]

    def update(vertices):
        world = (incidence @ vertices)[face_edges]
        covariance = world.transpose(0, 2, 1) @ weighted_chart
        u, _, vt = np.linalg.svd(covariance, full_matrices=False)
        rotated = weighted_chart @ (u @ vt).transpose(0, 2, 1)
        rhs_edges = np.zeros((len(edge_sum), 3))
        np.add.at(rhs_edges, face_edges.ravel(), rotated.reshape(-1, 3))
        return solve(incidence.T @ rhs_edges + anchor_rhs)

    return update


def _rigid_align(source, target):
    x, y = source - source.mean(axis=0), target - target.mean(axis=0)
    u, _, vt = np.linalg.svd(x.T @ y)
    u[:, -1] *= np.linalg.det(u @ vt)
    return x @ (u @ vt) + target.mean(axis=0)


def _plane(vertices):
    if len(vertices) < 4:
        return None
    _, s, vt = np.linalg.svd(vertices - vertices.mean(axis=0), full_matrices=False)
    if s[1] <= 1e-12 * max(s[0], 1.0) or s[2] > 1e-7 * s[0]:
        return None
    normal = vt[2].copy()
    normal *= np.sign(normal[np.argmax(np.abs(normal))])
    return vt[:2], normal


def _lift_flat(vertices, edges, lengths, weights, labels, components, pins,
               pin_targets, pin_strength, gtol):
    """Escape a compressed planar stationary point without random perturbations."""
    changed = False
    for component in components:
        plane = _plane(vertices[component])
        if plane is None:
            continue
        label = labels[component[0]]
        active = (labels[edges[:, 0]] == label) & (weights > 0)
        fixed = labels[pins] == label
        ij = np.searchsorted(component, edges[active])
        fixed_ids = np.searchsorted(component, pins[fixed])
        p = vertices[component]
        length = lengths[active]
        w2 = weights[active] ** 2
        delta = p[ij[:, 1]] - p[ij[:, 0]]
        strain = np.einsum("ij,ij->i", delta, delta) - length ** 2
        if not len(strain) or np.max(np.abs(strain) / length ** 2) <= gtol:
            continue
        basis, normal = plane
        xy = (p - p.mean(axis=0)) @ basis.T
        fixed_w = pin_strength[fixed]
        fixed_target = pin_targets[fixed]
        penalty = np.bincount(fixed_ids, fixed_w, minlength=len(p)).astype(float)
        best = p
        best_energy = float(w2 @ (strain ** 2) + fixed_w @ np.sum((p[fixed_ids] - fixed_target)**2, axis=1))

        def consider(candidate):
            nonlocal best, best_energy
            dif = candidate[ij[:, 1]] - candidate[ij[:, 0]]
            err = np.einsum("ij,ij->i", dif, dif) - length ** 2
            pin_err = candidate[fixed_ids] - fixed_target
            energy = w2 @ (err ** 2) + fixed_w @ np.sum(pin_err**2, axis=1)
            if energy < best_energy:
                best, best_energy = candidate, energy

        def try_profile(h):
            # On a plane the edge error along the normal is quadratic in height.
            # Its summed square has an exact minimizing amplitude along this profile.
            dh2 = (h[ij[:, 1]] - h[ij[:, 0]]) ** 2
            quadratic = 2 * (w2 * strain) @ dh2 + penalty @ (h * h)
            quartic = w2 @ (dh2 * dh2)
            if quadratic < 0 and quartic > 0:
                t = np.sqrt(-quadratic / (2 * quartic))
                consider(p + t * h[:, None] * normal)
                consider(p - t * h[:, None] * normal)

        # A smooth first choice for disk-like panels with a planar rim.
        profile = -np.sum(xy * xy, axis=1)
        profile -= (np.average(profile[fixed_ids], weights=fixed_w)
                    if len(fixed_ids) else profile.mean())
        try_profile(profile)

        def sum_at_vertices(values):
            return np.bincount(ij.ravel(), weights=np.repeat(values, 2), minlength=len(p))

        # The vertex with the greatest potential normal improvement is a local
        # trilateration candidate. Subtracting the neighboring sphere equations
        # gives a linear system for the in-plane foot. Both mirror heights are tried.
        a = sum_at_vertices(w2 * strain) + penalty / 2
        gain = np.minimum(a, 0)**2 / np.maximum(sum_at_vertices(w2), np.finfo(float).tiny)
        tip = np.argmax(gain)
        incident = (ij[:, 0] == tip) | (ij[:, 1] == tip)
        neighbours = ij[incident].sum(axis=1) - tip
        if len(neighbours) >= 3 and gain[tip] > 0:
            centers = xy[neighbours]
            w = w2[incident] / w2[incident].sum()
            rhs = np.sum(centers**2, axis=1) - length[incident]**2
            matrix = 2 * (centers - w @ centers)
            foot, _, rank, _ = np.linalg.lstsq(np.sqrt(w[:, None]) * matrix,
                                                np.sqrt(w) * (rhs - w @ rhs), rcond=None)
            h2 = w @ (length[incident]**2 - np.sum((centers - foot)**2, axis=1))
            if rank == 2 and h2 > 0:
                candidate = p.copy()
                for sign in (1, -1):
                    candidate[tip] = p.mean(axis=0) + foot @ basis + sign * np.sqrt(h2) * normal
                    consider(candidate.copy())

        if best is p:
            # At a planar optimum the normal Hessian is a signed stress
            # Laplacian. A negative eigenvalue supplies a descent direction
            # even when no *single* vertex can lower the energy.
            row = np.arange(len(ij)).repeat(2)
            incidence = sparse.csr_matrix((np.tile([-1., 1.], len(ij)),
                                           (row, ij.ravel())), shape=(len(ij), len(p)))
            stress = incidence.T @ incidence.multiply((2 * w2 * strain)[:, None]) + sparse.diags(penalty)
            if len(p) <= 32:
                mode = np.linalg.eigh(stress.toarray())[1][:, 0]
                try_profile(mode)
            else:
                try:
                    _, vecs = splinalg.eigsh(stress, k=1, which="SA", tol=1e-3,
                                              v0=np.random.default_rng(0).standard_normal(len(p)))
                    try_profile(vecs[:, 0])
                except splinalg.ArpackNoConvergence as err:
                    if err.eigenvectors.shape[1]:
                        try_profile(err.eigenvectors[:, 0])
        if best is not p:
            vertices[component] = best
            changed = True
    return changed


class _MetricResidual:
    """Metric, positional and optional dihedral residuals on fixed sparsity."""

    def __init__(self, edges, lengths, weights, pins, targets, pin_strength, n_vertices, bending):
        self.edges, self.lengths_sq, self.weights = edges, lengths**2, weights
        self.pins, self.targets, self.pin_scale = pins, targets, np.sqrt(pin_strength)
        self.bending = bending
        axes = np.arange(3)
        edge_cols = (3 * edges[:, :, None] + axes).ravel()
        pin_cols = (3 * pins[:, None] + axes).ravel()
        self.cols = np.concatenate((edge_cols, pin_cols))
        counts = np.repeat([6, 1], [len(edges), len(pin_cols)])
        if bending is not None:
            hinge_cols = (3 * bending.vertices[:, :, None] + axes).ravel()
            self.cols = np.r_[self.cols, hinge_cols]
            counts = np.r_[counts, np.full(len(bending.vertices), 12, dtype=int)]
        self.indptr = np.r_[0, counts.cumsum()]
        self.shape = (len(counts), 3 * n_vertices)

    def __call__(self, flat):
        vertices = flat.reshape(-1, 3)
        d = vertices[self.edges[:, 1]] - vertices[self.edges[:, 0]]
        edge = self.weights * (np.einsum("ij,ij->i", d, d) - self.lengths_sq)
        anchor = self.pin_scale[:, None] * (vertices[self.pins] - self.targets)
        blocks = [edge, anchor.ravel()]
        if self.bending is not None:
            blocks.append(self.bending.residual(vertices))
        return np.concatenate(blocks)

    def jac(self, flat):
        vertices = flat.reshape(-1, 3)
        d = vertices[self.edges[:, 1]] - vertices[self.edges[:, 0]]
        grad = (2 * self.weights)[:, None] * d
        entries = np.r_[np.column_stack((-grad, grad)).ravel(), np.repeat(self.pin_scale, 3)]
        if self.bending is not None:
            entries = np.r_[entries, self.bending.gradient(vertices).ravel()]
        return sparse.csr_matrix((entries, self.cols, self.indptr), shape=self.shape)


def shape_from_metric(
    mesh: TriangleMesh, target_edge_lengths, *, initial_positions=None, fixed_vertices=None,
    fixed_positions=None, vertex_weights=None, edge_weights=None, lambda_boundary: float = 0.01,
    bending_weight: float = 0.0, bending_target: str = "flat",
    fix_scale: bool = False, max_iter: int = 100, gtol: float = 1e-6, verbose: int = 1,
) -> EmbedResult:
    """Recover a spatial realization on the input connectivity.

    Alternating facewise rigid fits share a prefactored Poisson system. A
    sparse trust-region solve then refines the weighted squared-edge residual.
    Distances are normalized by the median target edge length internally.
    Soft positional constraints are in these same normalized units. Each
    unconstrained component is rigidly aligned to its starting pose.

    A positive `bending_weight` penalizes squared signed dihedral deviations,
    wrapped to [-pi, pi], relative to a flat hinge or the original mesh
    (`bending_target="reference"`). Each interior edge carries its squared
    target length divided by one third of the combined adjacent target areas,
    following the angle-difference variant of Sassen et al. (2024), §3.1.2
    (https://doi.org/10.1145/3658174). Higher weights trade stretch for bending.

    `max_iter` bounds outer updates and trust-region trial evaluations. The
    history and final cost report the edge-only residual in the input units.
    Target lengths fix the scale for either value of `fix_scale`. Fold choice
    can depend on initialization, and self-intersections are possible.
    """
    edges, lengths = prepare_edge_data(mesh, target_edge_lengths)
    weights = prepare_edge_weights(len(edges), edge_weights)
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 0:
        raise ValueError("max_iter must be a non-negative integer")
    if not np.isfinite(gtol) or gtol <= 0:
        raise ValueError("gtol must be positive and finite")
    if not np.isfinite(lambda_boundary) or lambda_boundary < 0:
        raise ValueError("lambda_boundary must be non-negative and finite")
    if not np.isfinite(bending_weight) or bending_weight < 0:
        raise ValueError("bending_weight must be non-negative and finite")
    if bending_target not in ("flat", "reference"):
        raise ValueError("bending_target must be 'flat' or 'reference'")
    seed = mesh.vertices if initial_positions is None else initial_positions
    reference = _positions("initial_positions", seed, mesh.n_vertices)
    pins, targets, penalties = _constraints(reference, fixed_vertices, fixed_positions,
                                            vertex_weights, lambda_boundary, len(edges))
    unit = float(np.median(lengths)) if len(edges) else 1.0
    scaled_lengths = lengths / unit
    face_edges, chart, areas = _face_charts(mesh, edges, scaled_lengths)
    origin = reference.mean(axis=0) if len(reference) else np.zeros(3)
    initial = (reference - origin) / unit
    current = initial.copy()
    targets = (targets - origin) / unit
    bending = None
    if bending_weight > 0:
        bending = HingeBending(mesh, edges, scaled_lengths, face_edges, areas, bending_weight, bending_target)
        if not np.isfinite(bending.angles(initial)).all():
            raise ValueError("initial positions must have nondegenerate triangles at each bending hinge")
    residual = _MetricResidual(edges, scaled_lengths, weights, pins, targets, penalties, len(current), bending)
    history = []

    def record(values):
        edge_loss = values[:len(edges)]
        history.append(float((edge_loss @ edge_loss) * unit ** 4))
        if verbose >= 2 and (len(history) == 1 or len(history) % 10 == 0):
            print(f"  update {len(history):5d}  edge cost={history[-1]:.8e}")

    value = residual(current)
    budget = max_iter
    if budget and np.max(np.abs(value), initial=0) > 32 * np.finfo(float).eps:
        row = np.arange(len(edges)).repeat(2)
        incidence = sparse.csr_matrix((np.tile([-1., 1.], len(edges)), (row, edges.ravel())),
                                      shape=(len(edges), len(current)))
        labels, components = _components(incidence, weights)
        update = _global_map(incidence, face_edges, chart, areas, weights, initial, labels,
                             components, pins, targets, penalties)

        for _ in range(min(4, budget)):
            if bending is not None and np.max(np.abs(value[:len(edges)]), initial=0) <= gtol:
                break  # An isometric fold can change only in the dihedral solve.
            candidate = update(current)
            new_value = residual(candidate)
            budget -= 1
            if not np.isfinite(new_value).all() or new_value @ new_value >= value @ value:
                break
            current, value = candidate, new_value
            record(value)
            if np.max(np.abs(value), initial=0) <= gtol:
                break

        def polish(vertices, allowance):
            # Recording on Jacobian evaluations includes accepted iterates only,
            # including on SciPy versions predating the callback API.
            first = True

            def jac(flat):
                nonlocal first
                if not first:
                    record(residual(flat))
                first = False
                return residual.jac(flat)

            tolerance = max(float(gtol), 10 * np.finfo(float).eps)
            result = least_squares(residual, vertices.ravel(), jac=jac, tr_solver="lsmr",
                                   max_nfev=allowance + 1, ftol=tolerance, xtol=tolerance,
                                   gtol=tolerance)
            return result.x.reshape(-1, 3), result.nfev - 1

        if budget and np.max(np.abs(value), initial=0) > gtol:
            planar = any(_plane(current[group]) is not None for group in components)
            allowance = max(1, budget // 3) if planar else budget
            current, used = polish(current, allowance)
            budget -= used
            if planar and budget:
                # A metric-only lift may increase the hinge energy. Accept
                # it only when the full objective improves.
                lifted = current.copy() if bending is not None else current
                changed = _lift_flat(lifted, edges, scaled_lengths, weights, labels,
                                     components, pins, targets, penalties, gtol)
                if changed and bending is not None:
                    before, after = residual(current), residual(lifted)
                    changed = after @ after < before @ before
                if changed:
                    current = lifted
                    budget -= 1
                    record(residual(current))
                if budget and (changed or used == allowance):
                    current, _ = polish(current, budget)

        if bending is not None:
            # A hinge can also couple vertices whose metric weights are zero.
            coupled = weights.copy()
            coupled[bending.face_edges] = 1
            labels, components = _components(incidence, coupled)
        anchored = set(labels[pins])
        for component in components:
            if labels[component[0]] not in anchored:
                current[component] = _rigid_align(current[component], initial[component])

    final = reference.copy() if np.array_equal(current, initial) else origin + unit * current
    cost = edge_residual_cost(final, edges, lengths, weights)
    if history:
        history[-1] = cost
    else:
        history.append(cost)
    if verbose:
        print(f"Shape from metric: {len(current)} vertices, {len(chart)} faces, edge cost={cost:.8e}")
    result_mesh = mesh.with_vertices(final)
    return EmbedResult(final, 1.0, cost, history, result_mesh,
                       relative_edge_rms=relative_edge_rms(result_mesh, lengths, scale_factor=1.0))
