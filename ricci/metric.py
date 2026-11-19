"""Discrete Riemannian metrics and circle packing metrics on triangle meshes.

Provides computation of angles, Gaussian curvature, area, and the
circle packing parameterisation used for Ricci flow.
"""

import os
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg
from scipy.optimize import least_squares

from .mesh import _partition_face


def _safe_arccos(x):
    """Numerically safe arccos for numpy arrays."""
    return np.arccos(np.clip(x, -1.0, 1.0))


def _triangle_angles(lengths):
    """Interior angles, with each column opposite the corresponding length."""
    a = np.roll(lengths, -1, axis=1)
    b = np.roll(lengths, 1, axis=1)
    return _safe_arccos((a*a + b*b - lengths*lengths) / (2*a*b))


def _angle_defects(mesh, angles):
    curvature = np.full(mesh.n_vertices, 2*np.pi)
    curvature[mesh.boundary_vertices] = np.pi
    return curvature - np.bincount(mesh.triangles.ravel(), angles.ravel(), minlength=mesh.n_vertices)


def _isfloat(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


class _ConstantMap:
    """Dict-like that returns a constant for any key."""
    def __init__(self, value=1):
        self._v = value
    def __getitem__(self, key):
        return self._v


class DiscreteMetric:
    """Discrete Riemannian metric on a triangle mesh.

    Given a mesh and edge lengths, computes angles via the law of cosines
    and Gaussian curvature via the angle defect.

    Attributes:
        mesh: the underlying TriangleMesh.
        gaussian_curvature: (N,) per-vertex Gaussian curvature.
        total_curvature: scalar sum of all curvatures.
        area: total surface area.
        is_valid: True if all triangle inequalities are satisfied.
    """

    def __init__(self, mesh, edge_lengths):
        """Create a metric from a mesh and edge length map.

        Args:
            mesh: TriangleMesh instance.
            edge_lengths: dict mapping frozenset({i,j}) -> length.
        """
        self.mesh = mesh
        self._n = mesh.n_vertices

        # Symmetrised directed edge lengths
        self._l = {}
        for e in mesh.edges:
            i, j = e
            self._l[i, j] = edge_lengths[e]
            self._l[j, i] = edge_lengths[e]

        # Angles and curvatures
        self._theta = {}
        self._K = None
        self._recompute()

    def _recompute(self):
        """Recompute angles and curvatures from current edge lengths."""
        self._face_lengths = np.asarray([
            [self._l[j, k], self._l[k, i], self._l[i, j]] for i, j, k in self.mesh.triangles
        ]).reshape(-1, 3)
        angles = _triangle_angles(self._face_lengths)
        for (i, j, k), (a, b, c) in zip(self.mesh.triangles, angles):
            self._theta[i, j, k] = self._theta[i, k, j] = a
            self._theta[j, k, i] = self._theta[j, i, k] = b
            self._theta[k, i, j] = self._theta[k, j, i] = c
        self._K = _angle_defects(self.mesh, angles)

    def _compute_angle(self, face, vert):
        """Compute interior angle at vert in face via law of cosines."""
        a, b, c = self._edge_lengths_at(face, vert)
        return _safe_arccos((a**2 + b**2 - c**2) / (2.0 * a * b))

    def _edge_lengths_at(self, face, vert):
        """Return (a, b, c) where a,b are edges incident to vert, c is opposite."""
        other = list(face - {vert})
        a = self._l[vert, other[0]]
        b = self._l[vert, other[1]]
        c = self._l[other[0], other[1]]
        return a, b, c

    def angle(self, face, vert):
        """Interior angle at vert in face."""
        j, k = face - {vert}
        return self._theta[(vert, j, k)]

    def angle_array(self, exclude=None):
        """All interior angles as a flat array, optionally excluding vertices."""
        exclude = set() if exclude is None else set(exclude)
        return np.array([
            self.angle(f, v) for f in self.mesh.faces for v in f
            if v not in exclude
        ])

    def _curvature_at(self, vert):
        """Gaussian curvature at a single vertex (angle defect)."""
        total_angle = sum(
            self.angle(f, vert) for f in self.mesh._adjacent_faces[vert])
        if vert in self.mesh.boundary_vertices:
            return np.pi - total_angle
        else:
            return 2 * np.pi - total_angle

    def _curvature_array(self):
        K = np.zeros(self._n)
        for v in range(self._n):
            K[v] = self._curvature_at(v)
        return K

    @property
    def gaussian_curvature(self):
        """Per-vertex Gaussian curvature array."""
        return self._K

    @property
    def total_curvature(self):
        return float(self._K.sum())

    @property
    def is_valid(self):
        """Check triangle inequality for all faces."""
        a, b, c = np.sort(self._face_lengths, axis=1).T
        return bool(np.all(np.isfinite(c) & (a > 0) & (a > c - b)))

    def length(self, edge):
        """Edge length for an edge (frozenset or tuple)."""
        i, j = edge
        return self._l[i, j]

    def face_area(self, face):
        i, j, k = face
        gamma = self._theta[(i, j, k)]
        a, b = self._l[i, j], self._l[i, k]
        return 0.5 * a * b * np.sin(gamma)

    @property
    def area(self):
        return sum(self.face_area(f) for f in self.mesh.faces)

    def enumerate_edges(self):
        """Return (edge_lengths_array, edge_map_dict) for optimisation."""
        edge_map = {}
        edgelen = []
        k = 0
        for (i, j), l in self._l.items():
            if j < i:
                edgelen.append(l)
                edge_map[i, j] = k
                edge_map[j, i] = k
                k += 1
        return edgelen, edge_map


# ---------------------------------------------------------------------------
# Circle packing Ricci energy and its gradient (module-level functions)
# ---------------------------------------------------------------------------

def _face_weights(eta, mesh):
    return np.array([[eta[j, k], eta[k, i], eta[i, j]] for i, j, k in mesh.triangles])


def _packing_lengths(radii, face_weights):
    rj = np.roll(radii, -1, axis=1)
    rk = np.roll(radii, 1, axis=1)
    return np.sqrt(rj*rj + rk*rk + 2*rj*rk*face_weights)


def _ricci_energy(r, face_weights, target_K, specified_verts, mesh):
    """Angle defect residual at vertices with prescribed curvature."""
    lengths = _packing_lengths(r[mesh.triangles], face_weights)
    K = _angle_defects(mesh, _triangle_angles(lengths))
    return K[specified_verts] - target_K[specified_verts]


def _grad_ricci_energy(x, face_weights, target_K, specified_verts, mesh, is_radius=True):
    """Assemble dK/du facewise by differentiating the cosine law."""
    faces = mesh.triangles
    radii = (x if is_radius else np.exp(x))[faces]
    lengths = _packing_lengths(radii, face_weights)
    angles = _triangle_angles(lengths)
    twice_area = lengths[:, 1] * lengths[:, 2] * np.sin(angles[:, 0])
    valid = np.all(angles > 0, axis=1) & (np.max(angles, axis=1) < np.pi)
    # Outside the triangle inequalities the extended angles are constant.
    inv_area = np.divide(1, twice_area, out=np.zeros_like(twice_area), where=valid)

    dangle_dl = np.empty((len(faces), 3, 3))
    dl_du = np.zeros_like(dangle_dl)
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        factor = lengths[:, i] * inv_area
        dangle_dl[:, i, i] = factor
        dangle_dl[:, i, j] = -factor * np.cos(angles[:, k])
        dangle_dl[:, i, k] = -factor * np.cos(angles[:, j])
        dl_du[:, i, j] = (radii[:, j] + face_weights[:, i] * radii[:, k]) / lengths[:, i]
        dl_du[:, i, k] = (radii[:, k] + face_weights[:, i] * radii[:, j]) / lengths[:, i]
    if not is_radius:
        dl_du *= radii[:, None, :]
    blocks = -(dangle_dl @ dl_du)
    rows = np.repeat(faces, 3, axis=1).ravel()
    cols = np.tile(faces, (1, 3)).ravel()
    matrix = sparse.coo_matrix((blocks.ravel(), (rows, cols)), shape=(mesh.n_vertices, mesh.n_vertices))
    return matrix.tocsr()[specified_verts, :]


def _fix_constraints(x, indices, value):
    return x[indices] - value

def _grad_fix_constraints(x, indices, value):
    Hm = sparse.dok_matrix((len(indices), len(x)))
    for i, j in enumerate(indices):
        Hm[i, j] = 1
    return Hm.tocsr()

def _edgelen_constraints(r, eta, fixed_edges, edgelen):
    L2 = np.array([2 * r[i] * r[j] * eta[i, j] + r[i]**2 + r[j]**2
                    for (i, j) in fixed_edges])
    return L2 - edgelen**2

def _grad_edgelen_constraints(x, eta, fixed_edges, edgelen, is_radius=True):
    r = x if is_radius else np.exp(x)
    Hm = sparse.dok_matrix((len(fixed_edges), len(r)))
    for k, (i, j) in enumerate(fixed_edges):
        if is_radius:
            Hm[k, i] = 2 * r[j] * eta[i, j] + 2 * r[i]
            Hm[k, j] = 2 * r[i] * eta[i, j] + 2 * r[j]
        else:
            Hm[k, i] = (2 * r[j] * eta[i, j] + 2 * r[i]) * r[i]
            Hm[k, j] = (2 * r[i] * eta[i, j] + 2 * r[j]) * r[j]
    return Hm.tocsr()

def _curvature_error(edgelen, edge_map, target_K, specified_verts, mesh):
    K = np.full(mesh.n_vertices, 2 * np.pi)
    K[mesh.boundary_vertices] = np.pi
    for i, j, k in mesh.faces:
        Li = edgelen[edge_map[j, k]]**2
        Lj = edgelen[edge_map[k, i]]**2
        Lk = edgelen[edge_map[i, j]]**2
        K[i] -= _safe_arccos((Lj + Lk - Li) / (2 * np.sqrt(Lj * Lk)))
        K[j] -= _safe_arccos((Lk + Li - Lj) / (2 * np.sqrt(Lk * Li)))
        K[k] -= _safe_arccos((Li + Lj - Lk) / (2 * np.sqrt(Li * Lj)))
    return K[specified_verts] - target_K[specified_verts]


class CirclePackingMetric(DiscreteMetric):
    """Circle packing metric for Ricci flow.

    Parameterises the metric by circle radii and structure coefficients (eta).
    The conformal factor u = log(radius) is the main optimisation variable.

    Attributes:
        conformal_factor: (N,) array u = log(radius).
        radius: (N,) circle packing radii.
        edge_weights: dict (i,j) -> eta_{ij} structure coefficient.
        scale_factor: global scaling factor.
    """

    def __init__(self, mesh):
        super().__init__(mesh, _ConstantMap())
        self.edge_weights = None
        self._eps = _ConstantMap()
        self.conformal_factor = None
        self.radius = None
        self.scale_factor = 1.0

    @classmethod
    def from_metric(cls, metric, scheme='inversive', alpha=-1):
        """Construct a circle packing from an existing discrete metric.

        Args:
            metric: DiscreteMetric instance.
            scheme: 'inversive', 'thurston', 'thurston2', 'combinatorial',
                    a float for constant eta, or a CSV file path.
            alpha: Thurston scaling factor (negative for auto).

        Returns:
            CirclePackingMetric instance.
        """
        cp = cls(metric.mesh)
        cp._init_from_metric(metric, scheme, alpha)
        return cp

    def _init_from_metric(self, g, scheme, alpha):
        """Compute radii and edge weights from an existing metric."""
        eta = {}
        mesh = g.mesh
        gamma = None

        if scheme == "inversive":
            pre_gamma = [[] for _ in range(g._n)]
            for face in mesh.faces:
                for i, (j, k) in _partition_face(face):
                    pre_gamma[i].append(
                        0.5 * (g.length((k, i)) + g.length((i, j)) - g.length((j, k))))
            gamma = np.array([min(pg) for pg in pre_gamma])

        elif scheme == "thurston":
            pre_gamma = [[] for _ in range(g._n)]
            for face in mesh.faces:
                for i, (j, k) in _partition_face(face):
                    pre_gamma[i].append(
                        0.5 * (g.length((k, i)) + g.length((i, j)) - g.length((j, k))))
            gamma = np.array([sum(pg) / len(pg) for pg in pre_gamma])

        elif scheme == "thurston2":
            gamma = np.array([
                (2.0 / 3.0) * min(g.length(e) for e in mesh.adjacent_edges(v))
                for v in range(mesh.n_vertices)
            ])

        if scheme in ("thurston", "thurston2"):
            if alpha < 0:
                alpha = 1.0
                for i, j in mesh.edges:
                    alpha = max(1.1 * g.length((i, j)) / (gamma[i] + gamma[j]), alpha)
            if alpha > 1.0:
                gamma *= alpha

        if scheme == "combinatorial" or _isfloat(scheme):
            eta_val = float(scheme) if _isfloat(scheme) else 1.0
            gamma = np.full(g._n, 1.0)
            for i, j in mesh.edges:
                eta[i, j] = eta_val
                eta[j, i] = eta_val
        elif isinstance(scheme, str) and os.path.isfile(scheme):
            etadata = np.loadtxt(scheme, delimiter=",", ndmin=2)
            gamma = np.full(g._n, 1.0)
            for i, j, l in etadata:
                eta[int(i), int(j)] = eta[int(j), int(i)] = l
            if any((i, j) not in eta for i, j in mesh.edges):
                raise ValueError("edge-weight CSV must specify every mesh edge")
        else:
            if gamma is None:
                raise ValueError(f"Unknown scheme: {scheme}")
            for edge in mesh.edges:
                i, j = edge
                sc = (g.length(edge)**2 - gamma[i]**2 - gamma[j]**2) / (2 * gamma[i] * gamma[j])
                if "thurston" in str(scheme):
                    sc = np.clip(sc, 0, 1)
                eta[i, j] = sc
                eta[j, i] = sc

        self.radius = gamma
        self.edge_weights = eta
        self.conformal_factor = np.log(gamma)
        self.scale_factor = np.exp(self.conformal_factor.mean())
        self.conformal_factor -= self.conformal_factor.mean()
        # Aliases for internal computation
        self._gamma = self.radius
        self._eta = self.edge_weights
        self.u = self.conformal_factor
        self._update()

    def _tau(self, i, j, k):
        return 0.5 * (self._l[j, k]**2 +
                      self._eps[j] * self._gamma[j]**2 -
                      self._eps[k] * self._gamma[k]**2)

    def compute_length(self, edge):
        i, j = list(edge)
        gi, gj = self._gamma[[i, j]]
        return np.sqrt(2 * gi * gj * self._eta[i, j] +
                       self._eps[i] * gi**2 + self._eps[j] * gj**2)

    def _packing_edge_lengths(self, u):
        i, j = self.mesh._edge_indices.T
        r = np.exp(u)
        eta = np.fromiter((self._eta[a, b] for a, b in self.mesh.edges), dtype=float)
        return np.sqrt(r[i]**2 + r[j]**2 + 2*r[i]*r[j]*eta)

    def scaled_edge_lengths(self, u=None):
        """Edge lengths in input units at an accepted log-radius vector."""
        if u is None:
            return {edge: self.scale_factor * self.length(edge) for edge in self.mesh.edges}
        return dict(zip(self.mesh.edges, self.scale_factor * self._packing_edge_lengths(u)))

    def _update(self):
        """Recompute lengths, angles, curvatures from current conformal factor."""
        self._gamma = np.exp(self.u)
        self.radius = self._gamma
        self.conformal_factor = self.u
        for (i, j), l in zip(self.mesh.edges, self._packing_edge_lengths(self.u)):
            self._l[i, j] = l
            self._l[j, i] = l
        self._recompute()

    def hessian(self):
        """Compute the Hessian dK/du as a sparse matrix."""
        return _grad_ricci_energy(
            self.u, _face_weights(self._eta, self.mesh), None, slice(None), self.mesh,
            is_radius=False)

    def run_gradient_descent(self, target_K, free_verts, dt=0.05, gtol=1e-4,
                             use_hessian=False, max_iter=100000, verbose=1, on_accept=None):
        """Gradient or Newton flow, backtracking to a valid improving metric."""
        free = np.asarray(free_verts, dtype=np.intp)
        fixed = set(range(self.mesh.n_vertices)).difference(free)
        gauges = [(group, self.u[group].mean()) for group in self.mesh.connected_components
                  if fixed.isdisjoint(group)]
        residual = (self._K - target_K)[free]
        history = [float(np.abs(residual).sum())]
        for iteration in range(1, max_iter + 1):
            if history[-1] <= gtol:
                break
            direction = residual
            if use_hessian:
                block = self.hessian()[free, :][:, free]
                direction = splinalg.lsqr(block, residual, atol=1e-10, btol=1e-10)[0]
            saved = self.u.copy()
            step = dt
            for _ in range(20):
                self.u = saved.copy()
                self.u[free] -= step * direction
                for group, initial_mean in gauges:
                    self.u[group] -= self.u[group].mean() - initial_mean
                self._update()
                residual = (self._K - target_K)[free]
                error = float(np.abs(residual).sum())
                if self.is_valid and error < history[-1]:
                    break
                step *= 0.5
            else:
                self.u = saved
                self._update()
                break
            history.append(error)
            if on_accept is not None:
                on_accept(self.u, error)
            if verbose >= 2 and iteration % 100 == 0:
                print(f"  iter {iteration}: |DeltaK|_1 = {error:.6e}")
        if verbose:
            print(f"  {len(history)-1} accepted steps, |DeltaK|_1 = {history[-1]:.6e}")
        return history

    def run_least_squares(self, target_K, specified_verts, free_verts,
                          target_u=None, target_lengths=None,
                          boundary_weight=1.0, opt_target='conformal_factor',
                          method='trf', gtol=1e-4, no_jacobian=False, verbose=1,
                          on_accept=None, max_nfev=None):
        """Scipy least_squares Ricci flow."""
        mesh = self.mesh
        weights = _face_weights(self._eta, mesh)
        fix_verts = list(set(range(mesh.n_vertices)) - set(free_verts))
        if target_lengths is None:
            w = np.sqrt(boundary_weight * len(specified_verts) / max(1, len(fix_verts)))
        else:
            w = np.sqrt(boundary_weight * len(specified_verts) / max(1, len(target_lengths)))

        if opt_target == 'conformal_factor':
            if target_lengths is None:
                if target_u is None:
                    target_u = self.u
                anchors = fix_verts
                radius_jac = _grad_fix_constraints(self.u, fix_verts, None)
                def boundary(x):
                    return x[fix_verts] - target_u[fix_verts]
                def boundary_jac(x):
                    return radius_jac
            else:
                anchors = mesh.boundary_vertices
                def boundary(x):
                    return _edgelen_constraints(np.exp(x), self._eta, mesh.boundary_edges, target_lengths)
                def boundary_jac(x):
                    return _grad_edgelen_constraints(
                        x, self._eta, mesh.boundary_edges, target_lengths, is_radius=False)

            # Curvature is invariant to a uniform change of log radius. Fix
            # the mean separately on components without a boundary scale.
            anchored = set(anchors) if w > 0 else set()
            groups = [v for v in mesh.connected_components if anchored.isdisjoint(v)]
            gauge = sparse.lil_matrix((len(groups), mesh.n_vertices))
            for row, vertices in enumerate(groups):
                gauge[row, vertices] = 1 / np.sqrt(len(vertices))
            gauge = gauge.tocsr()
            initial = self.u.copy()

            def curvature_residual(x):
                return _ricci_energy(np.exp(x), weights, target_K, specified_verts, mesh)

            def target(x):
                return np.concatenate((curvature_residual(x), w * boundary(x), gauge @ (x - initial)))

            def jacobian(x):
                matrix = sparse.vstack((
                    _grad_ricci_energy(x, weights, target_K, specified_verts, mesh, is_radius=False),
                    w * boundary_jac(x), gauge))
                # Both TRF and LM evaluate the Jacobian at accepted iterates.
                # Calls to the residual also include line-search trials.
                if on_accept is not None:
                    on_accept(x, float(np.abs(curvature_residual(x)).sum()))
                return matrix.toarray() if method == 'lm' else matrix

            self.u = least_squares(
                target, initial, jac='2-point' if no_jacobian else jacobian, verbose=verbose,
                method=method, xtol=gtol, gtol=gtol, max_nfev=max_nfev).x
            self._update()

        elif opt_target == 'radius':
            if target_lengths is None:
                boundary_r = np.exp(target_u)
                target = lambda x: np.concatenate([
                    _ricci_energy(x, weights, target_K, specified_verts, mesh),
                    w * _fix_constraints(x, fix_verts, boundary_r[fix_verts]),
                ])
                jac = lambda x: sparse.vstack([
                    _grad_ricci_energy(x, weights, target_K, specified_verts, mesh),
                    w * _grad_fix_constraints(x, fix_verts, target_u[fix_verts]),
                ])
            else:
                target = lambda x: np.concatenate([
                    _ricci_energy(x, weights, target_K, specified_verts, mesh),
                    w * _edgelen_constraints(x, self._eta, mesh.boundary_edges, target_lengths),
                ])
                jac = lambda x: sparse.vstack([
                    _grad_ricci_energy(x, weights, target_K, specified_verts, mesh),
                    w * _grad_edgelen_constraints(x, self._eta, mesh.boundary_edges, target_lengths),
                ])
            if no_jacobian:
                jac = '2-point'
            self.u = np.log(least_squares(
                target, self._gamma, jac=jac, verbose=verbose,
                method=method, xtol=gtol, gtol=gtol).x)
            self._update()

        elif opt_target == 'edge':
            edgelen, edge_map = self.enumerate_edges()
            fixed_e = [edge_map[i, j] for i, j in mesh.boundary_edges]
            boundary_e = [self._l[i, j] for i, j in mesh.boundary_edges]
            target = lambda x: np.concatenate([
                _curvature_error(x, edge_map, target_K, specified_verts, mesh),
                np.sqrt(boundary_weight) * _fix_constraints(x, fixed_e, boundary_e),
            ])
            res = least_squares(
                target, edgelen, bounds=(0, np.inf), verbose=verbose,
                method='trf', xtol=gtol, gtol=gtol).x
            for i, j in mesh.edges:
                self._l[i, j] = res[edge_map[i, j]]
                self._l[j, i] = res[edge_map[i, j]]
            self._recompute()
        else:
            raise ValueError(f"Unknown opt_target: {opt_target}")
