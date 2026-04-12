"""Discrete Riemannian metrics and circle packing metrics on triangle meshes.

Provides computation of angles, Gaussian curvature, area, and the
circle packing parameterisation used for Ricci flow.
"""

import os
import numpy as np
import functools
from scipy import sparse
from numpy.linalg import inv
from itertools import combinations
from scipy.sparse import linalg as splinalg
from scipy.optimize import least_squares

from .mesh import _partition_face


def _safe_arccos(x):
    """Numerically safe arccos for numpy arrays."""
    return np.arccos(x, where=(abs(x) < 1), out=np.full_like(x, np.pi))


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
        for face in self.mesh.faces:
            for i, (j, k) in _partition_face(face):
                theta = self._compute_angle(face, i)
                self._theta[(i, j, k)] = theta
                self._theta[(i, k, j)] = theta
        self._K = self._curvature_array()

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
        for face in self.mesh.faces:
            edges = [frozenset(e) for e in combinations(face, 2)]
            k = len(edges)
            for i in range(k):
                a, b = edges[(i + 1) % k], edges[(i + 2) % k]
                if self.length(a) + self.length(b) < self.length(edges[i]):
                    return False
        return True

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

def _ricci_energy(r, eta, target_K, specified_verts, mesh):
    """Residual of modified Ricci energy: K(r) - target_K at specified vertices."""
    K = np.full(len(r), 2 * np.pi)
    K[mesh.boundary_vertices] = np.pi
    for i, j, k in mesh.faces:
        Li = 2 * r[j] * r[k] * eta[j, k] + r[j]**2 + r[k]**2
        Lj = 2 * r[k] * r[i] * eta[k, i] + r[k]**2 + r[i]**2
        Lk = 2 * r[i] * r[j] * eta[i, j] + r[i]**2 + r[j]**2
        K[i] -= _safe_arccos((Lj + Lk - Li) / (2 * np.sqrt(Lj * Lk)))
        K[j] -= _safe_arccos((Lk + Li - Lj) / (2 * np.sqrt(Lk * Li)))
        K[k] -= _safe_arccos((Li + Lj - Lk) / (2 * np.sqrt(Li * Lj)))
    return K[specified_verts] - target_K[specified_verts]


def _grad_ricci_energy(x, eta, target_K, specified_verts, mesh, is_radius=True):
    """Jacobian of Ricci energy w.r.t. conformal factor or radius."""
    r = x if is_radius else np.exp(x)
    n = mesh.n_vertices

    l = {}
    for (i, j) in mesh.edges:
        l[i, j] = np.sqrt(2 * r[i] * r[j] * eta[i, j] + r[i]**2 + r[j]**2)
        l[j, i] = l[i, j]

    theta = {}
    for face in mesh.faces:
        for i, (j, k) in _partition_face(face):
            a, b, c = l[i, j], l[i, k], l[j, k]
            th = _safe_arccos((a**2 + b**2 - c**2) / (2.0 * a * b))
            theta[(i, j, k)] = th
            theta[(i, k, j)] = th

    H = {}
    for face in mesh.faces:
        i, j, k = face
        A = 0.5 * l[i, j] * l[i, k] * np.sin(theta[(i, j, k)])
        L = np.diag((l[j, k], l[i, k], l[i, j]))
        Linv = np.diag((1 / l[j, k], 1 / l[i, k], 1 / l[i, j]))
        tau = lambda ii, jj, kk: 0.5 * (l[jj, kk]**2 + r[jj]**2 - r[kk]**2)
        D = np.array([
            [0,           tau(i, j, k), tau(i, k, j)],
            [tau(j, i, k), 0,           tau(j, k, i)],
            [tau(k, i, j), tau(k, j, i), 0           ],
        ])
        def th(v):
            jj, kk = face - {v}
            return theta[(v, jj, kk)]
        Theta = np.cos(np.array([
            [np.pi, th(k), th(j)],
            [th(k), np.pi, th(i)],
            [th(j), th(i), np.pi],
        ]))
        Tijk = -0.5 / A * (L @ Theta @ Linv @ D)
        for a_v, row in zip((i, j, k), Tijk):
            for b_v, dtheta in zip((i, j, k), row):
                if (a_v, b_v) in H:
                    H[a_v, b_v] += dtheta
                else:
                    H[a_v, b_v] = dtheta

    Hm = sparse.dok_matrix((n, n))
    for (i, j), val in H.items():
        Hm[i, j] = -val / r[j] if is_radius else -val
    return Hm.tocsr()[specified_verts, :]


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
            etadata = np.loadtxt(scheme, delimiter=",")
            gamma = np.full(g._n, 1.0)
            for i, j, l in etadata:
                eta[int(i), int(j)] = l
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

    def _update(self):
        """Recompute lengths, angles, curvatures from current conformal factor."""
        self._gamma = np.exp(self.u)
        self.radius = self._gamma
        self.conformal_factor = self.u
        for edge in self.mesh.edges:
            i, j = edge
            l = self.compute_length(edge)
            self._l[i, j] = l
            self._l[j, i] = l
        self._recompute()

    def hessian(self):
        """Compute the Hessian dK/du as a sparse matrix."""
        n = self.mesh.n_vertices
        H = {}
        for face in self.mesh.faces:
            i, j, k = face
            A = self.face_area(face)
            L = np.diag((self._l[j, k], self._l[i, k], self._l[i, j]))
            D = np.array([
                [0,               self._tau(i, j, k), self._tau(i, k, j)],
                [self._tau(j, i, k), 0,               self._tau(j, k, i)],
                [self._tau(k, i, j), self._tau(k, j, i), 0              ],
            ])
            theta_fn = functools.partial(self.angle, face)
            Theta = np.cos(np.array([
                [np.pi,       theta_fn(k), theta_fn(j)],
                [theta_fn(k), np.pi,       theta_fn(i)],
                [theta_fn(j), theta_fn(i), np.pi      ],
            ]))
            Tijk = -0.5 / A * (L @ Theta @ inv(L) @ D)
            for a_v, row in zip((i, j, k), Tijk):
                for b_v, dtheta in zip((i, j, k), row):
                    if (a_v, b_v) in H:
                        H[a_v, b_v] += dtheta
                    else:
                        H[a_v, b_v] = dtheta
        Hm = sparse.dok_matrix((n, n))
        for (i, j), val in H.items():
            Hm[i, j] = val
        return Hm.tocsr()

    def run_gradient_descent(self, target_K, free_verts, dt=0.05, gtol=1e-4,
                             use_hessian=False, max_iter=100000, verbose=1):
        """Gradient descent (or Newton) Ricci flow."""
        DeltaK = self._K - target_K
        err = np.abs(DeltaK[free_verts]).sum()
        history = [err]
        fixed_verts = np.array(
            list(set(range(self.mesh.n_vertices)) - set(free_verts)))

        for niter in range(max_iter):
            if err <= gtol:
                break

            if use_hessian:
                if len(fixed_verts) > 0:
                    raise NotImplementedError(
                        "Hessian with fixed boundary is not implemented")
                H = self.hessian()
                deltau = splinalg.lsqr(H, DeltaK)[0]
                alpha = dt
                u_save = self.u.copy()
                for _ in range(10):
                    self.u = u_save + alpha * deltau
                    self.u -= self.u.mean()
                    self._update()
                    err_trial = np.abs((self._K - target_K)[free_verts]).sum()
                    if err_trial < err:
                        break
                    alpha *= 0.5
            else:
                alpha = dt
                u_save = self.u.copy()
                for _ in range(10):
                    self.u = u_save.copy()
                    self.u[free_verts] -= alpha * DeltaK[free_verts]
                    self.u -= self.u.mean()
                    self._update()
                    err_trial = np.abs((self._K - target_K)[free_verts]).sum()
                    if err_trial < err:
                        break
                    alpha *= 0.5

            DeltaK = self._K - target_K
            err = np.abs(DeltaK[free_verts]).sum()
            history.append(err)
            if niter % 100 == 0 and verbose > 0:
                print(f"  iter {niter}: |DeltaK|_1 = {err:.6e}")

        if verbose > 0:
            print(f"  Converged in {niter} iterations, |DeltaK|_1 = {err:.6e}")
        return history

    def run_least_squares(self, target_K, specified_verts, free_verts,
                          target_u=None, target_lengths=None,
                          boundary_weight=1.0, opt_target='conformal_factor',
                          method='trf', gtol=1e-4, no_jacobian=False, verbose=1):
        """Scipy least_squares Ricci flow."""
        mesh = self.mesh
        fix_verts = list(set(range(mesh.n_vertices)) - set(free_verts))
        if target_lengths is None:
            w = np.sqrt(boundary_weight * len(specified_verts) / max(1, len(fix_verts)))
        else:
            w = np.sqrt(boundary_weight * len(specified_verts) / max(1, len(target_lengths)))

        if opt_target == 'conformal_factor':
            if target_lengths is None:
                target = lambda x: np.concatenate([
                    _ricci_energy(np.exp(x), self._eta, target_K, specified_verts, mesh),
                    w * _fix_constraints(x, fix_verts, target_u[fix_verts]),
                ])
                jac = lambda x: sparse.vstack([
                    _grad_ricci_energy(x, self._eta, target_K, specified_verts, mesh, is_radius=False),
                    w * _grad_fix_constraints(x, fix_verts, target_u[fix_verts]),
                ])
            else:
                target = lambda x: np.concatenate([
                    _ricci_energy(np.exp(x), self._eta, target_K, specified_verts, mesh),
                    w * _edgelen_constraints(np.exp(x), self._eta, mesh.boundary_edges, target_lengths),
                ])
                jac = lambda x: sparse.vstack([
                    _grad_ricci_energy(x, self._eta, target_K, specified_verts, mesh, is_radius=False),
                    w * _grad_edgelen_constraints(x, self._eta, mesh.boundary_edges, target_lengths, is_radius=False),
                ])
            if no_jacobian:
                jac = '2-point'
            self.u = least_squares(
                target, self.u, jac=jac, verbose=verbose,
                method=method, xtol=gtol, gtol=gtol).x
            self._update()

        elif opt_target == 'radius':
            if target_lengths is None:
                boundary_r = np.exp(target_u)
                target = lambda x: np.concatenate([
                    _ricci_energy(x, self._eta, target_K, specified_verts, mesh),
                    w * _fix_constraints(x, fix_verts, boundary_r[fix_verts]),
                ])
                jac = lambda x: sparse.vstack([
                    _grad_ricci_energy(x, self._eta, target_K, specified_verts, mesh),
                    w * _grad_fix_constraints(x, fix_verts, target_u[fix_verts]),
                ])
            else:
                target = lambda x: np.concatenate([
                    _ricci_energy(x, self._eta, target_K, specified_verts, mesh),
                    w * _edgelen_constraints(x, self._eta, mesh.boundary_edges, target_lengths),
                ])
                jac = lambda x: sparse.vstack([
                    _grad_ricci_energy(x, self._eta, target_K, specified_verts, mesh),
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
