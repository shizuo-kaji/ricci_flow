"""Discrete shell bending on pairs of triangles sharing an edge."""

import numpy as np


class HingeBending:
    """Signed dihedral residuals and their four-vertex derivatives."""

    def __init__(self, mesh, edges, lengths, face_edges, areas, weight, target):
        # The intrinsic charts list the edges ab, bc, ac of sorted triangle abc.
        opposite = np.sort(mesh.triangles, axis=1)[:, [2, 0, 1]].ravel()
        flat_edges = face_edges.ravel()
        counts = np.bincount(flat_edges, minlength=len(edges))
        if np.any(counts > 2):
            raise ValueError("hinge bending requires at most two faces incident to an edge")
        interior = np.flatnonzero(counts == 2)
        first = (counts.cumsum() - counts)[interior]
        order = np.argsort(flat_edges, kind="stable")
        left, right = order[first], order[first + 1]
        if np.any(opposite[left] == opposite[right]):
            raise ValueError("hinge bending requires distinct triangles on both sides of an edge")
        self.vertices = np.column_stack((edges[interior], opposite[left], opposite[right]))
        adjacent = np.column_stack((left // 3, right // 3))
        self.face_edges = face_edges[adjacent].ravel()
        dual_area = areas[adjacent].sum(axis=1) / 3
        self.scale = np.sqrt(weight) * lengths[interior] / np.sqrt(dual_area)
        self.rest = np.zeros(len(interior))
        if target == "reference":
            self.rest = self.angles(mesh.vertices)
            if not np.isfinite(self.rest).all():
                raise ValueError("reference bending requires nondegenerate triangles at each hinge")

    def angles(self, points, *, derivative=False):
        """Return the signed hinge angles or their derivatives with respect to XYZ."""
        a, b, k, l = points[self.vertices].transpose(1, 0, 2)
        edge, arm_k, arm_l = b - a, k - a, l - a
        # Angles are homogeneous of degree zero. Form cross products in
        # local units, including when reading original, unnormalized points.
        unit = np.maximum.reduce([np.abs(v).max(axis=1) for v in (edge, arm_k, arm_l)])
        unit = np.where(unit > 0, unit, 1)
        edge, arm_k, arm_l = (v / unit[:, None] for v in (edge, arm_k, arm_l))
        length = np.linalg.norm(edge, axis=1)
        left, right = np.cross(edge, arm_k), np.cross(arm_l, edge)
        area_l, area_r = np.linalg.norm(left, axis=1), np.linalg.norm(right, axis=1)
        # Relative to the hinge arm lengths so a change of units does not
        # turn a well-shaped triangle into a degenerate one.
        eps = 32 * np.finfo(float).eps
        valid = ((area_l > eps * length * np.linalg.norm(arm_k, axis=1)) &
                 (area_r > eps * length * np.linalg.norm(arm_l, axis=1)))
        with np.errstate(divide="ignore", invalid="ignore"):
            nl, nr = left / area_l[:, None], right / area_r[:, None]
            axis = edge / length[:, None]
        if not derivative:
            # A collapsed trial has no angle. SciPy contracts the trust region
            # when its residual is nonfinite and retries from the last iterate.
            cosine = np.einsum("ij,ij->i", nl, nr)
            sine = np.einsum("ij,ij->i", axis, np.cross(nl, nr))
            return np.where(valid, np.arctan2(sine, cosine), np.nan)
        if not np.all(valid):
            raise ValueError("a collapsed hinge has no dihedral derivative")
        # The opposite vertex rotates its face about the edge. Its angle
        # derivative is the negative oriented normal divided by its altitude.
        gk = -(length / area_l / unit)[:, None] * nl
        gl = -(length / area_r / unit)[:, None] * nr
        tk = np.einsum("ij,ij->i", arm_k, edge) / length**2
        tl = np.einsum("ij,ij->i", arm_l, edge) / length**2
        gb = -tk[:, None] * gk - tl[:, None] * gl
        ga = -(gb + gk + gl)
        return np.stack((ga, gb, gk, gl), axis=1)

    def residual(self, points):
        delta = self.angles(points) - self.rest
        return self.scale * np.arctan2(np.sin(delta), np.cos(delta))

    def gradient(self, points):
        return self.scale[:, None, None] * self.angles(points, derivative=True)
