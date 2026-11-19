"""Extrinsic vertex-star parity from Chern et al., *Shape from Metric* (2018).

See Sections 2.1–2.2, https://doi.org/10.1145/3197517.3201276.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .mesh import TriangleMesh


@dataclass(frozen=True)
class ImmersionDiagnostics:
    """Counts and zero-based vertex IDs for the closed-star test.

    `classified_interior_vertices` counts all stars with a definite parity,
    including the figure-8 stars listed in `pinch_vertices`. A figure-0 star
    passes the parity test but can still overlap itself. The test is local and
    makes no claim about intersections between distant triangles.
    """

    classified_interior_vertices: int
    pinch_vertices: tuple[int, ...]
    indeterminate_vertices: tuple[int, ...]
    boundary_vertices: tuple[int, ...]


def _ordered_link(mesh, vertex):
    """Return (ordered neighbours, closed) for one cycle or path."""
    link = {}
    for face in mesh.adjacent_faces(vertex):
        i, j = face - {vertex}
        left = link.setdefault(i, [])
        right = link.setdefault(j, [])
        if j in left or len(left) >= 2 or len(right) >= 2:
            return None
        left.append(j)
        right.append(i)
    if not link:
        return None
    ends = [i for i, neighbours in link.items() if len(neighbours) == 1]
    if len(ends) not in (0, 2):
        return None
    start = ends[0] if ends else next(iter(link))
    ring = [start]
    previous, current = None, start
    while True:
        following = next((i for i in link[current] if i != previous), None)
        if following is None or following == start:
            break
        ring.append(following)
        previous, current = current, following
    if len(ring) != len(link):  # Disjoint fans meeting only at the center.
        return None
    return ring, not ends


def _pinch_parity(center, ring):
    # Work with unit directions so the ambiguity tolerance is dimensionless.
    tolerance = np.sqrt(np.finfo(float).eps)
    with np.errstate(over="ignore", invalid="ignore"):
        rays = ring - center
        tangents = np.roll(ring, -1, axis=0) - ring
        ray_lengths = np.hypot.reduce(rays, axis=1)
        tangent_lengths = np.hypot.reduce(tangents, axis=1)
    if (not np.all(np.isfinite(ray_lengths) & np.isfinite(tangent_lengths))
            or np.any(ray_lengths == 0) or np.any(tangent_lengths == 0)):
        return None
    rays /= ray_lengths[:, None]
    tangents /= tangent_lengths[:, None]
    normals = np.cross(rays, np.roll(rays, -1, axis=0))
    sines = np.linalg.norm(normals, axis=1)
    if np.any(sines <= tolerance):
        return None
    normals /= sines[:, None]
    # The segment between radial-edge midpoints is parallel to the opposite
    # edge. Consistent traversal supplies face orientations even if the input
    # file reverses some triangles.
    frames = np.stack((tangents, np.cross(normals, tangents), normals), axis=2)
    lifts = Rotation.from_matrix(frames).as_quat()
    # Re(conj(q_i) * q_j) = dot(q_i, q_j). Each negative dot changes sign
    # to select the paper's positive-real relative lift. Across a closed
    # loop all absolute lifts telescope, leaving just the product of signs.
    dots = np.einsum("ij,ij->i", lifts, np.roll(lifts, -1, axis=0))
    if np.any(np.abs(dots) <= tolerance):  # Ambiguous half-turn.
        return None
    return np.count_nonzero(dots < 0) % 2 == 0  # +1 = figure-8.


def immersion_diagnostics(mesh: TriangleMesh) -> ImmersionDiagnostics:
    """Test closed vertex stars for the pinch obstruction of Chern et al.

    Open manifold stars are listed as boundary vertices and excluded from
    the test. Malformed links, isolated vertices, and closed stars with
    degenerate triangles or ambiguous relative frame rotations are reported
    as indeterminate. Face winding in the input may be inconsistent.
    """
    classified = 0
    pinch, indeterminate, boundary = [], [], []
    for vertex in range(mesh.n_vertices):
        link = _ordered_link(mesh, vertex)
        if link is None:
            indeterminate.append(vertex)
            continue
        ring, closed = link
        if not closed:
            boundary.append(vertex)
            continue
        parity = _pinch_parity(mesh.vertices[vertex], mesh.vertices[ring])
        if parity is None:
            indeterminate.append(vertex)
        else:
            classified += 1
            if parity:
                pinch.append(vertex)
    return ImmersionDiagnostics(classified, tuple(pinch), tuple(indeterminate), tuple(boundary))
