"""Triangle mesh representation and I/O.

Provides TriangleMesh with PLY/OBJ loading, boundary detection,
adjacency computation, and edge length extraction.
"""

import numpy as np
from numpy.linalg import norm
from itertools import combinations
from functools import cached_property
from plyfile import PlyData, PlyElement


def _triangulate(poly):
    """Fan-triangulate a polygon."""
    if len(poly) == 3:
        yield poly
    else:
        e0 = poly[0]
        for e1, e2 in zip(poly[1:], poly[2:]):
            yield (e0, e1, e2)


def _partition_face(face):
    """Yield (vertex, (opposite_pair)) for each vertex in a triangle."""
    i, j, k = face
    yield i, (j, k)
    yield j, (k, i)
    yield k, (i, j)


def _read_obj(path):
    """Parse a Wavefront OBJ file, returning (vertices, faces)."""
    verts = []
    faces = []
    with open(path) as f:
        for line in f:
            tok = line.partition("#")[0].split()
            if not tok or tok[0] == "#":
                continue
            elif tok[0] == "v":
                verts.append([float(x) for x in tok[1:4]])
            elif tok[0] == "f":
                indices = [int(s.split("/")[0]) for s in tok[1:]]
                if 0 in indices:
                    raise ValueError("OBJ vertex indices start at 1")
                poly = [i - 1 if i > 0 else len(verts) + i for i in indices]
                faces.append(poly)
    return np.array(verts, dtype=np.float64), faces


class TriangleMesh:
    """Triangulated surface mesh.

    Attributes:
        vertices: (N, 3) vertex coordinates.
        faces: list of frozensets, each a triangle {i, j, k}.
        edges: list of frozensets, each an edge {i, j}.
        boundary_edges: edges on the mesh boundary.
        boundary_vertices: sorted list of boundary vertex indices.
        interior_vertices: non-boundary vertex indices.
        edge_lengths: dict mapping frozenset edge -> length.
        adjacency: per-vertex list of adjacent vertex indices.
        n_vertices, n_edges, n_faces: counts.
        euler_characteristic: V - E + F.
    """

    def __init__(self, vertices, faces):
        """Build mesh from vertices and faces.

        Args:
            vertices: (N, 3) array of vertex positions.
            faces: list of index tuples/lists (triangles or polygons).
        """
        self.vertices = np.asarray(vertices, dtype=np.float64)

        if self.vertices.size == 0:
            self.vertices = self.vertices.reshape(0, 3)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3 or not np.isfinite(self.vertices).all():
            raise ValueError("vertices must be a finite (N, 3) array")

        # Keep the input winding for export, and the legacy unordered faces
        # and edge enumeration for metric arrays supplied by existing callers.
        self.faces = []
        triangles = []
        edge_counts = {}
        self._adjacent_faces = [[] for _ in self.vertices]
        neighbours = [set() for _ in self.vertices]
        for poly in faces:
            poly = tuple(poly)
            if len(poly) < 3:
                raise ValueError("faces must have at least three vertices")
            for tri in _triangulate(poly):
                face = frozenset(tri)
                if (len(face) != 3 or any(not isinstance(v, (int, np.integer)) for v in tri)
                        or min(tri) < 0 or max(tri) >= len(self.vertices)):
                    raise ValueError("triangles must reference three distinct valid vertex indices")
                self.faces.append(face)
                triangles.append(tri)
                for v in tri:
                    self._adjacent_faces[v].append(face)
                for i, j in combinations(face, 2):
                    edge = frozenset((i, j))
                    edge_counts[edge] = edge_counts.get(edge, 0) + 1
                    neighbours[i].add(j)
                    neighbours[j].add(i)
        self.triangles = np.asarray(triangles, dtype=np.intp).reshape(-1, 3)
        self.edges = list(edge_counts)
        self._edge_indices = np.asarray([tuple(e) for e in self.edges], dtype=np.intp).reshape(-1, 2)
        self.edge_lengths = self._measure_edges(self.vertices)
        self.boundary_edges = [e for e, count in edge_counts.items() if count == 1]
        boundary = {v for e in self.boundary_edges for v in e}
        self.boundary_vertices = sorted(boundary)
        self.interior_vertices = [v for v in range(self.n_vertices) if v not in boundary]
        self.adjacency = [sorted(row) for row in neighbours]

    def _measure_edges(self, vertices):
        i, j = self._edge_indices.T
        return dict(zip(self.edges, norm(vertices[i] - vertices[j], axis=1)))

    @classmethod
    def from_ply(cls, path):
        """Load mesh from a PLY file."""
        plydata = PlyData.read(path)
        vertices = np.vstack([
            plydata['vertex']['x'],
            plydata['vertex']['y'],
            plydata['vertex']['z'],
        ]).astype(np.float64).T
        faces = plydata['face']['vertex_indices']
        return cls(vertices, faces)

    @classmethod
    def from_obj(cls, path):
        """Load mesh from a Wavefront OBJ file."""
        vertices, faces = _read_obj(path)
        return cls(vertices, faces)

    def save_ply(self, path, vertex_colours=None):
        """Write mesh to a PLY file.

        Args:
            path: output file path.
            vertex_colours: optional (N, 3) or (N, 4) uint8 array of RGB(A) colours.
        """
        nf = 3
        if vertex_colours is None:
            el1 = PlyElement.describe(
                np.array([(v[0], v[1], v[2]) for v in self.vertices],
                         dtype=[('x', 'f8'), ('y', 'f8'), ('z', 'f8')]),
                'vertex')
        else:
            el1 = PlyElement.describe(
                np.array([(v[0], v[1], v[2], c[0], c[1], c[2])
                          for v, c in zip(self.vertices, vertex_colours)],
                         dtype=[('x', 'f8'), ('y', 'f8'), ('z', 'f8'),
                                ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]),
                'vertex')
        face_arr = np.array(
            [(f, 0) for f in self.triangles],
            dtype=[('vertex_indices', 'i4', (nf,)), ('red', 'u1')])
        el2 = PlyElement.describe(face_arr, 'face')
        PlyData([el1, el2], text=True).write(path)

    def with_vertices(self, new_vertices):
        """Return a new mesh with the same connectivity but different vertex positions."""
        m = TriangleMesh.__new__(TriangleMesh)
        m.__dict__.update(self.__dict__)
        m.vertices = np.asarray(new_vertices, dtype=np.float64)
        if m.vertices.shape != self.vertices.shape or not np.isfinite(m.vertices).all():
            raise ValueError("new_vertices must be finite and match the original shape")
        m.edge_lengths = self._measure_edges(m.vertices)
        return m

    @property
    def n_vertices(self):
        return len(self.vertices)

    @property
    def n_edges(self):
        return len(self.edges)

    @property
    def n_faces(self):
        return len(self.faces)

    @property
    def euler_characteristic(self):
        return self.n_vertices - self.n_edges + self.n_faces

    @cached_property
    def connected_components(self):
        """Vertex indices for each connected component, including isolated vertices."""
        seen = np.zeros(self.n_vertices, dtype=bool)
        groups = []
        for start in range(self.n_vertices):
            if seen[start]:
                continue
            group = [start]
            seen[start] = True
            for vertex in group:
                for neighbour in self.adjacency[vertex]:
                    if not seen[neighbour]:
                        seen[neighbour] = True
                        group.append(neighbour)
            groups.append(np.array(group, dtype=np.intp))
        return groups

    def adjacent_faces(self, vertex):
        """Return faces adjacent to a vertex."""
        return self._adjacent_faces[vertex]

    def adjacent_edges(self, vertex):
        """Return edges adjacent to a vertex."""
        return [e for e in self.edges if vertex in e]

    def min_valence(self):
        """Return the minimum vertex valence."""
        return min(len(self._adjacent_faces[v]) for v in range(self.n_vertices))

    def __repr__(self):
        return (f"TriangleMesh(V={self.n_vertices}, E={self.n_edges}, "
                f"F={self.n_faces}, boundary_V={len(self.boundary_vertices)}, "
                f"chi={self.euler_characteristic})")
