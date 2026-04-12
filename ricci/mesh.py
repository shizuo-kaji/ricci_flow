"""Triangle mesh representation and I/O.

Provides TriangleMesh with PLY/OBJ loading, boundary detection,
adjacency computation, and edge length extraction.
"""

import numpy as np
from numpy.linalg import norm
from itertools import combinations
from functools import reduce
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
            tok = line.split()
            if not tok or tok[0] == "#":
                continue
            elif tok[0] == "v":
                verts.append([float(x) for x in tok[1:]])
            elif tok[0] == "f":
                poly = [int(s.split("/")[0]) - 1 for s in tok[1:]]
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

        # Triangulate and store as frozensets
        self.faces = []
        for poly in faces:
            self.faces.extend([frozenset(f) for f in _triangulate(poly)])

        # Build edges, lengths, and detect boundary
        edge_list = []
        lengths = {}
        boundary_edges = []
        for face in self.faces:
            for edge in combinations(face, 2):
                edge = frozenset(edge)
                if edge not in edge_list:
                    i, j = edge
                    edge_list.append(edge)
                    lengths[edge] = norm(self.vertices[i] - self.vertices[j])
                    boundary_edges.append(edge)
                else:
                    if edge in boundary_edges:
                        boundary_edges.remove(edge)

        self.edges = edge_list
        self.edge_lengths = lengths
        self.boundary_edges = boundary_edges

        if boundary_edges:
            self.boundary_vertices = sorted(
                list(reduce(lambda a, b: a.union(b), boundary_edges)))
        else:
            self.boundary_vertices = []

        all_verts = list(range(len(self.vertices)))
        self.interior_vertices = [v for v in all_verts if v not in set(self.boundary_vertices)]

        # Adjacency: per-vertex list of adjacent faces and neighbours
        self._adjacent_faces = [
            [f for f in self.faces if v in f] for v in all_verts]
        self.adjacency = [
            sorted(set(
                np.array([list(f) for f in self._adjacent_faces[v]]).flatten().astype(int)
            ) - {v})
            for v in all_verts
        ]

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
        nf = len(list(self.faces[0]))
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
            [(list(f), 0) for f in self.faces],
            dtype=[('vertex_indices', 'i4', (nf,)), ('red', 'u1')])
        el2 = PlyElement.describe(face_arr, 'face')
        PlyData([el1, el2], text=True).write(path)

    def with_vertices(self, new_vertices):
        """Return a new mesh with the same connectivity but different vertex positions."""
        face_tuples = [tuple(sorted(f)) for f in self.faces]
        m = TriangleMesh.__new__(TriangleMesh)
        m.__dict__.update(self.__dict__)
        m.vertices = np.asarray(new_vertices, dtype=np.float64)
        # Recompute edge lengths
        m.edge_lengths = {}
        for edge in self.edges:
            i, j = edge
            m.edge_lengths[edge] = norm(m.vertices[i] - m.vertices[j])
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
