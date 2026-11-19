import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from ricci import TriangleMesh, immersion_diagnostics


def fan(turns, *, height=0.0):
    # Coprime number of steps and vertices avoids coincident rim vertices
    # when the link winds around the origin more than once.
    n = 11
    phi = np.arange(n) * (2 * np.pi / n)
    angle = turns * phi
    points = np.vstack(([0, 0, 0], np.column_stack(
        (np.cos(angle), np.sin(angle), height * np.sin(phi)))))
    faces = [(0, i + 1, (i + 1) % n + 1) for i in range(n)]
    return TriangleMesh(points, faces)


class ImmersionDiagnosticTests(unittest.TestCase):
    def test_planar_winding_gives_paper_parity(self):
        for winding, pinches in [(1, ()), (2, (0,)), (3, ())]:
            with self.subTest(winding=winding):
                mesh = fan(winding)
                report = immersion_diagnostics(mesh)
                self.assertEqual(report.classified_interior_vertices, 1)
                self.assertEqual(report.pinch_vertices, pinches)
                self.assertEqual(report.indeterminate_vertices, ())
                self.assertEqual(report.boundary_vertices, tuple(range(1, mesh.n_vertices)))

    def test_spatial_star_survives_rigid_motion_reordering_and_mixed_winding(self):
        rotation = Rotation.from_rotvec([0.33, 1.72, -0.8]).as_matrix()
        for winding, pinches in [(1, ()), (2, (0,))]:
            mesh = fan(winding, height=0.23)
            # Change the first incident face and its orientation.
            faces = np.roll(mesh.triangles.copy(), 5, axis=0)
            faces[::2] = faces[::2, ::-1]
            for scale in [1e-12, 1e12]:
                moved = TriangleMesh(scale * (mesh.vertices @ rotation + [5, 8, -3]), faces)
                report = immersion_diagnostics(moved)
                self.assertEqual(report.pinch_vertices, pinches)
                self.assertEqual(report.classified_interior_vertices, 1)
                self.assertEqual(report.indeterminate_vertices, ())

    def test_half_turn_and_collapsed_triangle_are_indeterminate(self):
        faces = [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 1)]
        for rim in [([1,0,0], [0,1,0], [2,0,0], [0,-1,0]),
                    ([1,0,0], [2,0,0], [0,1,0], [-1,0,0])]:
            report = immersion_diagnostics(TriangleMesh([[0,0,0], *rim], faces))
            self.assertEqual(report.indeterminate_vertices, (0,))
            self.assertEqual(report.classified_interior_vertices, 0)

    def test_disconnected_link_and_unused_vertex(self):
        mesh = TriangleMesh([[0,0,0], [1,0,0], [0,1,0], [-1,0,0], [0,-1,0], [0,0,1]],
                            [(0,1,2), (0,3,4)])
        report = immersion_diagnostics(mesh)
        self.assertEqual(report.indeterminate_vertices, (0, 5))
        self.assertEqual(report.boundary_vertices, (1, 2, 3, 4))

    def test_all_vertices_of_tetrahedron_pass(self):
        tetra = TriangleMesh([[0,0,0], [1,0,0], [0,1,0], [0,0,1]],
                             [(0,1,2), (1,2,3), (0,2,3), (0,1,3)])
        report = immersion_diagnostics(tetra)
        self.assertEqual(report.classified_interior_vertices, 4)
        self.assertEqual((report.pinch_vertices, report.boundary_vertices,
                          report.indeterminate_vertices), ((), (), ()))
