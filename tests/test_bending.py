import unittest

import numpy as np

from ricci import TriangleMesh, embed, shape_from_metric
from ricci._embed_common import prepare_edge_data
from ricci._hinges import HingeBending
from ricci.shape_from_metric import _face_charts


def fold_angle(points):
    """Signed angle between normals for the fixture's common edge, in radians."""
    axis = points[1] - points[0]
    axis /= np.linalg.norm(axis)
    left = np.cross(axis, points[2] - points[0])
    right = np.cross(points[3] - points[0], axis)
    return np.arctan2(axis @ np.cross(left, right), left @ right)


class HingeBendingTests(unittest.TestCase):
    def setUp(self):
        self.mesh = TriangleMesh([[0, 0, 0], [1, 0, 0], [.4, .8, 0], [.65, -.6, .8]],
                                 [(0, 1, 2), (1, 0, 3)])

    def test_flat_target_unfolds_an_isometric_hinge_at_different_scales(self):
        for scale in [1e-9, 1., 1e7]:
            with self.subTest(scale=scale):
                mesh = self.mesh.with_vertices(scale * (self.mesh.vertices + [2, 1, -1]))
                same = shape_from_metric(mesh, mesh.edge_lengths, verbose=0)
                np.testing.assert_array_equal(same.vertices, mesh.vertices)
                fit = embed(mesh, mesh.edge_lengths, method="shape_from_metric",
                            bending_weight=.02, bending_target="flat", verbose=0)
                self.assertLess(abs(fold_angle(fit.vertices)), 1e-6)
                self.assertLess(fit.relative_edge_rms, 1e-6)
                self.assertEqual(fit.cost, fit.history[-1])

    def test_reference_uses_input_mesh_even_from_a_reflected_warm_start(self):
        mesh = self.mesh
        reflected = mesh.vertices * [1, 1, -1]
        fit = shape_from_metric(mesh, mesh.edge_lengths, initial_positions=reflected,
                                bending_weight=.02, bending_target="reference", verbose=0)
        self.assertAlmostEqual(fold_angle(fit.vertices), fold_angle(mesh.vertices), places=6)
        self.assertLess(fit.relative_edge_rms, 1e-6)

    def test_analytic_dihedral_derivative_and_intrinsic_weight(self):
        mesh = self.mesh
        edges, lengths = prepare_edge_data(mesh, mesh.edge_lengths)
        face_edges, _, areas = _face_charts(mesh, edges, lengths)
        weight = .2
        hinge = HingeBending(mesh, edges, lengths, face_edges, areas, weight, "flat")
        np.testing.assert_allclose(hinge.scale, np.sqrt(weight * 3 / areas.sum()))
        positions = mesh.vertices
        step = 1e-6
        direction = np.random.default_rng(5).normal(size=positions.shape)
        analytic = (hinge.gradient(positions) * direction[hinge.vertices]).sum(axis=(1, 2))
        numeric = (hinge.residual(positions + step * direction) -
                   hinge.residual(positions - step * direction)) / (2 * step)
        np.testing.assert_allclose(analytic, numeric, rtol=1e-7, atol=1e-9)

    def test_reference_angle_is_periodic_across_the_pi_seam(self):
        points = self.mesh.vertices.copy()
        points[3] = [.65, np.cos(.01), np.sin(.01)]
        mesh = self.mesh.with_vertices(points)
        edges, lengths = prepare_edge_data(mesh, mesh.edge_lengths)
        face_edges, _, areas = _face_charts(mesh, edges, lengths)
        hinge = HingeBending(mesh, edges, lengths, face_edges, areas, 1, "reference")
        reflected = points * [1, 1, -1]
        self.assertGreater(abs(fold_angle(reflected) - fold_angle(points)), 6)
        np.testing.assert_allclose(abs(hinge.residual(reflected)) / hinge.scale, .02)

    def test_active_hinges_validate_geometry_and_topology(self):
        mesh = self.mesh
        collapsed = mesh.vertices.copy()
        collapsed[3] = [.5, 0, 0]
        with self.assertRaisesRegex(ValueError, "nondegenerate"):
            shape_from_metric(mesh, mesh.edge_lengths, initial_positions=collapsed, bending_weight=1)
        with self.assertRaisesRegex(ValueError, "reference bending"):
            shape_from_metric(mesh.with_vertices(collapsed), mesh.edge_lengths,
                              initial_positions=mesh.vertices, bending_weight=1, bending_target="reference")
        three_faces = TriangleMesh(np.vstack((mesh.vertices, [0, 0, 2])),
                                   [(0, 1, 2), (1, 0, 3), (0, 1, 4)])
        with self.assertRaisesRegex(ValueError, "two faces"):
            shape_from_metric(three_faces, three_faces.edge_lengths, bending_weight=1)
        # Boundary-only triangles have no signed dihedral to validate.
        tri = TriangleMesh(collapsed[[0, 1, 3]], [(0, 1, 2)])
        fit = shape_from_metric(tri, [1, 1, 1], bending_weight=1,
                                bending_target="reference", max_iter=0, verbose=0)
        np.testing.assert_array_equal(fit.vertices, tri.vertices)

    def test_bending_options_are_validated(self):
        for kwargs in ({"bending_weight": -1}, {"bending_weight": np.nan},
                       {"bending_target": "unknown"}, {"method": "optimization", "bending_weight": 1}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "bending"):
                embed(self.mesh, self.mesh.edge_lengths, **kwargs)


if __name__ == "__main__":
    unittest.main()
