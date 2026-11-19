import unittest

import numpy as np

from ricci import TriangleMesh, embed, shape_from_metric


class ShapeFromMetricTests(unittest.TestCase):
    def setUp(self):
        self.disk = TriangleMesh(
            [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, 0]],
            [(0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)],
        )

    def test_flat_disk_can_recover_height_in_any_plane_or_unit(self):
        rotation = np.array([[0.8, 0, 0.6], [0, 1, 0], [-0.6, 0, 0.8]])
        for scale in (1e-8, 1, 1e5):
            with self.subTest(scale=scale):
                mesh = self.disk.with_vertices(scale * (self.disk.vertices @ rotation + [1, 3, 2]))
                lifted = mesh.vertices.copy()
                lifted[-1] += 0.8 * scale * rotation[2]
                metric = mesh.with_vertices(lifted).edge_lengths
                result = embed(mesh, metric, method="shape_from_metric", fixed_vertices=[0, 1, 2, 3],
                               lambda_boundary=1, verbose=0)
                ratios = [result.mesh.edge_lengths[e] / target for e, target in metric.items()]
                np.testing.assert_allclose(ratios, 1, atol=1e-6)
                np.testing.assert_allclose(result.vertices[:4] / scale, mesh.vertices[:4] / scale, atol=1e-6)
                height = (result.vertices[-1] - mesh.vertices[-1]) @ rotation[2]
                self.assertAlmostEqual(abs(height) / scale, 0.8, places=5)
                self.assertEqual(result.scale_factor, 1)
                self.assertEqual(result.history[-1], result.cost)

    def test_disconnected_and_zero_weight_vertices_retain_pose(self):
        vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [9, 0, 0], [10, 0, 0], [9, 1, 0], [0, 0, 5]]
        mesh = TriangleMesh(vertices, [(0, 1, 2), (3, 4, 5)])
        lengths = np.array([mesh.edge_lengths[e] * 1.5 for e in mesh.edges])
        result = shape_from_metric(mesh, lengths, verbose=0)
        for ids in ([0, 1, 2], [3, 4, 5], [6]):
            np.testing.assert_allclose(result.vertices[ids].mean(axis=0),
                                       mesh.vertices[ids].mean(axis=0), atol=1e-12)
        np.testing.assert_allclose(list(result.mesh.edge_lengths.values()), lengths, rtol=1e-6)
        weights = [float(max(e) < 3) for e in mesh.edges]
        result = shape_from_metric(mesh, lengths, edge_weights=weights,
                                   fixed_vertices=[0], fixed_positions=[[1, 1, 0]], verbose=0)
        np.testing.assert_allclose(result.vertices[3:], mesh.vertices[3:], atol=1e-12)
        np.testing.assert_allclose(result.vertices[0], [1, 1, 0], atol=1e-6)

    def test_identity_metric_and_zero_iterations(self):
        same = shape_from_metric(self.disk, self.disk.edge_lengths, verbose=0)
        np.testing.assert_array_equal(same.vertices, self.disk.vertices)
        self.assertLess(same.cost, 1e-26)
        lengths = np.array(list(self.disk.edge_lengths.values()))
        weights = np.arange(1, len(lengths) + 1)
        result = shape_from_metric(self.disk, 2 * lengths, max_iter=0, edge_weights=weights, verbose=0)
        expected = np.sum((weights * (lengths**2 - (2 * lengths)**2))**2)
        self.assertAlmostEqual(result.cost / expected, 1)
        self.assertAlmostEqual(result.relative_edge_rms, 0.5)
        np.testing.assert_array_equal(result.vertices, self.disk.vertices)

    def test_relative_edge_rms_uses_realized_lengths_and_fitted_scale(self):
        lengths = np.array(list(self.disk.edge_lengths.values())) * 2
        result = embed(self.disk, lengths, method="optimization", fixed_vertices=[0, 1, 2, 3],
                       optimizer="Adam", lr=0.1, max_iter=1, fix_scale=False, verbose=0)
        self.assertGreater(abs(result.scale_factor - 1), 0.01)
        target = lengths * np.sqrt(result.scale_factor)
        actual = np.array([result.mesh.edge_lengths[e] for e in self.disk.edges])
        self.assertAlmostEqual(result.relative_edge_rms, np.sqrt(np.mean(((actual - target) / target)**2)))
        empty = TriangleMesh([], [])
        self.assertEqual(shape_from_metric(empty, {}, verbose=0).relative_edge_rms, 0.0)
        self.assertEqual(embed(empty, {}, method="optimization", verbose=0).relative_edge_rms, 0.0)

    def test_breaks_symmetry_of_a_flattened_closed_mesh(self):
        mesh = TriangleMesh(self.disk.vertices[:4], [(0, 1, 2), (0, 2, 3), (0, 3, 1), (1, 3, 2)])
        target = dict.fromkeys(mesh.edges, 1.)
        result = shape_from_metric(mesh, target, verbose=0)
        np.testing.assert_allclose(list(result.mesh.edge_lengths.values()), 1, rtol=1e-6)
        self.assertGreater(np.linalg.svd(result.vertices - result.vertices.mean(axis=0),
                                         compute_uv=False)[-1], 0.5)

    def test_degenerate_triangle_rejected_at_any_scale(self):
        mesh = TriangleMesh(self.disk.vertices[:3], [(0, 1, 2)])
        for scale in (1e-9, 1e9):
            with self.assertRaisesRegex(ValueError, "triangle inequality on face 0"):
                shape_from_metric(mesh, scale * np.array([1, 1, 2]), verbose=0)


if __name__ == "__main__":
    unittest.main()
