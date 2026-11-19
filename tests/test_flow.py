import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ricci import CirclePackingMetric, DiscreteMetric, TriangleMesh, curvature_flow, ricci_flow


class FlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mesh = TriangleMesh.from_ply(Path(__file__).resolve().parents[1] / "demo/dome.ply")

    def test_inversive_jacobian_in_log_radii(self):
        mesh = self.mesh
        cp = CirclePackingMetric.from_metric(DiscreteMetric(mesh, mesh.edge_lengths))
        u = cp.u.copy()
        direction = np.linspace(-1, 1, mesh.n_vertices)
        analytic = cp.hessian() @ direction
        cp.u = u + 1e-6 * direction
        cp._update()
        plus = cp.gaussian_curvature.copy()
        cp.u = u - 1e-6 * direction
        cp._update()
        np.testing.assert_allclose(analytic, (plus - cp.gaussian_curvature) / 2e-6, atol=2e-8)

    def test_interval_and_final_observation_in_all_solvers(self):
        for method in ("trf", "lm", "sgd", "newton"):
            with self.subTest(method=method):
                frames = []
                run = ricci_flow(self.mesh, 0.1, method=method, verbose=0, lr=0.5,
                                 fix_boundary=True, max_iter=10, preview_every=3,
                                 preview_options={"max_iter": 2}, on_preview=frames.append)
                last = len(run.history) - 1
                expected = sorted(set([*range(3, last + 1, 3), last]))
                self.assertEqual([f.iteration for f in frames], expected)
                self.assertEqual(run.previews, [])
                self.assertGreater(run.history[0], run.history[-1])
                self.assertEqual(frames[-1].edge_lengths.keys(), run.edge_lengths.keys())
                np.testing.assert_allclose(list(frames[-1].edge_lengths.values()),
                                           list(run.edge_lengths.values()))
                for f in frames:
                    K = DiscreteMetric(self.mesh, f.edge_lengths).gaussian_curvature
                    err = abs(K - run.target_curvature)[self.mesh.interior_vertices].sum()
                    self.assertAlmostEqual(f.curvature_error, err, places=9)

    def test_opt_in_and_stationary_final_metric(self):
        target = np.full(self.mesh.n_vertices, 100)
        with patch("ricci.ricci.shape_from_metric", side_effect=AssertionError):
            result = ricci_flow(self.mesh, target, scheme="inversive", verbose=0)
            self.assertFalse(result.previews)
            self.assertLess(result.history[-1], 1e-10)
        result = ricci_flow(self.mesh, target, scheme="inversive", preview_every=8, verbose=0)
        self.assertEqual([f.iteration for f in result.previews], [0])
        np.testing.assert_allclose(result.previews[0].shape.vertices, self.mesh.vertices, atol=1e-8)

    def test_fixed_radii_and_per_component_csv_fill(self):
        result = ricci_flow(self.mesh, 0.1, fix_boundary=True, method="newton",
                            max_iter=6, lr=1, verbose=0)
        np.testing.assert_allclose(result.circle_packing.u[self.mesh.boundary_vertices], 0)
        mesh = TriangleMesh([[0,0,0], [2,0,0], [0,2,0], [5,0,0], [7,0,0], [5,2,0]],
                            [(0,1,2), (3,4,5)])
        with tempfile.TemporaryDirectory() as tmp:
            csv = Path(tmp) / "target.csv"
            csv.write_text("0,0.5\n", encoding="utf8")
            target = ricci_flow(mesh, csv, max_iter=1, verbose=0).target_curvature
        self.assertEqual(target[0], 0.5)
        self.assertAlmostEqual(sum(target[:3]), 2*np.pi)
        self.assertAlmostEqual(sum(target[3:]), 2*np.pi)

    def test_closed_components_keep_independent_scales(self):
        vertices = np.array([[0,0,0], [1,0,0], [0,2,0], [0.2,0.2,2.2]])
        faces = np.array([[0,1,2], [0,2,3], [0,3,1], [1,3,2]])
        mesh = TriangleMesh(np.vstack((vertices, 8*vertices+30)), np.vstack((faces, faces+4)))
        cp = CirclePackingMetric.from_metric(DiscreteMetric(mesh, mesh.edge_lengths))
        for method in ("trf", "lm", "newton"):
            run = ricci_flow(mesh, np.pi, scheme="inversive", method=method, lr=1, verbose=0)
            self.assertLess(run.history[-1], 1e-5)
            for group in mesh.connected_components:
                self.assertAlmostEqual(cp.u[group].mean(), run.circle_packing.u[group].mean(), places=6)

    def test_unrealizable_intermediate_can_be_skipped(self):
        def cross_degeneracy(cp, *args, on_accept, **kwargs):
            trial = cp.u.copy()
            trial[0] -= 5
            on_accept(trial, 1.0)
            on_accept(cp.u, 0.0)

        with patch.object(CirclePackingMetric, "run_least_squares", cross_degeneracy):
            run = ricci_flow(self.mesh, np.full(self.mesh.n_vertices, 100), scheme="inversive",
                             preview_every=1, verbose=0)
        self.assertEqual(run.skipped_preview_iterations, [1])
        self.assertEqual([f.iteration for f in run.previews], [2])
        np.testing.assert_allclose(run.previews[0].curvature, run.curvature)

    def test_direct_curvature_uses_boundary_angle_defects(self):
        target = DiscreteMetric(self.mesh, self.mesh.edge_lengths).gaussian_curvature
        result = curvature_flow(self.mesh, target, max_iter=1, lr=1e-7, verbose=0)
        self.assertLess(result.history[0], 1e-22)
        self.assertAlmostEqual(sum(result.curvature), self.mesh.euler_characteristic*2*np.pi)

    def test_obj_indices_and_face_winding_survive_ply_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            obj = Path(tmp) / "square.obj"
            obj.write_text("v 0 0 0\nv 0 1 0\nv 1 1 0\nv 1 0 0\nf -1 -2 -3 -4 # quad\n")
            mesh = TriangleMesh.from_obj(obj)
            ply = obj.with_suffix(".ply")
            mesh.with_vertices(mesh.vertices + 1).save_ply(ply)
            reread = TriangleMesh.from_ply(ply)
        np.testing.assert_array_equal(reread.triangles, [[3,2,1], [3,1,0]])
        np.testing.assert_allclose(reread.vertices, mesh.vertices + 1)


if __name__ == "__main__":
    unittest.main()
