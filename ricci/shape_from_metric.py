"""Reference-inspired shape-from-metric embedding.

This module implements a minimal Python version of the core reconstruction
loop used by the Shape from Metric workflow: recover per-face rigid motions
from an intrinsic metric, then reconstruct vertex positions through a global
Poisson-style least-squares solve. It intentionally omits the full spinor /
spin-connection topology machinery of the original Houdini implementation.
"""

from collections import deque
from typing import Optional

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg

from ._embed_common import (
    EmbedResult,
    edge_residual_cost,
    prepare_edge_data,
    prepare_edge_weights,
)
from .mesh import TriangleMesh


def _validate_vertex_array(name: str, values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _validate_vertex_indices(name: str, values: np.ndarray, n_vertices: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.int64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    if np.any((arr < 0) | (arr >= n_vertices)):
        raise ValueError(f"{name} contains out-of-range vertex indices")
    return arr


def _validate_vertex_weights(weights: np.ndarray, n_verts: int) -> np.ndarray:
    arr = np.asarray(weights, dtype=np.float64)
    if arr.shape != (n_verts,):
        raise ValueError("vertex_weights must match fixed_vertices")
    if not np.all(np.isfinite(arr)):
        raise ValueError("vertex_weights must be finite")
    if np.any(arr < 0) or not np.any(arr > 0):
        raise ValueError("vertex_weights must be non-negative and not all zero")
    return arr


def _weighted_similarity_parameters(
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    *,
    allow_scaling: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    if len(source) == 0:
        return np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64), 1.0

    w = weights / weights.sum()
    source_mean = np.sum(w[:, None] * source, axis=0)
    target_mean = np.sum(w[:, None] * target, axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean

    covariance = source_centered.T @ (w[:, None] * target_centered)
    U, _, Vt = np.linalg.svd(covariance, full_matrices=False)

    correction = np.eye(3, dtype=np.float64)
    if np.linalg.det(U @ Vt) < 0:
        correction[-1, -1] = -1.0
    rotation = U @ correction @ Vt

    if allow_scaling:
        source_var = float(np.sum(w * np.sum(source_centered**2, axis=1)))
        numerator = float(np.trace(rotation.T @ covariance))
        scale = numerator / source_var if source_var > 1e-15 else 1.0
    else:
        scale = 1.0

    translation = target_mean - scale * source_mean @ rotation
    return rotation, translation, scale


def _apply_similarity_alignment(
    vertices: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    *,
    allow_scaling: bool,
) -> tuple[np.ndarray, float]:
    rotation, translation, scale = _weighted_similarity_parameters(
        source,
        target,
        weights,
        allow_scaling=allow_scaling,
    )
    transformed = scale * vertices @ rotation + translation
    return transformed, scale**2


def _has_directed_edge(face: tuple[int, int, int], edge: tuple[int, int]) -> bool:
    a, b, c = face
    u, v = edge
    return (a == u and b == v) or (b == u and c == v) or (c == u and a == v)


def _build_oriented_faces(mesh: TriangleMesh) -> np.ndarray:
    face_sets = [set(face) for face in mesh.faces]
    provisional = [tuple(sorted(face)) for face in mesh.faces]
    edge_to_faces: dict[frozenset, list[int]] = {}

    for face_index, face in enumerate(face_sets):
        a, b, c = tuple(face)
        for edge in (frozenset({a, b}), frozenset({b, c}), frozenset({c, a})):
            edge_to_faces.setdefault(edge, []).append(face_index)

    oriented: list[Optional[tuple[int, int, int]]] = [None] * len(mesh.faces)

    for start in range(len(mesh.faces)):
        if oriented[start] is not None:
            continue

        oriented[start] = provisional[start]
        queue = deque([start])

        while queue:
            face_index = queue.popleft()
            face = oriented[face_index]
            assert face is not None

            for edge in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
                neighbours = edge_to_faces[frozenset(edge)]
                for neighbour_index in neighbours:
                    if neighbour_index == face_index:
                        continue

                    third_vertex = next(iter(face_sets[neighbour_index] - set(edge)))
                    candidate = (edge[1], edge[0], third_vertex)
                    assigned = oriented[neighbour_index]

                    if assigned is None:
                        oriented[neighbour_index] = candidate
                        queue.append(neighbour_index)
                        continue

                    if _has_directed_edge(assigned, edge):
                        raise ValueError("shape_from_metric requires an orientable mesh")

    return np.asarray(oriented, dtype=np.int64)


def _intrinsic_triangle_coordinates(
    face: np.ndarray,
    edge_lookup: dict[frozenset, float],
) -> tuple[np.ndarray, float, np.ndarray]:
    i, j, k = [int(v) for v in face]
    lij = edge_lookup[frozenset({i, j})]
    ljk = edge_lookup[frozenset({j, k})]
    lki = edge_lookup[frozenset({k, i})]

    xk = (lki**2 - ljk**2 + lij**2) / (2.0 * lij)
    yk_sq = lki**2 - xk**2
    if yk_sq <= 1e-20:
        raise ValueError("shape_from_metric requires non-degenerate target triangles")
    yk = np.sqrt(yk_sq)

    coords = np.array(
        [
            [0.0, 0.0],
            [lij, 0.0],
            [xk, yk],
        ],
        dtype=np.float64,
    )
    basis = np.column_stack((coords[1] - coords[0], coords[2] - coords[0]))
    area = 0.5 * lij * yk
    return coords, area, np.linalg.inv(basis)


def _build_face_data(
    mesh: TriangleMesh,
    target_edge_lengths: dict[frozenset, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    oriented_faces = _build_oriented_faces(mesh)
    face_coords = np.zeros((len(oriented_faces), 3, 2), dtype=np.float64)
    face_areas = np.zeros(len(oriented_faces), dtype=np.float64)
    face_basis_inv = np.zeros((len(oriented_faces), 2, 2), dtype=np.float64)

    for face_index, face in enumerate(oriented_faces):
        coords, area, basis_inv = _intrinsic_triangle_coordinates(face, target_edge_lengths)
        face_coords[face_index] = coords
        face_areas[face_index] = area
        face_basis_inv[face_index] = basis_inv

    return oriented_faces, face_coords, face_areas, face_basis_inv


def _closest_frame(jacobian: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(jacobian, full_matrices=False)
    return U @ Vt


def _local_step(
    vertices: np.ndarray,
    oriented_faces: np.ndarray,
    face_coords: np.ndarray,
    face_basis_inv: np.ndarray,
) -> np.ndarray:
    frames = np.zeros((len(oriented_faces), 3, 2), dtype=np.float64)

    for face_index, face in enumerate(oriented_faces):
        p = vertices[face]
        world_basis = np.column_stack((p[1] - p[0], p[2] - p[0]))
        jacobian = world_basis @ face_basis_inv[face_index]
        frames[face_index] = _closest_frame(jacobian)

    return frames


def _average_edge_targets(
    edges_arr: np.ndarray,
    user_edge_weights: np.ndarray,
    oriented_faces: np.ndarray,
    face_coords: np.ndarray,
    face_frames: np.ndarray,
    face_areas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    edge_index = {frozenset(edge): idx for idx, edge in enumerate(edges_arr)}
    accum = np.zeros((len(edges_arr), 3), dtype=np.float64)
    weight_sum = np.zeros(len(edges_arr), dtype=np.float64)

    for face_index, face in enumerate(oriented_faces):
        coords = face_coords[face_index]
        frame = face_frames[face_index]
        face_weight = face_areas[face_index]

        local_edges = (
            ((int(face[0]), int(face[1])), coords[1] - coords[0]),
            ((int(face[1]), int(face[2])), coords[2] - coords[1]),
            ((int(face[2]), int(face[0])), coords[0] - coords[2]),
        )

        for (src, dst), local_vec in local_edges:
            edge = frozenset({src, dst})
            idx = edge_index[edge]
            target_vec = frame @ local_vec

            canonical_src, canonical_dst = edges_arr[idx]
            if (src, dst) != (int(canonical_src), int(canonical_dst)):
                target_vec = -target_vec

            accum[idx] += face_weight * target_vec
            weight_sum[idx] += face_weight

    positive = weight_sum > 0
    target_vectors = np.zeros_like(accum)
    target_vectors[positive] = accum[positive] / weight_sum[positive, None]
    edge_strength = np.maximum(weight_sum, 1e-12) * user_edge_weights
    return target_vectors, edge_strength


def _build_global_system(
    n_vertices: int,
    edges_arr: np.ndarray,
    edge_vectors: np.ndarray,
    edge_strength: np.ndarray,
    reference_positions: np.ndarray,
    *,
    fixed_vertices: Optional[np.ndarray],
    fixed_positions: Optional[np.ndarray],
    vertex_weights: Optional[np.ndarray],
    lambda_boundary: float,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs_rows: list[np.ndarray] = []
    row_index = 0

    for edge_idx, (src, dst) in enumerate(edges_arr):
        weight = edge_strength[edge_idx]
        if weight <= 0:
            continue
        scale = np.sqrt(weight)
        rows.extend((row_index, row_index))
        cols.extend((int(src), int(dst)))
        data.extend((-scale, scale))
        rhs_rows.append(scale * edge_vectors[edge_idx])
        row_index += 1

    active_boundary = (
        fixed_vertices is not None
        and fixed_positions is not None
        and vertex_weights is not None
        and len(fixed_vertices) > 0
        and lambda_boundary > 0
    )

    if active_boundary:
        boundary_scale = np.sqrt(
            lambda_boundary * max(len(edges_arr), 1) / max(len(fixed_vertices), 1)
        )
        for local_index, vertex in enumerate(fixed_vertices):
            scale = boundary_scale * np.sqrt(vertex_weights[local_index])
            rows.append(row_index)
            cols.append(int(vertex))
            data.append(scale)
            rhs_rows.append(scale * fixed_positions[local_index])
            row_index += 1
    elif n_vertices > 0:
        rows.append(row_index)
        cols.append(0)
        data.append(1.0)
        rhs_rows.append(reference_positions[0])
        row_index += 1

    matrix = sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(row_index, n_vertices),
        dtype=np.float64,
    ).tocsr()
    rhs = np.vstack(rhs_rows) if rhs_rows else np.zeros((0, 3), dtype=np.float64)
    return matrix, rhs


def _global_step(
    n_vertices: int,
    edges_arr: np.ndarray,
    edge_vectors: np.ndarray,
    edge_strength: np.ndarray,
    reference_positions: np.ndarray,
    *,
    fixed_vertices: Optional[np.ndarray],
    fixed_positions: Optional[np.ndarray],
    vertex_weights: Optional[np.ndarray],
    lambda_boundary: float,
    gtol: float,
    max_iter: int,
) -> np.ndarray:
    matrix, rhs = _build_global_system(
        n_vertices,
        edges_arr,
        edge_vectors,
        edge_strength,
        reference_positions,
        fixed_vertices=fixed_vertices,
        fixed_positions=fixed_positions,
        vertex_weights=vertex_weights,
        lambda_boundary=lambda_boundary,
    )

    vertices = np.zeros((n_vertices, 3), dtype=np.float64)
    linear_iter = max(1000, 4 * max_iter, 4 * n_vertices)
    for axis in range(3):
        solution = splinalg.lsqr(
            matrix,
            rhs[:, axis],
            atol=gtol,
            btol=gtol,
            iter_lim=linear_iter,
        )[0]
        vertices[:, axis] = solution
    return vertices


def shape_from_metric(
    mesh: TriangleMesh,
    target_edge_lengths,
    *,
    initial_positions=None,
    fixed_vertices=None,
    fixed_positions=None,
    vertex_weights=None,
    edge_weights=None,
    lambda_boundary: float = 0.01,
    fix_scale: bool = False,
    max_iter: int = 100,
    gtol: float = 1e-6,
    verbose: int = 1,
) -> EmbedResult:
    """Recover a 3D embedding from target edge lengths.

    The implementation follows the core local/global structure of the Shape
    from Metric workflow: a local facewise rigid fit from the intrinsic metric,
    followed by a global Poisson-style reconstruction of vertex positions.
    This is a reference-inspired minimal version, not a full port of the
    quaternion / spin-structure solver from the Houdini code.
    """

    del fix_scale  # The metric fixes scale in this reconstruction path.

    edges_arr, lengths_arr = prepare_edge_data(mesh, target_edge_lengths)
    user_edge_weights = prepare_edge_weights(len(edges_arr), edge_weights)
    edge_lookup = {
        frozenset(edge): float(length)
        for edge, length in zip(edges_arr, lengths_arr)
    }

    if mesh.n_vertices == 0:
        empty = np.zeros((0, 3), dtype=np.float64)
        return EmbedResult(vertices=empty, scale_factor=1.0, cost=0.0, history=[0.0], mesh=mesh.with_vertices(empty))

    reference_positions = mesh.vertices.copy()
    if initial_positions is not None:
        reference_positions = _validate_vertex_array(
            "initial_positions",
            initial_positions,
            mesh.vertices.shape,
        )

    active_fixed_vertices = None
    active_fixed_positions = None
    active_vertex_weights = None
    if fixed_vertices is not None and lambda_boundary > 0:
        active_fixed_vertices = _validate_vertex_indices(
            "fixed_vertices",
            fixed_vertices,
            mesh.n_vertices,
        )
        if len(active_fixed_vertices) > 0:
            if fixed_positions is None:
                active_fixed_positions = reference_positions[active_fixed_vertices]
            else:
                active_fixed_positions = _validate_vertex_array(
                    "fixed_positions",
                    fixed_positions,
                    (len(active_fixed_vertices), 3),
                )
            if vertex_weights is None:
                active_vertex_weights = np.ones(len(active_fixed_vertices), dtype=np.float64)
            else:
                active_vertex_weights = _validate_vertex_weights(
                    vertex_weights,
                    len(active_fixed_vertices),
                )

    oriented_faces, face_coords, face_areas, face_basis_inv = _build_face_data(
        mesh,
        edge_lookup,
    )

    if verbose > 0:
        fixed_count = 0 if active_fixed_vertices is None else len(active_fixed_vertices)
        print(
            f"Shape-from-metric: {mesh.n_vertices} vertices, {len(oriented_faces)} faces, fixed={fixed_count}"
        )

    current = reference_positions.copy()
    history: list[float] = []
    use_alignment = active_fixed_vertices is None or len(active_fixed_vertices) == 0
    alignment_weights = np.ones(mesh.n_vertices, dtype=np.float64)

    for iteration in range(max_iter):
        face_frames = _local_step(current, oriented_faces, face_coords, face_basis_inv)
        edge_vectors, edge_strength = _average_edge_targets(
            edges_arr,
            user_edge_weights,
            oriented_faces,
            face_coords,
            face_frames,
            face_areas,
        )

        updated = _global_step(
            mesh.n_vertices,
            edges_arr,
            edge_vectors,
            edge_strength,
            reference_positions,
            fixed_vertices=active_fixed_vertices,
            fixed_positions=active_fixed_positions,
            vertex_weights=active_vertex_weights,
            lambda_boundary=lambda_boundary,
            gtol=gtol,
            max_iter=max_iter,
        )

        if use_alignment:
            updated, _ = _apply_similarity_alignment(
                updated,
                updated,
                reference_positions,
                alignment_weights,
                allow_scaling=False,
            )

        cost = edge_residual_cost(
            updated,
            edges_arr,
            lengths_arr,
            edge_weights=user_edge_weights,
            scale_factor=1.0,
        )
        history.append(cost)

        if verbose >= 2 and (iteration % 10 == 0 or iteration == max_iter - 1):
            print(f"  iter {iteration:5d}  cost={cost:.8e}")

        if iteration > 0:
            rel = abs(history[-1] - history[-2]) / max(abs(history[-2]), 1e-15)
            displacement = np.linalg.norm(updated - current) / max(np.linalg.norm(current), 1e-15)
            if rel < gtol or displacement < gtol:
                current = updated
                if verbose >= 1:
                    print(f"  Converged at iteration {iteration}")
                break

        current = updated

    final_cost = history[-1] if history else 0.0
    if verbose > 0:
        print(f"  cost={final_cost:.8e}, beta=1.000000, iters={len(history)}")

    result_mesh = mesh.with_vertices(current)
    return EmbedResult(
        vertices=current,
        scale_factor=1.0,
        cost=final_cost,
        history=history,
        mesh=result_mesh,
    )
