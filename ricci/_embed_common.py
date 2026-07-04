"""Shared embedding result types and helpers."""

from dataclasses import dataclass, field
from typing import Union

import numpy as np

from .mesh import TriangleMesh


@dataclass
class EmbedResult:
    """Result of metric embedding.

    Attributes:
        vertices: (N, 3) embedded vertex positions.
        scale_factor: global scaling factor applied to squared edge lengths.
        cost: final edge-length residual.
        history: per-iteration or per-stage objective values.
        mesh: new TriangleMesh with embedded vertex positions.
    """

    vertices: np.ndarray
    scale_factor: float
    cost: float
    history: list = field(default_factory=list)
    mesh: TriangleMesh = None


def prepare_edge_data(
    mesh: TriangleMesh,
    target_edge_lengths: Union[dict, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert metric data into arrays ordered like ``mesh.edges``."""

    if mesh.edges:
        edges_arr = np.array([sorted(edge) for edge in mesh.edges], dtype=np.int64)
    else:
        edges_arr = np.empty((0, 2), dtype=np.int64)

    if isinstance(target_edge_lengths, dict):
        missing = [frozenset(edge) for edge in edges_arr if frozenset(edge) not in target_edge_lengths]
        if missing:
            raise KeyError(f"target_edge_lengths is missing {len(missing)} mesh edges")
        lengths_arr = np.array(
            [target_edge_lengths[frozenset(edge)] for edge in edges_arr],
            dtype=np.float64,
        )
    else:
        lengths_arr = np.asarray(target_edge_lengths, dtype=np.float64)

    if lengths_arr.ndim != 1 or len(lengths_arr) != len(edges_arr):
        raise ValueError(
            "target_edge_lengths must be a dict or a 1D array matching mesh.edges"
        )
    if not np.all(np.isfinite(lengths_arr)):
        raise ValueError("target_edge_lengths must be finite")
    if np.any(lengths_arr <= 0):
        raise ValueError("target_edge_lengths must be strictly positive")

    return edges_arr, lengths_arr


def prepare_edge_weights(
    n_edges: int,
    edge_weights: Union[None, np.ndarray],
) -> np.ndarray:
    """Validate edge weights or fall back to uniform weights."""

    if edge_weights is None:
        weights = np.ones(n_edges, dtype=np.float64)
    else:
        weights = np.asarray(edge_weights, dtype=np.float64)

    if weights.ndim != 1 or len(weights) != n_edges:
        raise ValueError("edge_weights must be a 1D array matching mesh.edges")
    if not np.all(np.isfinite(weights)):
        raise ValueError("edge_weights must be finite")
    if np.any(weights < 0):
        raise ValueError("edge_weights must be non-negative")

    return weights


def edge_residual_cost(
    vertices: np.ndarray,
    edges_arr: np.ndarray,
    target_lengths: np.ndarray,
    edge_weights: Union[None, np.ndarray] = None,
    *,
    scale_factor: float = 1.0,
) -> float:
    """Return the squared edge-length residual used by both embedding paths."""

    if len(edges_arr) == 0:
        return 0.0

    weights = prepare_edge_weights(len(edges_arr), edge_weights)
    diff = vertices[edges_arr[:, 0]] - vertices[edges_arr[:, 1]]
    actual_len_sq = np.sum(diff * diff, axis=1)
    residuals = weights * (actual_len_sq - scale_factor * target_lengths**2)
    return float(np.sum(residuals**2))
