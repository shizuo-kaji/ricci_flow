"""Direct curvature flow: optimise vertex positions to achieve target curvatures.

An alternative to the two-stage Ricci flow + embedding pipeline.
Directly moves vertices so that angle defects match target Gaussian curvatures.
Uses PyTorch autograd.
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, Union

from .mesh import TriangleMesh
from .ricci import _parse_target_curvature


def _compute_gaussian_curvature(vertices, triangles, full_angles):
    """Differentiable angle defects, using pi at the boundary."""
    corners = vertices[triangles]
    left = torch.roll(corners, -1, dims=1) - corners
    right = torch.roll(corners, 1, dims=1) - corners
    cross = torch.linalg.cross(left, right, dim=-1)
    angles = torch.atan2(torch.linalg.vector_norm(cross, dim=-1), (left * right).sum(dim=-1))
    return full_angles.scatter_add(0, triangles.flatten(), -angles.flatten())


@dataclass
class CurvatureFlowResult:
    """Result of direct curvature flow.

    Attributes:
        vertices: (N, 3) optimised vertex positions.
        curvature: (N,) final Gaussian curvature at constrained vertices.
        target_curvature: (N,) target curvatures.
        history: per-epoch loss values.
        mesh: TriangleMesh with optimised vertices.
    """
    vertices: np.ndarray
    curvature: np.ndarray
    target_curvature: np.ndarray
    history: list = field(default_factory=list)
    mesh: TriangleMesh = None


def curvature_flow(
    mesh: TriangleMesh,
    target_curvature: Union[float, np.ndarray, str] = 0.0,
    *,
    fixed_vertices: Optional[np.ndarray] = None,
    fixed_positions: Optional[np.ndarray] = None,
    lambda_boundary: float = 0.0,
    strict_boundary: bool = False,
    optimizer: str = 'Adam',
    lr: float = 1e-3,
    max_iter: int = 1000,
    patience: int = 50,
    device: str = 'cpu',
    verbose: int = 1,
) -> CurvatureFlowResult:
    """Directly optimise vertex positions to achieve target curvatures.

    Unlike the Ricci flow + embedding pipeline, this method moves vertices
    directly. It is simpler but converges more slowly and has more local minima.

    Args:
        mesh: input TriangleMesh.
        target_curvature: target Gaussian curvature specification.
            - float: uniform for interior; boundary inferred via Gauss-Bonnet.
            - ndarray: per-vertex (values > 2*pi are inferred).
            - str: path to CSV with (vertex_id, curvature) rows.
        fixed_vertices: indices of vertices to keep fixed.
        fixed_positions: (F, 3) positions for fixed vertices.
        lambda_boundary: weight for soft boundary position constraint.
        strict_boundary: if True, hard-project fixed vertices each step.
        optimizer: 'Adam' (default), 'LBFGS', 'SGD', etc.
        lr: learning rate.
        max_iter: maximum epochs.
        patience: early stopping patience (0 to disable).
        device: 'cpu' or 'cuda'.
        verbose: 0=silent, 1=summary, 2=per-epoch.

    Returns:
        CurvatureFlowResult.
    """
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if not mesh.n_faces:
        raise ValueError("curvature_flow requires a mesh with faces")
    dev = torch.device(device)
    n = mesh.n_vertices
    K, _, _ = _parse_target_curvature(mesh, target_curvature, fix_boundary=False)

    # Fixed vertices
    if fixed_vertices is None:
        if mesh.boundary_vertices:
            fixed_vertices = np.array(mesh.boundary_vertices)
            fixed_positions = mesh.vertices[fixed_vertices]
        else:
            fixed_vertices = np.array([0])
            fixed_positions = mesh.vertices[fixed_vertices]
    elif fixed_positions is None:
        fixed_positions = mesh.vertices[fixed_vertices]

    triangles = torch.tensor(mesh.triangles, dtype=torch.long, device=dev)
    full_angles = torch.full((n,), 2*np.pi, dtype=torch.float64, device=dev)
    full_angles[mesh.boundary_vertices] = np.pi

    if verbose > 0:
        print(f"  V={n}, F={mesh.n_faces}, fixed={len(fixed_vertices)}")

    # Setup PyTorch optimisation
    K_t = torch.tensor(K, dtype=torch.float64, device=dev)
    fixed_pos_t = torch.tensor(fixed_positions, dtype=torch.float64, device=dev)
    fixed_idx_t = torch.tensor(fixed_vertices, dtype=torch.long, device=dev)
    coords = nn.Parameter(
        torch.tensor(mesh.vertices, dtype=torch.float64, device=dev).contiguous())

    optim_map = {
        'Adam': lambda p: torch.optim.Adam(p, lr=lr),
        'AdamW': lambda p: torch.optim.AdamW(p, lr=lr),
        'SGD': lambda p: torch.optim.SGD(p, lr=lr, momentum=0.9),
        'LBFGS': lambda p: torch.optim.LBFGS(
            p, lr=lr, max_iter=20, line_search_fn='strong_wolfe'),
    }
    if optimizer not in optim_map:
        raise ValueError(f'Unknown optimizer: {optimizer}')
    opt = optim_map[optimizer]([coords])

    # Cosine annealing LR
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=max(max_iter // 2, 1))

    history = []
    best_loss = float('inf')
    patience_ctr = 0

    for epoch in range(max_iter):
        def closure():
            opt.zero_grad()
            Kc = _compute_gaussian_curvature(coords, triangles, full_angles)
            loss = torch.mean((Kc - K_t) ** 2)
            if lambda_boundary > 0:
                loss = loss + lambda_boundary * torch.sum(
                    (fixed_pos_t - coords[fixed_idx_t]) ** 2)
            loss.backward()
            return loss

        if optimizer == 'LBFGS':
            loss_val = opt.step(closure)
        else:
            loss_val = closure()
            opt.step()

        if strict_boundary:
            with torch.no_grad():
                coords[fixed_idx_t] = fixed_pos_t

        scheduler.step()
        lv = loss_val.item()
        history.append(lv)

        if verbose >= 2 and (epoch % 20 == 0 or epoch == max_iter - 1):
            print(f"  epoch {epoch:5d}  loss={lv:.8e}  lr={opt.param_groups[0]['lr']:.6f}")

        if patience > 0:
            if lv < best_loss - 1e-10:
                best_loss = lv
                patience_ctr = 0
            else:
                patience_ctr += 1
            if patience_ctr >= patience:
                if verbose >= 1:
                    print(f"  Early stopping at epoch {epoch}")
                break

    # Collect results
    final_verts = coords.detach().cpu().numpy()
    with torch.no_grad():
        final_K = _compute_gaussian_curvature(coords, triangles, full_angles).cpu().numpy()

    if verbose >= 1:
        mae = np.abs(final_K - K).mean()
        print(f"  Final curvature MAE = {mae:.6e}, iters = {epoch+1}")

    result_mesh = mesh.with_vertices(final_verts)

    return CurvatureFlowResult(
        vertices=final_verts,
        curvature=final_K,
        target_curvature=K,
        history=history,
        mesh=result_mesh,
    )
