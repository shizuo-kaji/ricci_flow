"""Direct curvature flow: optimise vertex positions to achieve target curvatures.

An alternative to the two-stage Ricci flow + embedding pipeline.
Directly moves vertices so that angle defects match target Gaussian curvatures.
Uses PyTorch autograd.
"""

import math
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, Union

from .mesh import TriangleMesh


def _safe_arccos(x):
    """Numerically safe arccos for torch tensors."""
    return torch.arccos(torch.clamp(x, -1 + 1e-7, 1 - 1e-7))


def _build_vertex_star(n_verts, faces):
    """Build vertex star neighbourhood structure.

    For vertex i, N[i][k] = [a, b] means (a, i, b) are consecutive
    edges around i in face k.

    Returns:
        list of (K, 2) numpy arrays, one per vertex.
    """
    F = [[] for _ in range(n_verts)]
    for f in faces:
        f = list(f)
        for i in range(len(f) - 1):
            F[f[i]].append([f[i - 1], f[i + 1]])
        F[f[-1]].append([f[-2], f[0]])
    return [np.array(F[i]) for i in range(n_verts)]


def _compute_gaussian_curvature(vert, N, indices, device):
    """Compute Gaussian curvature at specified vertices via angle defect.

    Args:
        vert: (N, 3) torch tensor of vertex positions.
        N: vertex star from _build_vertex_star.
        indices: list of vertex indices to compute curvature at.
        device: torch device.

    Returns:
        list of torch scalar tensors (one per index), with autograd graph.
    """
    K = []
    for i in indices:
        Ni = torch.tensor(N[i], dtype=torch.long, device=device)
        L0 = torch.sum((vert[Ni[:, 0]] - vert[i]) ** 2, dim=1)
        L1 = torch.sum((vert[Ni[:, 1]] - vert[i]) ** 2, dim=1)
        D = torch.sum((vert[Ni[:, 1]] - vert[Ni[:, 0]]) ** 2, dim=1)
        cos_angles = (L0 + L1 - D) / (2 * torch.sqrt(L0 * L1))
        K.append(2 * math.pi - torch.sum(_safe_arccos(cos_angles)))
    return K


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
    import os
    dev = torch.device(device)
    n = mesh.n_vertices

    # Parse target curvature
    K = np.full(n, 4 * np.pi)
    if isinstance(target_curvature, (int, float)):
        K[mesh.interior_vertices] = target_curvature
    elif isinstance(target_curvature, np.ndarray):
        K = target_curvature.copy()
    elif isinstance(target_curvature, str) and os.path.isfile(target_curvature):
        tc = np.loadtxt(target_curvature, delimiter=",")
        K[tc[:, 0].astype(int)] = tc[:, 1]

    # Gauss-Bonnet fill for unspecified vertices
    free_K = K > 2 * np.pi
    if free_K.sum() > 0:
        uK = (2 * mesh.euler_characteristic * np.pi - K[~free_K].sum()) / free_K.sum()
        K[free_K] = uK
        if verbose > 0:
            print(f"  Uniform target K = {uK:.6f} at {free_K.sum()} vertices")

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

    constrained_verts = list(range(n))
    N = _build_vertex_star(n, mesh.faces)

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
    opt = optim_map.get(optimizer, optim_map['Adam'])([coords])

    # Cosine annealing LR
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=max(max_iter // 2, 1))

    history = []
    best_loss = float('inf')
    patience_ctr = 0

    for epoch in range(max_iter):
        def closure():
            opt.zero_grad()
            Kc = _compute_gaussian_curvature(coords, N, constrained_verts, dev)
            loss = sum((Kc[i] - K_t[constrained_verts[i]]) ** 2
                       for i in range(len(Kc))) / len(Kc)
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
        final_K_list = _compute_gaussian_curvature(
            torch.tensor(final_verts, dtype=torch.float64, device=dev),
            N, constrained_verts, dev)
    final_K = np.array([k.item() for k in final_K_list])

    if verbose >= 1:
        mae = np.abs(final_K - K[constrained_verts]).mean()
        print(f"  Final curvature MAE = {mae:.6e}, iters = {epoch+1}")

    result_mesh = mesh.with_vertices(final_verts)

    return CurvatureFlowResult(
        vertices=final_verts,
        curvature=final_K,
        target_curvature=K,
        history=history,
        mesh=result_mesh,
    )
