"""Shape from metric: embed edge lengths into R^3.

Given target edge lengths (e.g. from Ricci flow), find vertex positions
in 3D that realise those lengths subject to boundary, convexity, and
smoothness constraints. Uses PyTorch autograd for gradient computation.
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, Union

from .mesh import TriangleMesh


@dataclass
class EmbedResult:
    """Result of metric embedding.

    Attributes:
        vertices: (N, 3) optimised vertex positions.
        scale_factor: optimised global scaling factor (beta).
        cost: final loss value.
        history: per-iteration loss values.
        mesh: new TriangleMesh with optimised vertex positions.
    """
    vertices: np.ndarray
    scale_factor: float
    cost: float
    history: list = field(default_factory=list)
    mesh: TriangleMesh = None


class _EmbedModel(nn.Module):
    """Internal PyTorch model for embedding optimisation."""

    def __init__(self, vertices, edges, target_lengths, edge_weights,
                 fixed_vert_idx, fixed_coords, vert_weights,
                 all_faces, device):
        super().__init__()
        self.device = device
        self.n_verts = len(vertices)
        self.n_edges = len(edges)

        self.coords = nn.Parameter(
            torch.tensor(vertices, dtype=torch.float64, device=device).contiguous())
        self.beta = nn.Parameter(
            torch.tensor(1.0, dtype=torch.float64, device=device))

        self.register_buffer('edge_src',
            torch.tensor(edges[:, 0], dtype=torch.long, device=device))
        self.register_buffer('edge_dst',
            torch.tensor(edges[:, 1], dtype=torch.long, device=device))
        self.register_buffer('target_len_sq',
            torch.tensor(target_lengths**2, dtype=torch.float64, device=device))
        self.register_buffer('edge_weight',
            torch.tensor(edge_weights, dtype=torch.float64, device=device))

        if fixed_vert_idx is not None and len(fixed_vert_idx) > 0:
            self.register_buffer('fixed_idx',
                torch.tensor(fixed_vert_idx, dtype=torch.long, device=device))
            self.register_buffer('fixed_pos',
                torch.tensor(fixed_coords, dtype=torch.float64, device=device))
            self.register_buffer('vert_w',
                torch.tensor(vert_weights, dtype=torch.float64, device=device))
            self.has_boundary = True
        else:
            self.has_boundary = False

        self._build_adjacency(vertices, all_faces)

    def _build_adjacency(self, vertices, faces):
        n = self.n_verts
        adj = [[] for _ in range(n)]
        edge_count = {}
        for f in faces:
            f = list(f)
            for idx in range(len(f)):
                e = frozenset({f[idx], f[(idx + 1) % len(f)]})
                edge_count[e] = edge_count.get(e, 0) + 1
        b_verts = set()
        for e, c in edge_count.items():
            if c == 1:
                b_verts.update(e)
        self.free_verts_list = [i for i in range(n) if i not in b_verts]

        for f in faces:
            f = list(f)
            for i in range(len(f)):
                for j in range(len(f)):
                    if i != j and f[j] not in adj[f[i]]:
                        adj[f[i]].append(f[j])

        max_deg = max((len(a) for a in adj), default=1)
        pad = np.zeros((n, max_deg), dtype=np.int64)
        mask = np.zeros((n, max_deg), dtype=np.float64)
        deg = np.ones(n, dtype=np.float64)
        for i in range(n):
            d = len(adj[i])
            deg[i] = max(d, 1)
            for j, nb in enumerate(adj[i]):
                pad[i, j] = nb
                mask[i, j] = 1.0

        self.register_buffer('adj_pad', torch.tensor(pad, dtype=torch.long, device=self.device))
        self.register_buffer('adj_mask', torch.tensor(mask, dtype=torch.float64, device=self.device))
        self.register_buffer('adj_deg', torch.tensor(deg, dtype=torch.float64, device=self.device))

    def length_loss(self):
        diff = self.coords[self.edge_src] - self.coords[self.edge_dst]
        actual = torch.sum(diff**2, dim=1)
        residuals = self.edge_weight * (actual - self.beta * self.target_len_sq)
        return torch.sum(residuals**2)

    def boundary_loss(self):
        if not self.has_boundary:
            return torch.tensor(0.0, dtype=torch.float64, device=self.device)
        diff = self.fixed_pos - self.coords[self.fixed_idx]
        return torch.sum(self.vert_w[:, None] * diff**2)

    def convexity_loss(self):
        if not self.free_verts_list:
            return torch.tensor(0.0, dtype=torch.float64, device=self.device)
        idx = torch.tensor(self.free_verts_list, dtype=torch.long, device=self.device)
        z = self.coords[:, 2]
        nb_z = z[self.adj_pad[idx]] * self.adj_mask[idx]
        mean_z = nb_z.sum(dim=1) / self.adj_deg[idx]
        u = -z[idx] + mean_z
        swish = u * torch.sigmoid(u)
        return torch.sum(swish**2)

    def smoothness_loss(self):
        idx = torch.arange(self.n_verts, device=self.device)
        nb = self.coords[self.adj_pad] * self.adj_mask.unsqueeze(-1)
        mean_nb = nb.sum(dim=1) / self.adj_deg.unsqueeze(-1)
        diff = self.coords - mean_nb
        return torch.sum(diff**2)

    def total_loss(self, lb, lc, ls):
        ne, nv = self.n_edges, self.n_verts
        nf = len(self.fixed_idx) if self.has_boundary else 1
        loss = self.length_loss()
        if lb > 0 and self.has_boundary:
            loss = loss + np.sqrt(lb * ne / nf) * self.boundary_loss()
        if lc > 0:
            loss = loss + np.sqrt(lc * ne / nv) * self.convexity_loss()
        if ls > 0:
            loss = loss + np.sqrt(ls * ne / nv) * self.smoothness_loss()
        return loss


def embed(
    mesh: TriangleMesh,
    target_edge_lengths: Union[dict, np.ndarray],
    *,
    initial_positions: Optional[np.ndarray] = None,
    fixed_vertices: Optional[np.ndarray] = None,
    fixed_positions: Optional[np.ndarray] = None,
    vertex_weights: Optional[np.ndarray] = None,
    edge_weights: Optional[np.ndarray] = None,
    lambda_boundary: float = 0.01,
    lambda_convexity: float = 0.0,
    lambda_smoothness: float = 0.0,
    fix_scale: bool = False,
    optimizer: str = 'LBFGS',
    lr: float = 0.1,
    max_iter: int = 2000,
    gtol: float = 1e-6,
    device: str = 'cpu',
    verbose: int = 1,
) -> EmbedResult:
    """Embed a discrete metric into R^3.

    Given target edge lengths, finds 3D vertex positions that realise
    those lengths. Uses PyTorch L-BFGS with strong Wolfe line search
    by default.

    Args:
        mesh: TriangleMesh providing connectivity and initial positions.
        target_edge_lengths: target lengths, either:
            - dict mapping frozenset edge -> length, or
            - ndarray matching the order of mesh.edges.
        initial_positions: (N, 3) starting positions (default: mesh.vertices).
        fixed_vertices: indices of fixed boundary vertices.
        fixed_positions: (F, 3) target positions for fixed vertices.
        vertex_weights: (F,) per-vertex boundary constraint weights.
        edge_weights: per-edge weights (default: uniform).
        lambda_boundary: weight for boundary position constraint.
        lambda_convexity: weight for convexity (z-monotonicity) constraint.
        lambda_smoothness: weight for mean-curvature smoothness.
        fix_scale: if True, fix the global scale factor to 1.
        optimizer: 'LBFGS' (default), 'Adam', 'SGD'.
        lr: learning rate.
        max_iter: maximum iterations.
        gtol: convergence tolerance.
        device: 'cpu' or 'cuda'.
        verbose: 0=silent, 1=summary, 2=progress.

    Returns:
        EmbedResult with optimised vertex positions and metadata.
    """
    dev = torch.device(device)

    # Prepare edge arrays
    edges_arr = np.array([[min(e), max(e)] for e in mesh.edges], dtype=np.int64)
    if isinstance(target_edge_lengths, dict):
        lengths_arr = np.array([target_edge_lengths[frozenset(e)] for e in edges_arr])
    else:
        lengths_arr = np.asarray(target_edge_lengths, dtype=np.float64)

    if edge_weights is None:
        ew = np.ones(len(edges_arr))
    else:
        ew = np.asarray(edge_weights, dtype=np.float64)

    # Initial positions
    verts = mesh.vertices.copy() if initial_positions is None else initial_positions.copy()

    # Boundary
    fv = fixed_vertices
    fp = fixed_positions
    vw = vertex_weights
    if fv is not None:
        fv = np.asarray(fv, dtype=np.int64)
        if fp is None:
            fp = verts[fv]
        if vw is None:
            vw = np.ones(len(fv))
    if fv is None or len(fv) < 2:
        fix_scale = True

    if verbose > 0:
        print(f"Embedding: {mesh.n_vertices} vertices, {len(edges_arr)} edges, "
              f"fixed={0 if fv is None else len(fv)}, device={device}")

    # Build model
    model = _EmbedModel(
        verts, edges_arr, lengths_arr, ew,
        fv if lambda_boundary > 0 else None,
        fp if lambda_boundary > 0 else None,
        vw if lambda_boundary > 0 else None,
        mesh.faces, dev)

    if fix_scale:
        model.beta.requires_grad_(False)

    params = [p for p in model.parameters() if p.requires_grad]

    if optimizer == 'LBFGS':
        opt = torch.optim.LBFGS(
            params, lr=lr, max_iter=20, history_size=10,
            line_search_fn='strong_wolfe', tolerance_grad=gtol)
    elif optimizer == 'Adam':
        opt = torch.optim.Adam(params, lr=lr)
    else:
        opt = torch.optim.SGD(params, lr=lr, momentum=0.9)

    history = []
    best = float('inf')
    patience = max(50, max_iter // 10)
    patience_ctr = 0

    for it in range(max_iter):
        def closure():
            opt.zero_grad()
            loss = model.total_loss(lambda_boundary, lambda_convexity, lambda_smoothness)
            loss.backward()
            return loss

        loss_val = opt.step(closure) if optimizer == 'LBFGS' else (closure(), opt.step())[0]
        lv = loss_val.item()
        history.append(lv)

        if verbose >= 2 and (it % 50 == 0 or it == max_iter - 1):
            print(f"  iter {it:5d}  loss={lv:.8e}  beta={model.beta.item():.6f}")

        if len(history) > 1:
            rel = abs(history[-1] - history[-2]) / max(abs(history[-2]), 1e-15)
            if rel < gtol and lv < 1e-3:
                if verbose >= 1:
                    print(f"  Converged at iteration {it}")
                break

        if lv < best - 1e-12:
            best = lv
            patience_ctr = 0
        else:
            patience_ctr += 1
        if patience_ctr > patience and optimizer != 'LBFGS':
            if verbose >= 1:
                print(f"  Early stopping at iteration {it}")
            break

    final_verts = model.coords.detach().cpu().numpy()
    beta = model.beta.item()

    if verbose >= 1:
        print(f"  cost={lv:.8e}, beta={beta:.6f}, iters={it+1}")

    result_mesh = mesh.with_vertices(final_verts)

    return EmbedResult(
        vertices=final_verts,
        scale_factor=beta,
        cost=lv,
        history=history,
        mesh=result_mesh,
    )
