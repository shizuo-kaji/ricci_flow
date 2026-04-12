# ricci — Discrete Ricci Flow on Triangle Meshes

Design surfaces with prescribed Gaussian curvature via circle packing Ricci flow.

Based on: S. Kaji and J. Zhang, *Free-form Design of Discrete Architectural Surfaces by use of Circle Packing*, [arXiv:2103.07584](https://arxiv.org/abs/2103.07584).
Original Ricci flow code by [Harrison Chapman](https://github.com/hchapman/ricci-flow).

## Installation

```
pip install numpy scipy torch plyfile matplotlib seaborn
```

## Quick start

```python
from ricci import TriangleMesh, ricci_flow, embed

mesh = TriangleMesh.from_ply('demo/dome.ply')
result = ricci_flow(mesh, target_curvature=0.1, fix_boundary=True)
embedded = embed(mesh, result.edge_lengths)
embedded.mesh.save_ply('result.ply')
```

See `examples.ipynb` for a full walkthrough.

## Pipeline

The approach splits surface design into two stages:

1. **Ricci flow** (`ricci_flow`) — find edge lengths producing target Gaussian curvatures, preserving the conformal class.
2. **Embedding** (`embed`) — find 3D vertex positions realising those edge lengths.

This factoring separates the convex (metric) and non-convex (embedding) parts of the problem. An alternative single-stage approach is also available via `curvature_flow`.

## API reference

### `ricci_flow(mesh, target_curvature, **opts) -> RicciFlowResult`

Find edge lengths achieving target Gaussian curvatures.

| Parameter | Default | Description |
|---|---|---|
| `target_curvature` | `0.0` | `float` (uniform), `ndarray` (per-vertex), or CSV path |
| `method` | `'trf'` | `'sgd'`, `'newton'`, `'lm'`, `'trf'` |
| `scheme` | `'combinatorial'` | `'inversive'`, `'thurston'`, `'thurston2'`, `'combinatorial'`, `float`, or CSV path |
| `fix_boundary` | `False` | keep boundary radii unchanged |
| `boundary_constraint` | `'edge'` | `'edge'` or `'radius'` |
| `boundary_weight` | `0.01` | weight for boundary constraints |
| `gtol` | `1e-6` | convergence tolerance |

Returns `RicciFlowResult` with `.edge_lengths`, `.curvature`, `.target_curvature`, `.scale_factor`, `.history`.

### `embed(mesh, target_edge_lengths, **opts) -> EmbedResult`

Embed edge lengths into R^3 as vertex positions (PyTorch, L-BFGS).

| Parameter | Default | Description |
|---|---|---|
| `target_edge_lengths` | (required) | `dict` or `ndarray` of target lengths |
| `lambda_boundary` | `0.01` | boundary position constraint weight |
| `lambda_convexity` | `0.0` | soft convexity (z-monotonicity) weight |
| `lambda_smoothness` | `0.0` | mean curvature smoothing weight |
| `fix_scale` | `False` | fix global scale factor to 1 |
| `optimizer` | `'LBFGS'` | `'LBFGS'`, `'Adam'`, `'SGD'` |
| `device` | `'cpu'` | `'cpu'` or `'cuda'` |
| `gtol` | `1e-6` | convergence tolerance |

Returns `EmbedResult` with `.vertices`, `.scale_factor`, `.cost`, `.history`, `.mesh`.

### `curvature_flow(mesh, target_curvature, **opts) -> CurvatureFlowResult`

Directly optimise vertex positions to achieve target curvatures. This is a single-stage alternative to `ricci_flow` + `embed`: simpler, but converges more slowly and does not preserve the conformal class. Best for small deformations or when direct vertex control is needed.

| Parameter | Default | Description |
|---|---|---|
| `target_curvature` | `0.0` | same as `ricci_flow` |
| `fixed_vertices` | boundary | vertices to keep fixed |
| `fixed_positions` | from mesh | target positions for fixed vertices |
| `lambda_boundary` | `0.0` | soft boundary position weight |
| `strict_boundary` | `False` | hard-project fixed vertices each step |
| `optimizer` | `'Adam'` | `'Adam'`, `'LBFGS'`, `'SGD'` |
| `lr` | `1e-3` | learning rate |
| `max_iter` | `1000` | maximum epochs |
| `patience` | `50` | early stopping patience (0 to disable) |
| `device` | `'cpu'` | `'cpu'` or `'cuda'` |

Returns `CurvatureFlowResult` with `.vertices`, `.curvature`, `.target_curvature`, `.history`, `.mesh`.

**When to use `curvature_flow` vs `ricci_flow` + `embed`:**

| | `ricci_flow` + `embed` | `curvature_flow` |
|---|---|---|
| Convergence | Guaranteed for admissible targets | May get stuck in local minima |
| Conformal class | Preserved | Not preserved |
| Speed | Fast (convex + non-convex split) | Slow (fully non-convex) |
| Simplicity | Two-stage | Single function call |

### Mesh I/O

```python
mesh = TriangleMesh.from_ply('input.ply')
mesh = TriangleMesh.from_obj('input.obj')
mesh.save_ply('output.ply')
mesh.save_ply('output.ply', vertex_colours=colours)
```

### Visualisation

```python
from ricci import plot_mesh, plot_curvature, plot_convergence, plot_comparison

plot_mesh(mesh, show=True)
plot_curvature(mesh, curvature_array, show=True)
plot_convergence(result.history, show=True)
plot_comparison([K_init, K_target, K_final], labels=['Init', 'Target', 'Final'], show=True)
```

## Examples

### Fixed boundary, uniform curvature

```python
result = ricci_flow(mesh, target_curvature=0.1, fix_boundary=True, scheme='combinatorial')
embedded = embed(mesh, result.edge_lengths)
```

### Per-vertex curvature from CSV

```python
result = ricci_flow(mesh, 'dome_targetK_hat.csv', method='trf')
embedded = embed(mesh, result.edge_lengths, lambda_boundary=0)
```

CSV format: each row is `vertex_id, gaussian_curvature`. Vertices with K > 2pi or not listed are filled uniformly via the Gauss-Bonnet theorem.

### Closed surface (torus)

```python
torus = TriangleMesh.from_obj('demo/torus.obj')
result = ricci_flow(torus, method='newton', scheme='inversive')
embedded = embed(torus, result.edge_lengths, lambda_boundary=0)
```

### Convexity-enforced embedding

```python
embedded = embed(mesh, result.edge_lengths, lambda_convexity=1.0)
```

### GPU acceleration

```python
embedded = embed(mesh, result.edge_lengths, device='cuda')
```

## Circle packing schemes

| Scheme | Description |
|---|---|
| `'combinatorial'` | Uniform edge weights; seeks equilateral triangles |
| `'inversive'` | Preserves geometry via inversive distance |
| `'thurston'` | Thurston's circle packing with averaged radii |
| `'thurston2'` | Thurston's with fixed radius ratio |
| `float` (e.g. `'0.5'`) | Constant edge weight |
| CSV path | Load custom edge weights from file |

## Background

The discrete Gaussian curvature at a vertex is the **angle defect**: 2pi minus the sum of incident face angles (pi minus the sum for boundary vertices). The Gauss-Bonnet theorem constrains the total curvature: sum(K) = 2pi * chi, where chi is the Euler characteristic.

A **circle packing metric** parameterises edge lengths by per-vertex radii r_i and edge weights eta_{ij}:

    L_{ij} = sqrt(2 * r_i * r_j * eta_{ij} + r_i^2 + r_j^2)

The **conformal factor** u = log(r) lives in Euclidean space, making optimisation well-behaved. Ricci flow minimises the modified Ricci energy ||K(u) - K_target||^2 via scipy least-squares with analytical Jacobians.

The **embedding** stage finds vertex positions matching the optimised edge lengths. This is a non-convex problem solved by PyTorch L-BFGS with automatic differentiation.

## Dependencies

- numpy, scipy
- torch (>= 2.0)
- plyfile
- matplotlib, seaborn
