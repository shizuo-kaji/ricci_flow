# ricci — Discrete Ricci Flow on Triangle Meshes

Design surfaces with prescribed Gaussian curvature via circle packing Ricci flow.

**[Interactive Demo](https://shizuo-kaji.github.io/ricci_flow/)** — Paint target curvature on a sample mesh or your own PLY file and watch the reconstructed shape change in your browser.

Based on: S. Kaji and J. Zhang, *Free-form Design of Discrete Architectural Surfaces by use of Circle Packing*, [arXiv:2103.07584](https://arxiv.org/abs/2103.07584).
Original Ricci flow code by [Harrison Chapman](https://github.com/hchapman/ricci-flow).

## Ricci Flow Lab

**[Try here](https://shizuo-kaji.github.io/ricci_flow/)**

Choose an **Input mesh** or click **Open PLY / OBJ** to load a triangle mesh from your device. PLY 1.0 input can be ASCII, little-endian binary or big-endian binary. The reader uses the vertex coordinates and triangle indices and skips extra properties such as colours and normals. Point clouds need a surface mesh, and polygonal faces need triangulation before import. Files and calculations stay in the browser.

Try the **Trefoil knot** for a closed, knotted tube, the **Catenoid** for a saddle-shaped neck between two circular rims, or the **Ruffled disk** for an open sheet with five waves along its edge. The torus, dome and bunny are also available.

Under **Paint preset**, choose **Bullseye rings**, **Opposing poles** or **Interference waves** and click **Apply**. Try **Trefoil knot** with **Interference waves** to see how ripples wrap around the tube. The **Change** colours show the target minus the input angle defect. Select **Target** to see the absolute values. The patterns use distances along mesh edges and apply independently on each connected component, including on imported meshes. A preset replaces the current painting. Brush over it to customize the target or use **Undo** once to restore the previous painting.

To prescribe values by vertex ID, select **Per-vertex values (CSV)**. **Preview intermediate shapes** reconstructs the mesh at the interval set by **Preview every**. The default computes only the final shape. Use **Shape history** to inspect a saved iteration, **Check vertex stars** to test it for pinch points, and **Export PLY** to save that shape.

Choose **Paint on mesh** under **Target Gaussian curvature**. Enter a signed angle defect in radians and drag across the mesh. **Single vertex** picks one visible vertex per dab. A wider brush follows edges up to the selected number of rings. Dots mark the vertices with explicit targets. Shift-drag or select **Orbit** to turn the mesh. **Erase** releases vertices, **Undo** restores the previous stroke, and **Clear** releases the entire mesh.

Painted targets use 100% strength. Each untouched component keeps its input curvature. On an edited component, the unpainted vertices share the correction needed to keep the total at 2πχ, in proportion to how much their angle defects can change in the required direction. The editor flags incompatible values before a run. After optimization, **Edit painted target** reopens the same field on the input mesh. You can switch back to the last shape until you change the target. Loading a new mesh starts a fresh painting.

The web app lives in [web/](web/). To run it locally, start the following command from the repository root, then open [the local demo](http://127.0.0.1:8000/).

```sh
python -m http.server 8000 --bind 127.0.0.1 --directory web
```

## Installation

```sh
python -m pip install -e .
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
| `fix_boundary` | `False` | target interior curvature only. SGD/Newton also hold boundary radii fixed |
| `boundary_constraint` | `'edge'` | `'edge'` or `'radius'` |
| `boundary_weight` | `0.01` | weight for boundary constraints |
| `gtol` | `1e-6` | convergence tolerance |
| `max_iter` | `None` | maximum steps for SGD/Newton or function evaluations for TRF/LM |
| `preview_every` | `0` | positive interval in accepted iterations for shape reconstruction. Includes the final metric |
| `preview_options` | `None` | keyword arguments for the preview's `shape_from_metric` call |
| `on_preview` | `None` | callback to stream each reconstructed frame as it becomes available |

Returns `RicciFlowResult` with `.edge_lengths`, `.curvature`, `.target_curvature`, `.scale_factor`, `.history` and `.previews`. History begins with the initial metric and records the sum of absolute curvature errors on prescribed vertices at accepted iterates. TRF and LM apply soft boundary penalties controlled by `boundary_weight`.

### `embed(mesh, target_edge_lengths, **opts) -> EmbedResult`

Embed edge lengths into R^3 as vertex positions.

| Parameter | Default | Description |
|---|---|---|
| `target_edge_lengths` | (required) | `dict` or `ndarray` of target lengths |
| `method` | `'shape_from_metric'` | local/global initialization and sparse refinement, or `'optimization'` for the PyTorch optimizer |
| `lambda_boundary` | `0.01` | boundary position constraint weight |
| `lambda_convexity` | `0.0` | soft convexity (z-monotonicity) weight |
| `lambda_smoothness` | `0.0` | mean curvature smoothing weight |
| `bending_weight` | `0.0` | strength of the interior-edge bending penalty in `shape_from_metric` |
| `bending_target` | `'flat'` | prefer zero dihedral angles, or use `'reference'` to prefer the folds in the input mesh |
| `fix_scale` | `False` | fix global scale factor to 1 |
| `optimizer` | `'LBFGS'` | `'LBFGS'`, `'Adam'`, `'SGD'` |
| `device` | `'cpu'` | `'cpu'` or `'cuda'` |
| `gtol` | `1e-6` | convergence tolerance |

Returns `EmbedResult` with `.vertices`, `.scale_factor`, `.cost`, `.relative_edge_rms`, `.history` and `.mesh`. An RMS of `0.01` means 1% relative edge-length error over all edges. For the free-scale optimizer, the reference length is the input target multiplied by `sqrt(scale_factor)`.

`method='optimization'` keeps the original PyTorch/L-BFGS edge-length fitting. `method='shape_from_metric'` performs batched per-face rigid fits, reuses a sparse Poisson factorization, and refines the 3D coordinates with a sparse analytic Jacobian. The target lengths set the physical scale. `initial_positions` chooses a starting configuration, and `fixed_vertices` with `fixed_positions` supplies soft positional constraints.

If a flat configuration stalls, the solver tests out-of-plane displacements, including candidates derived from distances to neighbouring vertices. This uses the trilateration idea in Kim and Santangelo's work. Multiple configurations can realize the same metric, so the starting shape and anchors can influence the fold. Fixed connectivity can also leave residual stretch or self-intersections. [Design Principles for Realizable Discrete Surface Embeddings in Physical Systems (2025)](https://arxiv.org/abs/2505.07696)

Periodic previews reuse the previous realization as the starting shape, adapting the continuation strategy in Liu et al., §5.3. Their convex-cap uniqueness result applies to the paper's geometric setting. [Curvature Driven Deformable Surface Modeling based on Ricci flow and Alexandrov Rigidity Theory (2026)](https://epubs.siam.org/doi/10.1137/1.9781611979138.21)

### `shape_from_metric(mesh, target_edge_lengths, **opts) -> EmbedResult`

Standalone entry point for the shape-from-metric embedding used by `embed(..., method='shape_from_metric')`.

### Choosing a fold and checking the result

Two triangles sharing an edge can fold through many angles with exactly the same edge lengths. A positive `bending_weight` lets the reconstruction prefer a fold. **Flat** encourages adjacent faces to be coplanar. **Input dihedral angles** favors the signed folds measured on the original mesh, even when a preview uses another starting shape. Both options are under **Solver settings** in the web lab.

```python
from ricci import immersion_diagnostics, shape_from_metric

fit = shape_from_metric(mesh, result.edge_lengths,
                        bending_weight=0.001, bending_target="reference")
stars = immersion_diagnostics(fit.mesh)
print(fit.relative_edge_rms, stars.pinch_vertices, stars.indeterminate_vertices)
```

The added energy sums the squared signed angle differences wrapped to [−π, π], weighted on each interior edge by its squared target length divided by one third of the two incident target triangle areas combined. Incompatible lengths and preferred angles produce a tradeoff. For example, on `demo/dome5.ply` with uniform interior target curvature 0.05 and a fixed boundary, increasing the Python flat-fold weight from 0 to 0.01 reduced the RMS dihedral angle from 0.149 to 0.120 radians, while the relative edge RMS increased to 0.00586. The browser uses relative edge-length residuals and Python uses squared-length residuals, so the same positive weight can produce different fits.

### Shape from Metric and subsequent research

- **Chern, Knöppel, Pinkall & Schröder (2018).** Their full reconstruction transports face spinors and selects a regular homotopy class. We implemented their extrinsic diagnostic from §§2.1–2.2. Transport around a closed vertex star gives a quaternion sign. A positive sign identifies the figure-eight pinch obstruction. The diagnostic reports open boundary stars separately and marks degenerate or ambiguous stars as indeterminate. A regular parity result can coexist with local overlaps or distant triangle intersections. Our coordinate solver does not prescribe a regular homotopy class. [Shape from Metric](https://cseweb.ucsd.edu/~alchern/projects/ShapeFromMetric/)
- **Chern, Knöppel, Pedit, Pinkall & Schröder (2019 preprint, 2021 chapter).** The authors develop spinor functionals for nearly conformal immersions in a chosen regular homotopy class. Their isometric limiting shapes are numerical observations. This formulation is relevant when control of the immersion class is an explicit design input. [Finding Conformal and Isometric Immersions of Surfaces](https://cseweb.ucsd.edu/~alchern/projects/FindingConformalImmersions/)
- **Sassen, Schumacher, Rumpf & Crane (2024).** Their near-isometric examples combine membrane stretch, weak bending and nonlocal tangent-point repulsion. We adapted the local hinge weight and angle penalty in §3.1.2 and the flat-fold preference in §7.3.2. Collision avoidance in their examples additionally relies on the nonlocal energy, which this solver omits. [Repulsive Shells](https://www.cs.cmu.edu/~kmcrane/Projects/RepulsiveShells/index.html)
- **Gao, Hu, Li & Zhang (2026 preprint).** Their high-order finite-element formulation gives convergence estimates for the positive-curvature Weyl embedding problem and for Ricci flow in that setting. Those hypotheses do not cover the flat torus or the open triangular panels supported here. [Finite element methods for isometric embedding of Riemannian manifolds](https://arxiv.org/abs/2602.18722)

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
| Convergence | Check intrinsic curvature and final edge residual separately | May get stuck in local minima |
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

### Shape-from-metric embedding

```python
result = ricci_flow(mesh, target_curvature=0.1, fix_boundary=True)
embedded = embed(mesh, result.edge_lengths, method='shape_from_metric')
```

### Watch the metric become a shape

Set a positive interval to enable the extra embedding solves. For a notebook, this displays each accepted snapshot in the same output cell.

```python
from IPython.display import display
from ricci import TriangleMesh, plot_mesh, ricci_flow

mesh = TriangleMesh.from_ply("demo/dome.ply")
view = display(plot_mesh(mesh, title="Input"), display_id=True)

def show_frame(frame):
    title = f"Iteration {frame.iteration}    curvature L1 = {frame.curvature_error:.3g} rad"
    view.update(plot_mesh(frame.shape.mesh, title=title))

run = ricci_flow(mesh, 0.1, fix_boundary=True, preview_every=2,
                 on_preview=show_frame, verbose=0)
```

Omit `on_preview` to keep snapshots in `run.previews`. Each has its iteration, curvature L1 error, per-vertex intrinsic curvature, physical edge lengths and full embedding result (`frame.shape`). A callback streams frames without retaining them in the result. The last frame is emitted once even if the solve stops between intervals or at the initial state. The least-squares flow can pass through edge lengths that violate a triangle inequality. Such intermediate states are listed in `run.skipped_preview_iterations`. A degenerate final metric raises an error.

### Per-vertex curvature from CSV

```python
result = ricci_flow(mesh, 'demo/dome_targetK_hat.csv', method='trf')
embedded = embed(mesh, result.edge_lengths, lambda_boundary=0)
```

CSV rows contain `vertex_id, gaussian_curvature` in radians with zero-based vertex IDs. A value of `100` keeps the input angle defect. Other values above 2π and omitted vertices share the remaining curvature equally within each connected component.

### Closed surface (torus)

```python
torus = TriangleMesh.from_obj('demo/torus.obj')
result = ricci_flow(torus, method='newton', scheme='inversive')
embedded = embed(torus, result.edge_lengths, lambda_boundary=0)
```

### Convexity-enforced embedding

```python
embedded = embed(mesh, result.edge_lengths, method='optimization', lambda_convexity=1.0)
```

### GPU acceleration

```python
embedded = embed(mesh, result.edge_lengths, method='optimization', device='cuda')
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

The **embedding** stage fits vertex positions to the optimised edge lengths. The default solver alternates facewise rigid fits and Poisson reconstruction before refining the coordinates. The PyTorch path supports L-BFGS and optional convexity and smoothing penalties.

## Verification

```sh
python -m unittest discover -s tests
node --test
```

As a reconstruction benchmark, take the 96 vertices of `demo/dome5.ply`, multiply the input coordinates by `[0.9, 1.0, 1.3]`, and use the deformed mesh's edge lengths as the target. Starting from the original dome, with no position constraints, `max_iter=100` and `gtol=1e-6`, the previous local/global solver gave relative edge RMS `1.96e-3`. The refined solver gave `3.72e-9` on the same target. This benchmark has a known realization on the original connectivity.

## Dependencies

- numpy, scipy
- torch (>= 2.0)
- plyfile
- matplotlib, seaborn
