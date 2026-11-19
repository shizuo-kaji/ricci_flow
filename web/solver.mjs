// Intrinsic stage: Euclidean inversive-distance circle packing on a fixed triangulation.
// Faces store edges in opposite-vertex order (jk, ki, ij).
import {HingeBending} from './bending.mjs';

const TAU = 2 * Math.PI;
const sum = a => a.reduce((x, y) => x + y, 0);
const dot = (a, b) => a.reduce((value, x, i) => value + x * b[i], 0);
const rms = a => Math.sqrt(dot(a, a) / a.length);

function cross(u, v) {
  return [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
}

function subtract(a, b) {
  return a.map((x, i) => x - b[i]);
}

export function prepareMesh(vertices, faces) {
  if (!Array.isArray(vertices) || !vertices.length || !Array.isArray(faces) || !faces.length)
    throw new Error('A triangle mesh with vertices and faces is required.');
  if (!vertices.every(v => Array.isArray(v) && v.length === 3 && v.every(Number.isFinite)))
    throw new Error('Vertex coordinates must be finite 3D points.');
  const n = vertices.length;
  const edges = [], edgeFaces = [], lengths = [], faceEdges = [];
  const edgeIds = new Map(), faceKeys = new Set();
  const incident = Array.from({length: n}, () => []);
  const neighbors = Array.from({length: n}, () => []);
  for (const [f, face] of faces.entries()) {
    if (!Array.isArray(face) || face.length !== 3 ||
        !face.every(i => Number.isInteger(i) && i >= 0 && i < n) || new Set(face).size !== 3)
      throw new Error('Each face must contain three distinct valid vertex indices.');
    const key = [...face].sort((a, b) => a - b).join(',');
    if (faceKeys.has(key)) throw new Error('Duplicate triangle in mesh.');
    faceKeys.add(key);
    faceEdges.push(face.map((i, j) => {
      incident[i].push(f);
      const a = face[(j + 1) % 3], b = face[(j + 2) % 3];
      const u = Math.min(a, b), v = Math.max(a, b), key = u + ',' + v;
      if (!edgeIds.has(key)) {
        edgeIds.set(key, edges.length);
        edges.push([u, v]);
        lengths.push(Math.hypot(...subtract(vertices[u], vertices[v])));
        neighbors[u].push(v);
        neighbors[v].push(u);
        edgeFaces.push([]);
      }
      const id = edgeIds.get(key);
      edgeFaces[id].push(f);
      if (edgeFaces[id].length > 2) throw new Error('A mesh edge belongs to more than two faces.');
      return id;
    }));
  }
  if (incident.some(star => !star.length)) throw new Error('The mesh contains unused vertices.');
  if (!lengths.every(x => x > 0 && Number.isFinite(x))) throw new Error('Invalid mesh edge length.');
  const scale = sum(lengths) / edges.length;
  if (!Number.isFinite(scale)) throw new Error('Mesh coordinate range is too large.');
  const base = Float64Array.from(lengths, l => l / scale);
  const boundary = new Uint8Array(n);
  edgeFaces.forEach((fs, e) => {
    if (fs.length === 1) for (const i of edges[e]) boundary[i] = 1;
  });
  // Edge manifoldness alone does not detect a vertex where disconnected fans touch.
  incident.forEach((star, i) => {
    const visited = new Set([star[0]]), stack = [star[0]];
    while (stack.length) {
      const f = stack.pop(), j = faces[f].indexOf(i);
      for (const side of [1, 2]) {
        for (const next of edgeFaces[faceEdges[f][(j + side) % 3]]) {
          if (!visited.has(next)) { visited.add(next); stack.push(next); }
        }
      }
    }
    if (visited.size !== star.length) throw new Error('Disconnected face fans meet at a vertex.');
  });
  const components = [], seen = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    if (seen[i]) continue;
    const ids = [i];
    seen[i] = 1;
    for (let p = 0; p < ids.length; p++) for (const j of neighbors[ids[p]]) {
      if (!seen[j]) { seen[j] = 1; ids.push(j); }
    }
    components.push(ids);
  }
  const radii = new Float64Array(n).fill(Infinity);
  faces.forEach((face, f) => {
    const l = faceEdges[f].map(e => base[e]);
    for (let j = 0; j < 3; j++) {
      const radius = 0.5 * (l[(j + 1) % 3] + l[(j + 2) % 3] - l[j]);
      if (!(radius > 1e-14 * Math.max(...l))) throw new Error('Degenerate or nearly collinear triangle.');
      radii[face[j]] = Math.min(radii[face[j]], radius);
    }
  });
  // Store 2*eta*ri*rj directly to avoid division by tiny initial circle radii.
  const mixing = base.map((l, e) => {
    const [i, j] = edges[e];
    return l*l - radii[i]**2 - radii[j]**2;
  });
  const faceCounts = incident.map(star => star.length);
  return {vertices, faces, n, edges, edgeFaces, faceEdges, faceCounts, components, boundary, scale, base, radii, mixing};
}

export function packingMetric(mesh, u) {
  const {n, edges, faces, faceEdges, radii, mixing, boundary} = mesh;
  const factors = u.map(Math.exp), r = factors.map((s, i) => s * radii[i]);
  const lengths = new Float64Array(edges.length), shares = new Float64Array(edges.length);
  for (let e = 0; e < edges.length; e++) {
    const [i, j] = edges[e];
    const sq = r[i]**2 + r[j]**2 + mixing[e] * factors[i] * factors[j];
    lengths[e] = Math.sqrt(sq);
    shares[e] = 0.5 * (1 + (r[i]**2 - r[j]**2) / sq);
  }
  const curvature = Float64Array.from(boundary, b => b ? Math.PI : TAU);
  const weights = new Float64Array(edges.length);
  for (let f = 0; f < faces.length; f++) {
    const face = faces[f], ids = faceEdges[f];
    const [a,b,c] = ids.map(e => lengths[e]);
    // Heron's four-factor product equals (4*area)^2.
    const s = Math.max(a,b,c), x = a/s, y = b/s, z = c/s;
    const area4 = Math.sqrt((x+y+z) * (-x+y+z) * (x-y+z) * (x+y-z));
    if (!(area4 > 1e-12)) return null;
    const cosNumerators = [y*y+z*z-x*x, z*z+x*x-y*y, x*x+y*y-z*z];
    const cots = cosNumerators.map(value => value / area4);
    const share = (e, i) => i === edges[e][0] ? shares[e] : 1 - shares[e];
    for (let i = 0; i < 3; i++) {
      curvature[face[i]] -= Math.atan2(area4, cosNumerators[i]);
      const j = (i + 1) % 3, k = (i + 2) % 3;
      // d(angle_i)/du_j via derivatives w.r.t. log edge lengths.
      weights[ids[k]] += (cots[j] + cots[k]) * share(ids[i], face[j])
                           - cots[j] * share(ids[k], face[j]);
    }
  }
  return {curvature, lengths, weights};
}

function makeTarget(mesh, initial, {target = 'uniform', targetCurvature, strength = 1}) {
  if (!Number.isFinite(strength) || strength < 0 || strength > 1)
    throw new Error('Target strength must be in [0, 1].');
  const custom = targetCurvature != null;
  if (custom && (targetCurvature.length !== mesh.n || !Array.from(targetCurvature).every(Number.isFinite)))
    throw new Error('Custom curvature must contain one finite value per vertex.');
  if (!custom && !['uniform', 'initial', 'flat-interior'].includes(target))
    throw new Error('Choose a known curvature target or supply custom curvature values.');
  const result = new Float64Array(mesh.n);
  for (const ids of mesh.components) {
    const total = ids.reduce((s, i) => s + initial[i], 0);
    const nBoundary = ids.reduce((s, i) => s + mesh.boundary[i], 0);
    for (const i of ids) {
      let value = total / ids.length;
      if (custom) value = targetCurvature[i];
      else if (target === 'initial') value = initial[i];
      else if (target === 'flat-interior' && nBoundary)
        value = mesh.boundary[i] ? total / nBoundary : 0;
      result[i] = initial[i] + strength * (value - initial[i]);
      const upper = mesh.boundary[i] ? Math.PI : TAU;
      if (result[i] >= upper || result[i] <= upper - Math.PI * mesh.faceCounts[i])
        throw new Error('Target curvature is outside the attainable angle-defect interval at a vertex.');
    }
    const requested = ids.reduce((s, i) => s + result[i], 0);
    // Permit decimal CSV roundoff. Keep the supplied values in diagnostics so the
    // measured RMS still reveals any irreducible component-wide offset.
    const magnitude = ids.reduce((s, i) => s + Math.abs(result[i]), 0);
    if (Math.abs(total - requested) > 1e-7 * Math.max(1, magnitude))
      throw new Error('Target curvature must sum to 2πχ on each connected component (Gauss–Bonnet).');
  }
  return result;
}

// Solve a damped system with a diagonal preconditioner and matrix-free products.
function pcg(rhs, diagonal, multiply, maxIterations, tolerance) {
  const x = new Float64Array(rhs.length), residual = rhs.slice();
  const z = rhs.map((v, i) => v / diagonal[i]), direction = z.slice();
  const out = new Float64Array(x.length);
  let rz = dot(residual, z);
  const stop = tolerance**2 * dot(rhs, rhs);
  for (let iter = 0; rz > 0 && iter < maxIterations; iter++) {
    multiply(direction, out);
    const curvature = dot(direction, out);
    if (!(curvature > 0)) break;
    const step = rz / curvature;
    for (let i = 0; i < x.length; i++) {
      x[i] += step * direction[i];
      residual[i] -= step * out[i];
    }
    if (dot(residual, residual) <= stop) break;
    for (let i = 0; i < x.length; i++) z[i] = residual[i] / diagonal[i];
    const next = dot(residual, z), ratio = next / rz;
    for (let i = 0; i < x.length; i++) direction[i] = z[i] + ratio * direction[i];
    rz = next;
  }
  return x;
}

function curvatureStep(mesh, state, rhs) {
  const {weights} = state, {edges, components} = mesh;
  const damping = 1e-7;
  const diagonal = rhs.map(() => damping);
  for (let e = 0; e < edges.length; e++) {
    const [i, j] = edges[e];
    diagonal[i] += weights[e];
    diagonal[j] += weights[e];
  }
  const step = pcg(rhs, diagonal.map(d => Math.max(d, damping)), (x, out) => {
    for (let i = 0; i < x.length; i++) out[i] = damping * x[i];
    for (let e = 0; e < edges.length; e++) {
      const [i, j] = edges[e], v = weights[e] * (x[i] - x[j]);
      out[i] += v;
      out[j] -= v;
    }
  }, 100, 1e-4);
  // Curvatures are invariant under multiplying all radii in a component.
  for (const ids of components) {
    const mean = ids.reduce((s, i) => s + step[i], 0) / ids.length;
    for (const i of ids) step[i] -= mean;
  }
  return step;
}

// A perfectly planar start has zero out-of-plane derivatives of every edge length.
// Seed a shallow bowl/saddle only for planar components requesting interior curvature.
function seedPlanar(mesh, x, curvature) {
  for (const ids of mesh.components) {
    let positive = 0, negative = 0;
    for (const i of ids) if (!mesh.boundary[i]) {
      positive = Math.max(positive, curvature[i]);
      negative = Math.max(negative, -curvature[i]);
    }
    if (Math.max(positive, negative) < 1e-7) continue;
    const origin = 3 * ids[0];
    const offset = i => [x[3*i]-x[origin], x[3*i+1]-x[origin+1], x[3*i+2]-x[origin+2]];
    let first = [0,0,0], extent = 0;
    for (const i of ids) {
      const p = offset(i), distance = Math.hypot(...p);
      if (distance > extent) { first = p; extent = distance; }
    }
    first = first.map(v => v / extent);
    let normal = [0,0,0], width = 0;
    for (const i of ids) {
      const perp = cross(first, offset(i)), distance = Math.hypot(...perp);
      if (distance > width) { normal = perp; width = distance; }
    }
    normal = normal.map(v => v / width);
    if (ids.some(i => Math.abs(dot(offset(i), normal)) > 1e-9 * extent)) continue;
    const second = cross(normal, first), sign = positive >= negative ? 1 : -1;
    // Coordinates are component-centered; the added height has no preferred world axis.
    for (const i of ids) {
      const p = x.subarray(3*i, 3*i+3);
      const height = .12 * (dot(p, first)**2 + sign * dot(p, second)**2) / extent;
      for (let axis = 0; axis < 3; axis++) p[axis] += height * normal[axis];
    }
  }
}

function shapeResidual(x, edges, lengths, jacobian) {
  let error = 0;
  for (let e = 0; e < edges.length; e++) {
    const [i, j] = edges[e], s = 3*i, t = 3*j, l = lengths[e];
    const dx = x[s]-x[t], dy = x[s+1]-x[t+1], dz = x[s+2]-x[t+2];
    const d = Math.hypot(dx, dy, dz), residual = d/l - 1;
    error += residual * residual;
    if (!jacobian) continue;
    const {directions, rhs, diagonal} = jacobian;
    const inv = d > 0 ? 1/(d*l) : 0;
    directions[3*e] = dx * inv;
    directions[3*e+1] = dy * inv;
    directions[3*e+2] = dz * inv;
    for (let a = 0; a < 3; a++) {
      const gradient = directions[3*e+a];
      rhs[s+a] -= gradient * residual;
      rhs[t+a] += gradient * residual;
      diagonal[s+a] += gradient * gradient;
      diagonal[t+a] += gradient * gradient;
    }
  }
  return Math.sqrt(error / edges.length);
}

// Damped Gauss–Newton fit of coordinate edge lengths using sparse J and Jᵀ products.
export function reconstruct(mesh, state, initial, maxIterations, options = {}) {
  const {edges, components, scale} = mesh, lengths = state.lengths;
  const weight = options.bendingWeight ?? 0, target = options.bendingTarget ?? 'flat';
  if (!Number.isFinite(weight) || weight < 0) throw new Error('Bending weight must be finite and nonnegative.');
  if (!['flat', 'reference'].includes(target)) throw new Error('Select Flat or Input dihedral angles.');
  const bending = weight > 0 ? new HingeBending(mesh, lengths, weight, target) : null;
  // Measure the fit on the returned coordinates. Large world-space offsets can
  // round away displacements that are representable in the centered solve.
  const worldError = points => rms(lengths.map((l, e) => {
    const [i, j] = edges[e];
    return Math.hypot(...subtract(points[i], points[j])) / scale / l - 1;
  }));
  const initialError = worldError(initial);
  const combined = (edgeRms, x, jacobian) =>
    bending ? Math.hypot(edgeRms, Math.sqrt(bending.evaluate(x, jacobian)/edges.length)) : edgeRms;
  const initialObjective = combined(initialError, bending && initial.flat());
  if (!Number.isFinite(initialObjective)) throw new Error('The starting shape contains an undefined hinge.');
  if (initialObjective < 1e-9) return {vertices: initial.map(v => v.slice()), edgeError: initialError};
  let x = new Float64Array(3 * mesh.n);
  const centers = components.map(ids => {
    const anchor = initial[ids[0]], center = [0,0,0];
    for (const i of ids) for (let a = 0; a < 3; a++) center[a] += (initial[i][a] - anchor[a]) / ids.length;
    for (let a = 0; a < 3; a++) center[a] += anchor[a];
    for (const i of ids) for (let a = 0; a < 3; a++) x[3*i+a] = (initial[i][a] - center[a]) / scale;
    return center;
  });
  seedPlanar(mesh, x, state.curvature);
  let damping = 1e-3;
  const directions = new Float64Array(3 * edges.length);
  for (let iter = 0; iter < maxIterations; iter++) {
    const rhs = new Float64Array(x.length), diagonal = new Float64Array(x.length);
    const jacobian = {rhs, diagonal, directions};
    const error = combined(shapeResidual(x, edges, lengths, jacobian), x, jacobian);
    if (error <= 1e-7) break;
    const lambda = damping * sum(diagonal) / x.length;
    for (let i = 0; i < x.length; i++) diagonal[i] += lambda;
    const delta = pcg(rhs, diagonal, (v, out) => {
      for (let i = 0; i < v.length; i++) out[i] = lambda * v[i];
      for (let e = 0; e < edges.length; e++) {
        const [i,j] = edges[e], s = 3*i, t = 3*j, p = 3*e;
        const q = directions[p]*(v[s]-v[t]) + directions[p+1]*(v[s+1]-v[t+1])
                 + directions[p+2]*(v[s+2]-v[t+2]);
        for (let a = 0; a < 3; a++) {
          const contribution = q * directions[p+a];
          out[s+a] += contribution;
          out[t+a] -= contribution;
        }
      }
      if (bending) bending.addNormalProduct(v, out);
    }, 72, .005);
    const proposal = x.map((value, i) => value + delta[i]);
    const nextError = combined(shapeResidual(proposal, edges, lengths), proposal);
    if (nextError < error) {
      x = proposal;
      if (error - nextError < 1e-10 * Math.max(error, 1e-3)) break;
      damping = Math.max(1e-8, damping * .55);
    } else {
      damping *= 8;
      if (damping > 1e8) break;
    }
  }
  const vertices = new Array(mesh.n);
  components.forEach((ids, c) => {
    const drift = [0,0,0];
    for (const i of ids) for (let a = 0; a < 3; a++) drift[a] += x[3*i+a] / ids.length;
    for (const i of ids) vertices[i] = drift.map((d,a) => (x[3*i+a] - d) * scale + centers[c][a]);
  });
  const renderedError = worldError(vertices);
  if (!(combined(renderedError, bending && vertices.flat()) <= initialObjective)) {
    return {vertices: initial.map(v => v.slice()), edgeError: initialError};
  }
  return {vertices, edgeError: renderedError};
}

function integerOption(value, name, minimum) {
  if (!Number.isSafeInteger(value) || value < minimum)
    throw new Error(name + ' must be an integer ≥ ' + minimum + '.');
  return value;
}

// emit receives light progress records on accepted steps. Previews are observation-only;
// the final fit starts from the same input mesh regardless of the preview schedule.
export function solve(vertices, faces, options = {}, emit = () => {}) {
  const mesh = prepareMesh(vertices, faces);
  const maxIterations = integerOption(options.maxIterations ?? 80, 'Iteration limit', 1);
  const previewEvery = integerOption(options.previewEvery ?? 0, 'Preview interval', 0);
  const shapeIterations = integerOption(options.shapeIterations ?? 100, 'Shape iteration limit', 1);
  const tolerance = options.tolerance ?? 1e-5;
  if (!Number.isFinite(tolerance) || !(tolerance > 0)) throw new Error('Tolerance must be positive.');
  let u = new Float64Array(mesh.n), state = packingMetric(mesh, u);
  if (!state) throw new Error('The input metric contains a degenerate triangle.');
  const target = makeTarget(mesh, state.curvature, options);
  const residual = k => target.map((v, i) => v - k[i]);
  let rhs = residual(state.curvature), curvatureError = rms(rhs);
  const history = [curvatureError], initialCurvatureError = curvatureError;
  let iteration = 0, reason = 'iteration-limit', previewVertices = vertices;
  const report = () => ({iteration, curvatureError, initialCurvatureError});
  const arrays = () => ({curvature: Array.from(state.curvature), targetCurvature: Array.from(target)});
  emit({type:'progress', phase:'metric', ...report(), ...arrays()});
  while (iteration < maxIterations && curvatureError > tolerance) {
    const direction = curvatureStep(mesh, state, rhs);
    let accepted = false;
    // Damped Newton steps plus a Ricci-gradient fallback near difficult triangles.
    for (const step of [direction, rhs]) {
      for (let rate = .6; rate >= 1e-6; rate *= .5) {
        const trial = u.map((value, i) => value + rate * step[i]);
        const metric = packingMetric(mesh, trial);
        if (!metric) continue;
        const next = residual(metric.curvature), nextError = rms(next);
        if (nextError < curvatureError) {
          u = trial;
          state = metric;
          rhs = next;
          curvatureError = nextError;
          accepted = true;
          break;
        }
      }
      if (accepted) break;
    }
    if (!accepted) { reason = 'line-search'; break; }
    iteration++;
    history.push(curvatureError);
    emit({type:'progress', phase:'metric', ...report()});
    if (previewEvery > 0 && iteration % previewEvery === 0 && curvatureError > tolerance && iteration < maxIterations) {
      const shape = reconstruct(mesh, state, previewVertices, Math.min(shapeIterations, 24), options);
      previewVertices = shape.vertices;
      emit({type:'preview', phase:'shape', ...report(), ...arrays(), ...shape});
    }
  }
  if (curvatureError <= tolerance) reason = 'tolerance';
  emit({type:'progress', phase:'shape', ...report()});
  const shape = reconstruct(mesh, state, vertices, shapeIterations, options);
  return {type:'done', ...report(), ...arrays(), ...shape, history,
          converged: reason === 'tolerance', reason};
}
