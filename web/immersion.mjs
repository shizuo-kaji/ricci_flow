// Chern et al., Shape from Metric (2018), Sections 2.1–2.2.
// Inspect extrinsic vertex-star parity; this is a local obstruction test.
const tolerance = Math.sqrt(Number.EPSILON);

function orderedLink(faces) {
  const link = new Map();
  for (const [i, j] of faces) {
    if (!link.has(i)) link.set(i, []);
    if (!link.has(j)) link.set(j, []);
    const left = link.get(i), right = link.get(j);
    if (left.includes(j) || left.length === 2 || right.length === 2) return null;
    left.push(j);
    right.push(i);
  }
  if (link.size === 0) return null;
  const ends = [];
  for (const [i, neighbours] of link) if (neighbours.length === 1) ends.push(i);
  if (ends.length !== 0 && ends.length !== 2) return null;
  const start = ends[0] ?? link.keys().next().value;
  const ring = [start];
  let current = start, previous;
  for (;;) {
    const next = link.get(current).find(i => i !== previous);
    if (next === undefined || next === start) break;
    ring.push(next);
    previous = current;
    current = next;
  }
  if (ring.length !== link.size) return null;
  return {ring, closed: ends.length === 0};
}

const subtract = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a, b) => [a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]];

function unit(v, minimum = 0) {
  const length = Math.hypot(...v);
  return Number.isFinite(length) && length > minimum ? v.map(x => x / length) : null;
}

// Quaternion (x,y,z,w) of an orthonormal frame whose columns are t, b and n.
// Pick a large component to avoid division by zero at half-turns.
function frameQuaternion(t, b, n) {
  const trace = t[0] + b[1] + n[2];
  if (trace > 0) {
    const w = Math.sqrt(1 + trace) / 2, s = 4 * w;
    return [(b[2]-n[1])/s, (n[0]-t[2])/s, (t[1]-b[0])/s, w];
  }
  if (t[0] >= b[1] && t[0] >= n[2]) {
    const x = Math.sqrt(1 + t[0] - b[1] - n[2]) / 2, s = 4 * x;
    return [x, (b[0]+t[1])/s, (n[0]+t[2])/s, (b[2]-n[1])/s];
  }
  if (b[1] >= n[2]) {
    const y = Math.sqrt(1 - t[0] + b[1] - n[2]) / 2, s = 4 * y;
    return [(b[0]+t[1])/s, y, (n[1]+b[2])/s, (n[0]-t[2])/s];
  }
  const z = Math.sqrt(1 - t[0] - b[1] + n[2]) / 2, s = 4 * z;
  return [(n[0]+t[2])/s, (n[1]+b[2])/s, z, (t[1]-b[0])/s];
}

function pinchParity(center, ring) {
  const rays = ring.map(p => unit(subtract(p, center)));
  if (rays.some(ray => ray === null)) return null;
  const lifts = [];
  for (let i = 0; i < ring.length; i++) {
    const j = (i + 1) % ring.length;
    const tangent = unit(subtract(ring[j], ring[i]));
    const normal = unit(cross(rays[i], rays[j]), tolerance);
    if (!tangent || !normal) return null;
    lifts.push(frameQuaternion(tangent, cross(normal, tangent), normal));
  }
  // Lift every relative rotation with positive real part. Absolute frame
  // quaternions cancel in the closed product, leaving these signs alone.
  let sign = 1;
  for (let i = 0; i < lifts.length; i++) {
    const a = lifts[i], b = lifts[(i + 1) % lifts.length];
    const dot = a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3];
    if (Math.abs(dot) <= tolerance) return null;
    if (dot < 0) sign = -sign;
  }
  return sign === 1;
}

/** Pinch vertices are figure-8 closed stars. Boundary stars have open links.
 * Malformed links and geometrically ambiguous closed stars are indeterminate.
 * Passing parity alone does not certify local or global collision freedom.
 */
export function diagnoseVertexStars(vertices, faces) {
  if (!Array.isArray(vertices) || !vertices.every(p => Array.isArray(p) && p.length === 3 && p.every(Number.isFinite)))
    throw new Error('Vertex coordinates must be finite 3D points.');
  const stars = vertices.map(() => []);
  for (const f of faces) {
    if (f.length !== 3 || !f.every(i => Number.isInteger(i) && i >= 0 && i < stars.length) || new Set(f).size !== 3)
      throw new Error('Each face must contain three distinct valid vertex indices.');
    for (let i = 0; i < 3; i++) stars[f[i]].push([f[(i+1)%3], f[(i+2)%3]]);
  }
  const pinchVertices = [], indeterminateVertices = [], boundaryVertices = [];
  let classifiedInteriorVertices = 0;
  for (let v = 0; v < stars.length; v++) {
    const link = orderedLink(stars[v]);
    if (!link) {
      indeterminateVertices.push(v);
    } else if (!link.closed) {
      boundaryVertices.push(v);
    } else {
      const parity = pinchParity(vertices[v], link.ring.map(i => vertices[i]));
      if (parity === null) {
        indeterminateVertices.push(v);
      } else {
        classifiedInteriorVertices++;
        if (parity) pinchVertices.push(v);
      }
    }
  }
  return {classifiedInteriorVertices, pinchVertices, indeterminateVertices, boundaryVertices};
}
