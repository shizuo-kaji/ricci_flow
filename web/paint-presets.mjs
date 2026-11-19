export const PAINT_PRESETS = [
  {id: 'bullseye', label: 'Bullseye rings', description: 'Alternating high and low curvature in rings around a surface point.'},
  {id: 'dipole', label: 'Opposing poles', description: 'Raise curvature at one end of the surface and lower it at the other.'},
  {id: 'waves', label: 'Interference waves', description: 'Two sets of ripples overlap to form curved bands and islands.'}
];

// Dijkstra distances follow the edges, including around holes and across seams.
// The min-heap accepts duplicate entries; only the shortest entry is expanded.
function distancesFrom(source, vertices, adjacent, distances) {
  const queue = [[0, source]];
  distances[source] = 0;
  while (queue.length) {
    const [length, i] = queue[0], last = queue.pop();
    if (queue.length) {
      let pos = 0;
      while (2 * pos + 1 < queue.length) {
        let child = 2 * pos + 1;
        if (child + 1 < queue.length && queue[child + 1][0] < queue[child][0]) child++;
        if (last[0] <= queue[child][0]) break;
        queue[pos] = queue[child];
        pos = child;
      }
      queue[pos] = last;
    }
    if (length !== distances[i]) continue;
    const a = vertices[i];
    for (const j of adjacent[i]) {
      const b = vertices[j], next = length + Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
      if (next >= distances[j]) continue;
      distances[j] = next;
      let pos = queue.length;
      queue.push(null);
      while (pos > 0) {
        const parent = Math.floor((pos - 1) / 2);
        if (queue[parent][0] <= next) break;
        queue[pos] = queue[parent];
        pos = parent;
      }
      queue[pos] = [next, j];
    }
  }
}

// Break near ties by vertex ID so that rotating a symmetric mesh leaves its
// landmarks unchanged despite floating-point roundoff in transformed edges.
function landmark(ids, distance, radius, scale) {
  let best = ids[0], cost = Infinity;
  for (const i of ids) {
    const next = Math.round(Math.abs(distance[i] - radius) / scale * 1e10);
    if (next < cost || (next === cost && i < best)) { cost = next; best = i; }
  }
  return best;
}

// Express areas in units of squared intrinsic span for each component. This
// makes the pattern independent of model units and keeps area sums in range.
function normalizedAreas({vertices, faces}, spans) {
  const areas = new Float64Array(vertices.length);
  for (const [i, j, k] of faces) {
    const [ax, ay, az] = vertices[j].map((x, t) => (x - vertices[i][t]) / spans[i]);
    const [bx, by, bz] = vertices[k].map((x, t) => (x - vertices[i][t]) / spans[i]);
    const share = Math.hypot(ay*bz - az*by, az*bx - ax*bz, ax*by - ay*bx) / 6;
    areas[i] += share; areas[j] += share; areas[k] += share;
  }
  return areas;
}

function paintComponent(ids, scores, paint, targets, areas, dipole) {
  const low = ids.reduce((v, i) => Math.min(v, scores[i]), Infinity);
  const high = ids.reduce((v, i) => Math.max(v, scores[i]), -Infinity);
  const positive = [], negative = [];
  const totalArea = ids.reduce((sum, i) => sum + areas[i], 0);
  // Moderate the net transfer on long, thin surfaces. A low-frequency dipole
  // accumulates stretch over the entire tube; ripples cancel more locally.
  const aspect = Math.min(1, 1.35 * totalArea);
  const strength = dipole ? aspect * aspect : aspect;
  for (const i of ids) {
    const signed = 2 * (scores[i] - low) / (high - low) - 1;
    const weight = (Math.abs(signed) - .45) / .55;
    if (!(weight > 1e-9)) continue;
    // Angle defect is integrated curvature. Refined regions carry smaller
    // per-vertex changes for the same prescribed curvature density.
    const amplitude = Math.min(.28, 48 * strength * (areas[i] / totalArea));
    const room = signed > 0 ? paint.upper[i] - paint.initial[i] : paint.initial[i] - paint.lower[i];
    const amount = weight * Math.min(amplitude, .3 * room);
    if (amount > 0) (signed > 0 ? positive : negative).push({i, amount});
  }
  // Coarse, symmetric meshes can sample only at the peaks. Leave at least
  // one vertex free so subsequent manual strokes can still be balanced.
  if (positive.length + negative.length === ids.length) {
    const group = positive.length > negative.length ? positive : negative;
    const weakest = group.reduce((k, point, j) => point.amount < group[k].amount ? j : k, 0);
    group.splice(weakest, 1);
  }
  const sum = group => group.reduce((total, point) => total + point.amount, 0);
  const up = sum(positive), down = sum(negative), budget = Math.min(up, down);
  for (const {i, amount} of positive) targets.set(i, paint.initial[i] + amount * (budget / up));
  for (const {i, amount} of negative) targets.set(i, paint.initial[i] - amount * (budget / down));
}

export function applyPaintPreset(id, mesh, paint) {
  if (!PAINT_PRESETS.some(preset => preset.id === id)) throw Error('Choose a known paint preset.');
  const n = mesh.vertices.length, first = new Float64Array(n).fill(Infinity);
  const second = new Float64Array(n).fill(Infinity), scores = new Float64Array(n);
  const targets = new Map(), spans = new Float64Array(n);
  for (const ids of paint.components) {
    distancesFrom(ids[0], mesh.vertices, paint.adjacent, first);
    let span = ids.reduce((v, i) => Math.max(v, first[i]), 0);
    if (id !== 'bullseye') {
      const source = landmark(ids, first, id === 'dipole' ? span : .6 * span, span);
      distancesFrom(source, mesh.vertices, paint.adjacent, second);
      if (id === 'dipole') {
        // Approximate diameter endpoints by two farthest-point sweeps.
        span = ids.reduce((v, i) => Math.max(v, second[i]), 0);
        const opposite = landmark(ids, second, span, span);
        for (const i of ids) first[i] = Infinity;
        distancesFrom(opposite, mesh.vertices, paint.adjacent, first);
      }
    }
    for (const i of ids) {
      spans[i] = span;
      const radius = first[i] / span;
      if (id === 'dipole') scores[i] = (second[i] - first[i]) / span;
      else if (id === 'bullseye') scores[i] = Math.cos(3 * Math.PI * radius);
      else scores[i] = Math.cos(3 * Math.PI * radius) + .7 * Math.cos(3 * Math.PI * second[i] / span);
    }
  }
  const areas = normalizedAreas(mesh, spans);
  for (const ids of paint.components) paintComponent(ids, scores, paint, targets, areas, id === 'dipole');
  paint.end();
  paint.begin();
  try {
    for (const i of paint.indices) if (!targets.has(i)) paint.dab(i, 0, null);
    for (const [i, value] of targets) paint.dab(i, 0, value);
    const {error} = paint.field();
    if (error) throw Error(error);
    return paint.end();
  } catch (error) {
    paint.end(true);
    throw error;
  }
}
