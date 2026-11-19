import {parsePly} from './ply.mjs';

// The default periodic mesh is generated locally; both seams share vertex IDs.
export function torus(segments = 32, rings = 14) {
  const vertices = [], faces = [];
  const index = (i, j) => (i % segments) * rings + (j % rings);
  for (let i = 0; i < segments; i++) {
    for (let j = 0; j < rings; j++) {
      const u = 2 * Math.PI * i / segments, v = 2 * Math.PI * j / rings;
      const r = 1 + 0.44 * Math.cos(v);
      vertices.push([r * Math.cos(u), r * Math.sin(u), 0.44 * Math.sin(v)]);
      const a = index(i, j), b = index(i + 1, j);
      const c = index(i + 1, j + 1), d = index(i, j + 1);
      faces.push([a, b, c], [a, c, d]);
    }
  }
  return {vertices, faces};
}

function parseObj(lines) {
  const vertices = [], faces = [];
  for (const line of lines) {
    const [kind, ...values] = line.split('#')[0].trim().split(/\s+/);
    if (kind === 'v') vertices.push(values.slice(0, 3).map(Number));
    if (kind === 'f') faces.push(values.map(value => {
      const id = Number(value.split('/')[0]);
      return id < 0 ? vertices.length + id : id - 1;
    }));
  }
  return {vertices, faces};
}

export function parseMesh(source, filename = '') {
  const isText = typeof source === 'string';
  const bytes = isText ? null : ArrayBuffer.isView(source)
    ? new Uint8Array(source.buffer, source.byteOffset, source.byteLength) : new Uint8Array(source);
  const decoder = new TextDecoder();
  const prefix = isText ? source.trimStart().slice(0, 16) : decoder.decode(bytes.subarray(0, 16));
  if (/\.ply$/i.test(filename) || /^\s*ply(?:\s|$)/.test(prefix)) {
    return parsePly(isText ? new TextEncoder().encode(source.trimStart()) : bytes, isText);
  }
  if (/\.obj$/i.test(filename)) return parseObj((isText ? source : decoder.decode(bytes)).split(/\r\n?|\n/));
  throw Error('Choose a triangle mesh in PLY or OBJ format.');
}

export function topology({vertices, faces}) {
  if (vertices.length < 3 || !faces.length) throw Error('The mesh must contain at least three vertices and one face.');
  if (vertices.some(v => v.length !== 3 || !v.every(Number.isFinite))) throw Error('Each vertex needs three finite coordinates.');
  const edges = new Map(), n = vertices.length, adjacent = Array.from({length: n}, () => []);
  for (const face of faces) {
    if (face.length !== 3) throw Error('Triangulate the faces before importing the mesh.');
    if (face.some(i => !Number.isInteger(i) || i < 0 || i >= n) || new Set(face).size !== 3) {
      throw Error('Each face must reference three distinct vertex IDs in the mesh.');
    }
    for (let k = 0; k < 3; k++) {
      const a = face[k], b = face[(k + 1) % 3], key = Math.min(a, b) * n + Math.max(a, b);
      const count = (edges.get(key) || 0) + 1;
      if (count > 2) throw Error('An edge is shared by more than two faces.');
      edges.set(key, count);
      if (count === 1) { adjacent[a].push(b); adjacent[b].push(a); }
    }
  }
  const boundary = new Uint8Array(n);
  for (const [key, count] of edges) {
    if (count === 1) { boundary[Math.floor(key / n)] = 1; boundary[key % n] = 1; }
  }
  if (adjacent.some(row => !row.length)) throw Error('The mesh contains a vertex that belongs to no face.');
  return {boundary, adjacent, edgeCount: edges.size, chi: n - edges.size + faces.length};
}

export function angleDefects(vertices, faces, boundary) {
  const values = Float64Array.from(boundary, on => (on ? 1 : 2) * Math.PI);
  for (const f of faces) {
    for (let j = 0; j < 3; j++) {
      const a = vertices[f[j]], b = vertices[f[(j + 1) % 3]], c = vertices[f[(j + 2) % 3]];
      const x = b[0] - a[0], y = b[1] - a[1], z = b[2] - a[2];
      const u = c[0] - a[0], v = c[1] - a[1], w = c[2] - a[2];
      const area = Math.hypot(y*w-z*v, z*u-x*w, x*v-y*u);
      if (!Number.isFinite(area) || area === 0) throw Error('A face has zero area or coordinates outside the numerical range.');
      values[f[j]] -= Math.atan2(area, x*u + y*v + z*w);
    }
  }
  return values;
}

export function parseTarget(source, initial, adjacent) {
  const rows = source.trim().split(/\r?\n/).map(s => s.split('#')[0].trim()).filter(Boolean);
  const target = Array(initial.length).fill(null);
  if (!rows.length) throw Error('The curvature CSV is empty.');
  rows.forEach((row, index) => {
    const cells = row.split(',').map(s => s.trim());
    const id = cells.length === 1 ? index : Number(cells[0]), val = Number(cells.at(-1));
    if (cells.length > 2 || cells.includes('') || !Number.isInteger(id) || id < 0 || id >= target.length ||
        !Number.isFinite(val) || target[id] !== null) throw Error('Use vertex ID and curvature columns, or one curvature per row in vertex order. Each ID must be unique and each curvature must be finite.');
    target[id] = val === 100 ? initial[id] : val;
  });
  // Fill unspecified vertices on each component using its own Gauss–Bonnet sum.
  const seen = new Uint8Array(target.length);
  for (let start = 0; start < target.length; start++) {
    if (seen[start]) continue;
    const stack = [start], missing = [];
    seen[start] = 1;
    let required = 0, magnitude = 0;
    while (stack.length) {
      const i = stack.pop();
      required += initial[i];
      if (target[i] === null || target[i] > 2 * Math.PI) missing.push(i);
      else { required -= target[i]; magnitude += Math.abs(target[i]); }
      for (const j of adjacent[i]) if (!seen[j]) { seen[j] = 1; stack.push(j); }
    }
    if (!missing.length && Math.abs(required) > 1e-7 * Math.max(1, magnitude)) {
      throw Error('Target curvatures must sum to 2πχ on each connected component. Omit vertices to distribute the remaining curvature automatically.');
    }
    for (const i of missing) target[i] = required / missing.length;
  }
  return target;
}

export function encodePly(vertices, faces) {
  return ['ply', 'format ascii 1.0', 'element vertex ' + vertices.length,
    'property double x', 'property double y', 'property double z',
    'element face ' + faces.length, 'property list uchar int vertex_indices', 'end_header',
    ...vertices.map(v => v.join(' ')), ...faces.map(f => '3 ' + f.join(' ')), ''].join('\n');
}
