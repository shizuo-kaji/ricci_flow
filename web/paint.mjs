function connectedComponents(adjacent) {
  const components = [], seen = new Uint8Array(adjacent.length);
  for (let vertex = 0; vertex < adjacent.length; vertex++) {
    if (seen[vertex]) continue;
    const ids = [vertex];
    seen[vertex] = 1;
    for (let head = 0; head < ids.length; head++) {
      for (const neighbor of adjacent[ids[head]]) {
        if (!seen[neighbor]) {
          seen[neighbor] = 1;
          ids.push(neighbor);
        }
      }
    }
    components.push(ids);
  }
  return components;
}

// Painted values are absolute angle defects. Store the explicit constraints
// separately so that erasing a vertex also releases it for curvature balancing.
export class CurvaturePaint {
  constructor(mesh, topology, initial) {
    this.initial = Float64Array.from(initial);
    this.adjacent = topology.adjacent;
    this.components = connectedComponents(this.adjacent);
    const faceCounts = new Uint32Array(initial.length);
    for (const face of mesh.faces) for (const vertex of face) faceCounts[vertex]++;
    this.upper = Float64Array.from(topology.boundary, b => (b ? 1 : 2) * Math.PI);
    this.lower = this.upper.map((hi, i) => hi - Math.PI * faceCounts[i]);
    this.painted = new Map();
    this.history = [];
    this.stroke = null;
    this.cachedField = null;
  }

  get count() { return this.painted.size; }
  get canUndo() { return this.history.length > 0; }
  get indices() { return this.painted.keys(); }

  begin() {
    if (!this.stroke) this.stroke = new Map();
  }

  dab(vertex, radius, value) {
    if (!Number.isInteger(vertex) || vertex < 0 || vertex >= this.initial.length ||
        !Number.isInteger(radius) || radius < 0 || (value !== null && !Number.isFinite(value))) return false;
    const singleDab = this.stroke === null;
    this.begin();
    const vertices = [vertex], seen = new Set(vertices);
    let first = 0;
    for (let ring = 0; ring < radius && first < vertices.length; ring++) {
      const end = vertices.length;
      for (; first < end; first++) {
        for (const neighbor of this.adjacent[vertices[first]]) {
          if (!seen.has(neighbor)) {
            seen.add(neighbor);
            vertices.push(neighbor);
          }
        }
      }
    }
    let changed = false;
    for (const i of vertices) {
      const previous = this.painted.get(i) ?? null;
      if (previous === value) continue;
      if (!this.stroke.has(i)) this.stroke.set(i, previous);
      if (value === null) this.painted.delete(i);
      else this.painted.set(i, value);
      changed = true;
    }
    if (changed) this.cachedField = null;
    if (singleDab) this.end();
    return changed;
  }

  restore(previous) {
    for (const [i, value] of previous) {
      if (value === null) this.painted.delete(i);
      else this.painted.set(i, value);
    }
    this.cachedField = null;
  }

  end(cancelled = false) {
    const changes = this.stroke;
    this.stroke = null;
    if (!changes) return false;
    for (const [i, previous] of changes) {
      if ((this.painted.get(i) ?? null) === previous) changes.delete(i);
    }
    if (!changes.size) return false;
    if (cancelled) this.restore(changes);
    else this.history.push(changes);
    return true;
  }

  undo() {
    if (this.end(true)) return true;
    const previous = this.history.pop();
    if (!previous) return false;
    this.restore(previous);
    return true;
  }

  clear() {
    this.end();
    if (!this.count) return false;
    this.history.push(new Map(this.painted));
    this.painted.clear();
    this.cachedField = null;
    return true;
  }

  field() {
    if (this.cachedField) return this.cachedField;
    const values = this.initial.slice();
    for (const [i, value] of this.painted) values[i] = value;
    let error = null;
    if (this.count) {
      for (const [index, ids] of this.components.entries()) {
        const issue = this.balanceComponent(values, ids, index + 1);
        error ??= issue;
      }
    }
    return this.cachedField = {values, error};
  }

  balanceComponent(values, ids, component) {
    let delta = 0, magnitude = 0, freeCount = 0;
    for (const i of ids) {
      if (!this.painted.has(i)) { freeCount++; continue; }
      if (values[i] <= this.lower[i] || values[i] >= this.upper[i]) {
        return `Vertex ${i} needs a target strictly between ${this.lower[i].toFixed(4)} and ${this.upper[i].toFixed(4)} rad. Repaint or erase it.`;
      }
      delta += this.initial[i] - values[i];
      magnitude += Math.abs(this.initial[i]) + Math.abs(values[i]);
    }
    if (!delta) return null;
    if (!freeCount) {
      const roundoff = 64 * Number.EPSILON * Math.max(1, magnitude);
      if (Math.abs(delta) <= roundoff) return null;
      return `All vertices on component ${component} are painted and their sum misses 2πχ. Erase a vertex or change its value.`;
    }
    // Move the same fraction of every free vertex's available range. The
    // changes sum to delta and preserve the strict local angle bounds.
    const bounds = delta > 0 ? this.upper : this.lower;
    let capacity = 0;
    for (const i of ids) if (!this.painted.has(i)) capacity += bounds[i] - this.initial[i];
    const fraction = delta / capacity;
    const infeasible = `The unpainted vertices on component ${component} cannot absorb this curvature. Erase more vertices or change the painted values.`;
    if (!(fraction >= 0 && fraction < 1)) return infeasible;
    for (const i of ids) {
      if (this.painted.has(i)) continue;
      values[i] += fraction * (bounds[i] - this.initial[i]);
      // A request within machine precision of a strict bound may round to it.
      if (!(values[i] > this.lower[i] && values[i] < this.upper[i])) return infeasible;
    }
    return null;
  }
}
