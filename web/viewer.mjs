const stops = [
  [95, 100, 209], [72, 151, 224], [73, 195, 196], [143, 214, 167], [238, 212, 137], [251, 154, 109]
];

function pigment(t) {
  const x = Math.min(stops.length - 1, Math.max(0, t * (stops.length - 1)));
  const a = Math.floor(x), b = Math.min(stops.length - 1, a + 1);
  return stops[a].map((v, j) => v + (x - a) * (stops[b][j] - v));
}

function bounds(vertices) {
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (const v of vertices) for (let i = 0; i < 3; i++) {
    lo[i] = Math.min(lo[i], v[i]); hi[i] = Math.max(hi[i], v[i]);
  }
  const center = hi.map((v, i) => (v + lo[i]) / 2);
  let radius = 0, verticalRadius = 0;
  const cy = Math.cos(-.96), sy = Math.sin(-.96), sv = Math.sin(.60), cv = Math.cos(.60);
  for (const v of vertices) {
    const [x, y, z] = v.map((value, i) => value - center[i]);
    radius = Math.max(radius, Math.hypot(x, y, z));
    verticalRadius = Math.max(verticalRadius, Math.abs(-sv*(cy*x + sy*y) + cv*z));
  }
  return {center, radius: radius || 1, verticalRadius: Math.max(verticalRadius, .55*radius) || 1, floor: lo[2]};
}

const pixelTolerance = 1e-7;

// Z increases toward the camera. Compare depths at the pointer itself, even for
// overlapping triangles whose centroids would give a different drawing order.
export function pickFrontVertex(x, y, points, faces) {
  let depth = -Infinity, nearest = null, nearestDistance = Infinity;
  for (const face of faces) {
    const a = points[face[0]], b = points[face[1]], c = points[face[2]];
    if (x < Math.min(a[0], b[0], c[0]) - pixelTolerance || x > Math.max(a[0], b[0], c[0]) + pixelTolerance ||
        y < Math.min(a[1], b[1], c[1]) - pixelTolerance || y > Math.max(a[1], b[1], c[1]) + pixelTolerance) continue;
    const bx = b[0] - a[0], by = b[1] - a[1], cx = c[0] - a[0], cy = c[1] - a[1];
    const px = x - a[0], py = y - a[1], det = bx * cy - by * cx;
    if (Math.abs(det) <= 8 * Number.EPSILON * (Math.abs(bx * cy) + Math.abs(by * cx))) continue;
    const u = (px * cy - py * cx) / det, v = (bx * py - by * px) / det;
    if (u < -1e-10 || v < -1e-10 || u + v > 1 + 1e-10) continue;
    const z = a[2] + u * (b[2] - a[2]) + v * (c[2] - a[2]);
    if (z < depth) continue;
    if (z > depth) nearestDistance = Infinity;
    depth = z;
    for (const id of face) {
      const p = points[id], distance = (p[0] - x) ** 2 + (p[1] - y) ** 2;
      if (distance < nearestDistance || (distance === nearestDistance && id < nearest)) {
        nearest = id;
        nearestDistance = distance;
      }
    }
  }
  return nearest;
}

// Bucket projected triangles once per camera position so pointer samples and
// visibility checks for painted dots only inspect nearby faces.
function createPicker(points, faces, w, h) {
  const cellSize = Math.max(32, w / 48, h / 48);
  const cols = Math.max(1, Math.ceil(w / cellSize)), rows = Math.max(1, Math.ceil(h / cellSize));
  const cells = Array.from({length: cols * rows}, () => []);
  const cell = value => Math.floor(value / cellSize);
  for (const face of faces) {
    const [a, b, c] = face.map(id => points[id]);
    const x0 = Math.max(0, cell(Math.min(a[0], b[0], c[0]) - pixelTolerance));
    const x1 = Math.min(cols - 1, cell(Math.max(a[0], b[0], c[0]) + pixelTolerance));
    const y0 = Math.max(0, cell(Math.min(a[1], b[1], c[1]) - pixelTolerance));
    const y1 = Math.min(rows - 1, cell(Math.max(a[1], b[1], c[1]) + pixelTolerance));
    for (let row = y0; row <= y1; row++) {
      for (let col = x0; col <= x1; col++) cells[row * cols + col].push(face);
    }
  }
  return (x, y) => {
    if (x < 0 || y < 0 || x >= w || y >= h) return null;
    return pickFrontVertex(x, y, points, cells[cell(y) * cols + cell(x)]);
  };
}

export class MeshViewer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.wire = true;
    this.mode = 'curvature';
    this.paintTool = null;
    this.paintedVertices = [];
    this.hoverVertex = null;
    this.stroke = null;
    this.gesture = 'idle';
    this.reset();
    const pointers = new Map();
    let distance = 0;
    const pinch = () => {
      const [a, b] = pointers.values();
      return Math.hypot(a[0] - b[0], a[1] - b[1]);
    };
    canvas.addEventListener('pointerdown', e => {
      if (e.button !== 0 && e.button !== 2) return;
      canvas.setPointerCapture(e.pointerId);
      pointers.set(e.pointerId, [e.clientX, e.clientY]);
      if (pointers.size > 1) {
        this.gesture = 'pinch';
        this.endStroke(true);
        this.setHover(null);
        distance = pinch();
      } else if (this.paintTool && e.button === 0 && !e.shiftKey) {
        this.gesture = 'paint';
        this.paintSegment(e.clientX, e.clientY, e.clientX, e.clientY);
      } else {
        this.gesture = 'rotate';
        canvas.style.cursor = 'grabbing';
        this.setHover(null);
      }
    });
    canvas.addEventListener('pointermove', e => {
      const previous = pointers.get(e.pointerId);
      if (!previous) {
        if (this.paintTool && !pointers.size) this.setHover(this.pickAt(e.clientX, e.clientY));
        return;
      }
      pointers.set(e.pointerId, [e.clientX, e.clientY]);
      if (this.gesture === 'pinch' && pointers.size === 2) {
        const next = pinch();
        if (next > 0 && distance > 0) this.zoomBy(next / distance);
        distance = next;
      } else if (this.gesture === 'paint') {
        this.paintSegment(...previous, e.clientX, e.clientY);
      } else if (this.gesture === 'rotate') {
        this.rotate((previous[0] - e.clientX) * .009, (e.clientY - previous[1]) * .009);
      }
    });
    for (const event of ['pointerup', 'pointercancel', 'lostpointercapture']) {
      canvas.addEventListener(event, e => {
        const previous = pointers.get(e.pointerId);
        if (!previous) return;
        pointers.delete(e.pointerId);
        if (this.gesture === 'paint') {
          if (event === 'pointerup') this.paintSegment(...previous, e.clientX, e.clientY);
          this.endStroke(event !== 'pointerup');
        }
        if (event !== 'pointerup' || e.pointerType === 'touch') this.setHover(null);
        if (pointers.size === 0) {
          this.gesture = 'idle';
          canvas.style.cursor = this.paintTool ? 'crosshair' : '';
        }
        else if (pointers.size === 2) distance = pinch();
      });
    }
    canvas.addEventListener('pointerleave', () => this.setHover(null));
    canvas.addEventListener('contextmenu', e => e.preventDefault());
    canvas.addEventListener('wheel', e => {
      e.preventDefault();
      this.zoomBy(Math.exp(-Math.max(-150, Math.min(150, e.deltaY)) * .002));
    }, {passive: false});
    canvas.addEventListener('keydown', e => {
      const rotations = {ArrowLeft: [-.1, 0], ArrowRight: [.1, 0], ArrowUp: [0, .1], ArrowDown: [0, -.1]};
      if (e.key in rotations) this.rotate(...rotations[e.key]);
      else if (e.key === '+' || e.key === '=') this.zoomBy(1.15);
      else if (e.key === '-') this.zoomBy(1 / 1.15);
      else if (e.key === 'Home') this.reset();
      else return;
      e.preventDefault();
    });
    new ResizeObserver(() => this.redraw()).observe(canvas);
  }

  invalidateProjection() {
    if (this.gesture === 'paint') {
      this.gesture = 'idle';
      this.endStroke(true);
    }
    this.geometry = null;
    this.setHover(null);
  }

  reset() {
    this.invalidateProjection();
    this.azimuth = -.96;
    this.elevation = .60;
    this.zoom = 1;
    if (this.vertices) this.frame = bounds(this.vertices);
    this.redraw();
  }

  rotate(horizontal, vertical) {
    this.invalidateProjection();
    this.azimuth += horizontal;
    this.elevation = Math.max(-1.53, Math.min(1.53, this.elevation + vertical));
    this.redraw();
  }

  zoomBy(factor) {
    this.invalidateProjection();
    this.zoom = Math.min(6, Math.max(.22, this.zoom * factor));
    this.redraw();
  }

  setMesh(vertices, faces, values, fit = false) {
    this.invalidateProjection();
    if (this.faces !== faces) this.paintedVertices = [];
    this.vertices = vertices;
    this.faces = faces;
    if (!fit && !this.frame) this.frame = bounds(vertices);
    this.setValues(values);
    if (fit) this.reset();
  }

  setValues(values) {
    this.values = values;
    let limit = 0;
    for (const value of values) limit = Math.max(limit, Math.abs(value));
    this.limit = limit < 1e-12 ? 0 : limit;
    this.faceColors = null;
    this.redraw();
  }

  setPaintTool(tool) {
    if (tool === this.paintTool) return;
    if (this.gesture === 'paint') this.gesture = 'idle';
    this.endStroke(true);
    this.setHover(null);
    this.paintTool = tool;
    this.canvas.style.cursor = tool ? 'crosshair' : '';
  }

  setPaintedVertices(ids) {
    this.paintedVertices = ids ? Array.from(ids) : [];
    this.redraw();
  }

  setHover(id) {
    if (id === this.hoverVertex) return;
    this.hoverVertex = id;
    this.paintTool?.hover?.(id);
    this.redraw();
  }

  endStroke(cancelled) {
    const stroke = this.stroke;
    this.stroke = null;
    if (stroke) stroke.tool.end(cancelled);
  }

  pickAt(clientX, clientY, rect = this.canvas.getBoundingClientRect()) {
    if (!this.vertices || !rect.width || !rect.height) return null;
    const x = (clientX - rect.left) * this.canvas.clientWidth / rect.width;
    const y = (clientY - rect.top) * this.canvas.clientHeight / rect.height;
    return this.pick(x, y);
  }

  pick(x, y) {
    const geometry = this.projection();
    geometry.pick ??= createPicker(geometry.points, this.faces, geometry.w, geometry.h);
    return geometry.pick(x, y);
  }

  paintSegment(x0, y0, x1, y1) {
    const rect = this.canvas.getBoundingClientRect();
    const samples = Math.max(1, Math.min(96, Math.ceil(Math.hypot(x1 - x0, y1 - y0) / 5)));
    let id = null;
    for (let i = 1; i <= samples && this.paintTool && this.gesture === 'paint'; i++) {
      const t = i / samples;
      id = this.pickAt(x0 + t * (x1 - x0), y0 + t * (y1 - y0), rect);
      if (id === null) continue;
      if (!this.stroke) {
        this.stroke = {tool: this.paintTool, last: null};
        this.paintTool.begin();
      }
      if (this.stroke && this.stroke.last !== id) {
        this.stroke.last = id;
        this.stroke.tool.dab(id);
      }
    }
    this.setHover(this.paintTool ? id : null);
  }

  redraw() {
    if (this.pending) return;
    this.pending = requestAnimationFrame(() => { this.pending = 0; this.draw(); });
  }

  projection() {
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    if (this.geometry && this.geometry.w === w && this.geometry.h === h) return this.geometry;
    const sy = Math.sin(this.azimuth), cy = Math.cos(this.azimuth);
    const sv = Math.sin(this.elevation), cv = Math.cos(this.elevation);
    const {center, radius, verticalRadius} = this.frame;
    const scale = Math.min(w*.38/radius, h*.40/verticalRadius) * this.zoom;
    const project = (x, y, z) => {
      x -= center[0]; y -= center[1]; z -= center[2];
      const radial = cy*x + sy*y;
      return [w*.5 + scale*(-sy*x + cy*y), h*.48 - scale*(-sv*radial + cv*z), cv*radial + sv*z];
    };
    const points = this.vertices.map(p => project(...p));
    const order = this.faces.map((f, i) => [i, points[f[0]][2] + points[f[1]][2] + points[f[2]][2]]);
    order.sort((a, b) => a[1] - b[1]);
    this.geometry = {w, h, project, points, order, scale};
    return this.geometry;
  }

  draw() {
    const c = this.ctx, el = this.canvas;
    if (!c || !this.vertices) return;
    const w = el.clientWidth, h = el.clientHeight, dpr = Math.min(2, window.devicePixelRatio || 1);
    if (!w || !h) return;
    if (el.width !== Math.round(w*dpr) || el.height !== Math.round(h*dpr)) {
      el.width = Math.round(w*dpr); el.height = Math.round(h*dpr);
    }
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, w, h);
    const {center, radius, floor} = this.frame;
    const geometry = this.projection();
    const {project, points, order, scale} = geometry;
    const path = (a, b) => { c.moveTo(a[0], a[1]); c.lineTo(b[0], b[1]); };
    const floorZ = floor - .05*radius;
    const step = radius / 4;
    c.beginPath();
    for (let i = -22; i <= 22; i++) {
      const d = i*step, span = 22*step;
      path(project(center[0]+d, center[1]-span, floorZ), project(center[0]+d, center[1]+span, floorZ));
      path(project(center[0]-span, center[1]+d, floorZ), project(center[0]+span, center[1]+d, floorZ));
    }
    c.strokeStyle = 'rgba(131,174,185,.115)'; c.lineWidth = .7; c.stroke();
    if (!this.faceColors && this.mode === 'curvature') {
      this.faceColors = this.faces.map(([a, b, d]) => {
        const mean = (this.values[a] + this.values[b] + this.values[d]) / 3;
        return pigment(this.limit ? .5 + .5 * mean / this.limit : .5);
      });
    }
    c.lineJoin = 'round';
    for (const [i] of order) {
      const f = this.faces[i];
      const [a, b, d] = f.map(id => points[id]);
      const ux = (b[0]-a[0])/scale, uy = (a[1]-b[1])/scale, uz = b[2]-a[2];
      const vx = (d[0]-a[0])/scale, vy = (a[1]-d[1])/scale, vz = d[2]-a[2];
      const nx = uy*vz-uz*vy, ny = uz*vx-ux*vz, nz = ux*vy-uy*vx;
      const sign = nz >= 0 ? 1 : -1;
      const light = .70 + .34*Math.max(0, sign*(-.44*nx+.72*ny+.53*nz) / (Math.hypot(nx,ny,nz) || 1));
      const rgb = this.mode === 'curvature' ? this.faceColors[i] : [118, 210, 193];
      c.fillStyle = 'rgb(' + rgb.map(x => Math.round(x*light)).join(',') + ')';
      c.beginPath(); c.moveTo(a[0], a[1]); c.lineTo(b[0], b[1]); c.lineTo(d[0], d[1]); c.closePath();
      c.fill(); c.lineWidth = this.wire ? .70 : .55;
      c.strokeStyle = this.wire ? 'rgba(13,44,55,.53)' : c.fillStyle; c.stroke();
    }
    if (this.paintedVertices.length) {
      geometry.visible ??= new Map();
      c.beginPath();
      for (const id of this.paintedVertices) {
        const p = points[id];
        if (!p) continue;
        if (!geometry.visible.has(id)) geometry.visible.set(id, this.pick(p[0], p[1]) === id);
        if (!geometry.visible.get(id)) continue;
        c.moveTo(p[0] + 2, p[1]); c.arc(p[0], p[1], 2, 0, 2 * Math.PI);
      }
      c.fillStyle = '#f7fffc'; c.strokeStyle = '#163d43'; c.lineWidth = 1.1;
      c.fill(); c.stroke();
    }
    if (this.hoverVertex !== null) {
      const p = points[this.hoverVertex];
      c.beginPath(); c.arc(p[0], p[1], 5, 0, 2 * Math.PI);
      c.strokeStyle = '#17343d'; c.lineWidth = 3.5; c.stroke();
      c.strokeStyle = '#fff'; c.lineWidth = 1.5; c.stroke();
    }
    const ox = w-42, oy = h-34;
    const origin = project(...center), axis = 20 / scale;
    c.lineWidth = 1.4; c.font = '10px ui-monospace, monospace';
    for (const [j, label, color] of [[0,'X','#f3ad94'], [1,'Y','#78daba'], [2,'Z','#89b9f0']]) {
      const p = center.slice(); p[j] += axis;
      const end = project(...p), dx = end[0]-origin[0], dy = end[1]-origin[1];
      c.strokeStyle = color; c.fillStyle = color; c.beginPath(); path([ox, oy], [ox+dx, oy+dy]); c.stroke();
      c.fillText(label, ox+1.24*dx-3, oy+1.24*dy+3);
    }
  }
}
