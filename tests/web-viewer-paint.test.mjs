import assert from 'node:assert/strict';
import test from 'node:test';
import {MeshViewer, pickFrontVertex} from '../web/viewer.mjs';

test('pick the nearest corner of the foremost triangle at the hit position', () => {
  const points = [[0, 0, 1], [12, 0, 1], [0, 12, 1],
                  [0, 0, 4], [12, 0, -8], [0, 12, -8]];
  const faces = [[0, 1, 2], [5, 4, 3]];
  // The tilted triangle has the smaller mean depth but passes in front near the origin.
  assert.equal(pickFrontVertex(.5, .5, points, faces), 3);
  assert.equal(pickFrontVertex(.5, .5, points, [...faces].reverse()), 3);
  assert.equal(pickFrontVertex(10, 1, points, faces), 1);
  assert.equal(pickFrontVertex(10, 10, points, faces), null);
  assert.equal(pickFrontVertex(0, 12, points, faces), 2);
});

test('edge-on faces never hide a valid hit; shared edges pick a stable vertex', () => {
  const points = [[0, 0, 0], [6, 0, 0], [0, 6, 0], [6, 6, 0],
                  [3, 0, 4], [3, 6, 4], [3, 3, 4]];
  const faces = [[4, 5, 6], [0, 1, 2], [1, 3, 2]];
  assert.equal(pickFrontVertex(3, 3, points, faces), 0);
  assert.equal(pickFrontVertex(3, 3, points, [...faces].reverse()), 0);
  assert.equal(pickFrontVertex(3, 3, points, [faces[0]]), null);
});

globalThis.requestAnimationFrame = () => 1;
globalThis.ResizeObserver = class { observe() {} };
globalThis.window = {devicePixelRatio: 2};

function scene() {
  const listeners = new Map(), arcs = [];
  const context = new Proxy({arc: (...args) => arcs.push(args)}, {
    get: (target, key) => target[key] ?? (() => {})
  });
  const rect = {left: 53, top: 91, width: 720, height: 480};
  const canvas = {
    style: {}, clientWidth: 360, clientHeight: 240,
    getContext: () => context, getBoundingClientRect: () => rect, setPointerCapture() {},
    addEventListener(event, fn) { listeners.set(event, fn); }
  };
  const viewer = new MeshViewer(canvas);
  const vertices = [-1, 1].flatMap(x => [[x, -1, -1], [x, 1, -1], [x, 0, 1]]);
  const faces = [[0, 1, 2], [3, 4, 5]];
  viewer.setMesh(vertices, faces, new Float64Array(6), true);
  viewer.rotate(.96, -.60); // view along x; both layers have the same projected outline
  function location(id) {
    const {points} = viewer.projection(), p = points[id];
    const cx = (points[0][0] + points[1][0] + points[2][0]) / 3;
    const cy = (points[0][1] + points[1][1] + points[2][1]) / 3;
    // Sample a little inside the edge, including CSS scaling and page offsets.
    return [rect.left + 2 * (.95 * p[0] + .05 * cx), rect.top + 2 * (.95 * p[1] + .05 * cy)];
  }
  function send(type, [clientX, clientY] = [0, 0], options = {}) {
    const e = {button: 0, pointerId: 1, pointerType: 'mouse', clientX, clientY,
      preventDefault() { this.defaultPrevented = true; }, ...options};
    listeners.get(type)(e);
    return e;
  }
  const events = [], hover = [];
  viewer.setPaintTool({
    begin: () => events.push('start'), dab: id => events.push(id),
    end: cancelled => events.push(cancelled ? 'cancel' : 'commit'), hover: id => hover.push(id)
  });
  return {viewer, canvas, arcs, events, hover, send, location};
}

test('brush stroke samples front faces in scaled client coordinates and commits once', () => {
  const {viewer, events, hover, send, location} = scene();
  const left = location(3), right = location(4);
  const azimuth = viewer.azimuth;
  send('pointermove', left);
  assert.deepEqual(hover, [3]);
  send('pointerdown', left);
  send('pointermove', left);
  send('pointermove', right);
  send('pointerup', right);
  send('lostpointercapture', right);
  assert.deepEqual(events, ['start', 3, 4, 'commit']);
  assert.deepEqual(hover, [3, 4]);
  assert.equal(viewer.azimuth, azimuth);
  send('pointerleave', right);
  send('pointerdown', [0, 0]);
  send('pointerup', [0, 0]);
  assert.deepEqual(events, ['start', 3, 4, 'commit']);
  assert.equal(hover.at(-1), null);
});

test('a second touch cancels the stroke and changes the gesture to pinch until release', () => {
  const {viewer, events, send, location} = scene();
  const [x, y] = location(3);
  const finger = {pointerType: 'touch'}, other = {...finger, pointerId: 2};
  send('pointerdown', [x, y], finger);
  send('pointerdown', [x + 20, y], other);
  send('pointermove', [x + 35, y], other);
  assert.equal(viewer.zoom, 1.75);
  send('pointerup', [x + 35, y], other);
  send('pointermove', [x + 4, y], finger);
  send('pointerup', [x + 4, y], finger);
  assert.deepEqual(events, ['start', 3, 'cancel']);
  send('pointerdown', location(4), finger);
  send('pointerup', location(4), finger);
  assert.deepEqual(events.slice(3), ['start', 4, 'commit']);
});

test('fast drags sample intermediate vertices even when only the release event arrives', () => {
  const {viewer, events, send, location} = scene();
  const vertices = [], faces = [];
  for (let i = 0; i <= 4; i++) {
    vertices.push([0, i, -1], [0, i, 1]);
    const n = 2 * i;
    if (i) faces.push([n - 2, n, n - 1], [n - 1, n, n + 1]);
  }
  viewer.setMesh(vertices, faces, new Float64Array(vertices.length), true);
  viewer.rotate(.96, -.60);
  send('pointerdown', location(0));
  send('pointerup', location(8));
  assert.deepEqual(events, ['start', 0, 2, 4, 6, 8, 'commit']);
});

test('pointer cancellation, capture loss, camera and tool changes roll back an open stroke', () => {
  for (const action of ['pointercancel', 'lostpointercapture', 'tool change', 'camera change']) {
    const {viewer, events, send, location} = scene();
    const p = location(3);
    send('pointerdown', p);
    if (action === 'tool change') viewer.setPaintTool(null);
    else if (action === 'camera change') viewer.reset();
    else send(action, p);
    send('pointerup', p);
    assert.deepEqual(events, ['start', 3, 'cancel'], action);
  }
});

test('rotation modifiers, orbit mode and keyboard/wheel retain their camera controls', () => {
  for (const options of [{button: 2}, {shiftKey: true}, {orbit: true}]) {
    const {viewer, events, send, location} = scene();
    if (options.orbit) viewer.setPaintTool(null);
    const [x, y] = location(3);
    send('pointerdown', [x, y], options);
    send('pointermove', [x + 12, y], options);
    send('pointerup', [x + 12, y], options);
    assert.equal(viewer.azimuth, -12 * .009);
    assert.ok(send('keydown', undefined, {key: 'ArrowRight'}).defaultPrevented);
    assert.equal(viewer.azimuth, -12 * .009 + .1);
    assert.ok(send('wheel', [x, y], {deltaY: -100}).defaultPrevented);
    assert.ok(viewer.zoom > 1);
    assert.deepEqual(events, []);
  }
});

test('constraint dots are occluded by the front surface and update on orbit and clear', () => {
  const {viewer, arcs, location} = scene();
  viewer.setPaintedVertices(new Set([0, 3, 4]));
  viewer.draw();
  assert.equal(arcs.length, 2);
  const initialProjection = viewer.projection();
  viewer.setValues([0, 0, 2, 1, 0, -3]);
  assert.equal(viewer.limit, 3);
  assert.equal(viewer.projection(), initialProjection);
  viewer.rotate(Math.PI, 0);
  assert.equal(viewer.pickAt(...location(0)), 0);
  arcs.length = 0;
  viewer.draw();
  assert.equal(arcs.length, 1);
  viewer.setPaintedVertices(null);
  arcs.length = 0;
  viewer.draw();
  assert.equal(arcs.length, 0);
});
