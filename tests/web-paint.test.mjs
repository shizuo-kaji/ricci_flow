import test from 'node:test';
import assert from 'node:assert/strict';
import {angleDefects, topology, torus} from '../web/mesh.mjs';
import {CurvaturePaint} from '../web/paint.mjs';
import {solve} from '../web/solver.mjs';

const disk = {
  vertices: [[0,0,0], [1,0,0], [0,1,0], [-1,0,0], [0,-1,0]],
  faces: [[0,1,2], [0,2,3], [0,3,4], [0,4,1]]
};
const triangle = {vertices: [[0,0,0], [1,0,0], [0,.05,0]], faces: [[0,1,2]]};
const sum = values => values.reduce((total, x) => total + x, 0);
const near = (a, b, tolerance = 1e-12) => assert.ok(Math.abs(a - b) < tolerance, `${a} vs ${b}`);

function editor(mesh) {
  const graph = topology(mesh);
  const initial = angleDefects(mesh.vertices, mesh.faces, graph.boundary);
  return {graph, initial, paint: new CurvaturePaint(mesh, graph, initial)};
}

test('edge-ring brush crosses a closed torus seam and preserves painted defects', () => {
  const mesh = torus(12, 8);
  const {initial, paint, graph} = editor(mesh);
  const before = paint.field().values;
  paint.begin();
  assert.equal(paint.dab(0, 1, -.12), true);
  assert.equal(paint.dab(0, 1, -.12), false);
  paint.end();
  const selected = new Set([0, ...graph.adjacent[0]]);
  assert.deepEqual(new Set(paint.indices), selected);
  assert.ok(selected.has(7) && selected.has(88));
  const {values, error} = paint.field();
  assert.equal(error, null);
  assert.equal(paint.count, selected.size);
  for (const i of selected) assert.equal(values[i], -.12);
  near(sum(values), sum(initial));
  assert.deepEqual(before, initial);
  const result = solve(mesh.vertices, mesh.faces, {targetCurvature:values});
  assert.ok(result.converged && result.curvatureError < 1e-5);
});

test('a painted disk center and its auto-balanced boundary run through the shape solver', () => {
  const {paint} = editor(disk);
  paint.dab(0, 0, .6);
  const {values, error} = paint.field();
  assert.equal(error, null);
  assert.equal(values[0], .6);
  for (const value of values.slice(1)) near(value, (2*Math.PI - .6)/4);
  const result = solve(disk.vertices, disk.faces, {targetCurvature:values, strength:1, tolerance:1e-8});
  assert.ok(result.converged);
  near(result.curvature[0], .6, 1e-7);
  assert.ok(result.edgeError < 1e-6);
});

test('brush and balancing stay on each sheet of a coincident disconnected mesh', () => {
  const double = {
    vertices: [...disk.vertices, ...disk.vertices],
    faces: [...disk.faces, ...disk.faces.map(f => f.map(i => i + disk.vertices.length))]
  };
  const {paint, initial} = editor(double);
  paint.dab(1, 1, 1.2);
  assert.equal(paint.count, 4);
  let field = paint.field();
  assert.equal(field.error, null);
  assert.deepEqual(field.values.slice(5), initial.slice(5));
  near(sum(field.values.slice(0, 5)), 2*Math.PI);
  paint.dab(5, 0, .4);
  field = paint.field();
  assert.equal(field.error, null);
  near(sum(field.values.slice(0, 5)), 2*Math.PI);
  near(sum(field.values.slice(5)), 2*Math.PI);
  assert.equal(field.values[5], .4);
  const result = solve(double.vertices, double.faces, {targetCurvature:field.values});
  assert.ok(result.converged && result.curvatureError < 1e-5);
});

test('balancing respects a skinny corner even when an equal shift would exceed π', () => {
  const {paint, initial} = editor(triangle);
  paint.dab(0, 0, 1.1);
  const {error, values} = paint.field();
  assert.equal(error, null);
  assert.ok(initial[1] + (initial[0] - values[0])/2 > Math.PI);
  assert.ok(values[1] > initial[1] && values[1] < Math.PI);
  near(sum(values), 2*Math.PI);
});

test('invalid bounds, insufficient free curvature and incompatible full coverage are recoverable', () => {
  const {paint, initial} = editor(triangle);
  for (const limit of [0, Math.PI]) {
    paint.dab(0, 0, limit);
    assert.match(paint.field().error, /Vertex 0.*strictly between/);
    assert.equal(paint.field().values[0], limit);
  }
  paint.dab(0, 0, .1);
  paint.dab(1, 0, .1);
  assert.match(paint.field().error, /cannot absorb/);
  paint.dab(0, 1, 2);
  assert.match(paint.field().error, /All vertices.*sum/);
  paint.begin();
  initial.forEach((value, i) => paint.dab(i, 0, value));
  paint.end();
  assert.equal(paint.count, 3);
  assert.equal(paint.field().error, null);
  assert.deepEqual(paint.field().values, initial);
});

test('stroke undo, cancellation, erase and undoable clear preserve exact constraints', () => {
  const {paint, initial} = editor(disk);
  paint.begin();
  paint.dab(0, 0, .8);
  paint.dab(0, 0, .6);
  paint.dab(2, 0, 1.4);
  paint.end();
  const painted = paint.field().values;
  paint.begin();
  paint.dab(2, 0, null);
  paint.dab(3, 0, 1.8);
  assert.equal(paint.end(true), true);
  assert.deepEqual(paint.field().values, painted);
  paint.dab(0, 0, null);
  assert.deepEqual([...paint.indices], [2]);
  paint.undo();
  assert.deepEqual(paint.field().values, painted);
  paint.clear();
  assert.equal(paint.count, 0);
  assert.deepEqual(paint.field().values, initial);
  paint.undo();
  assert.deepEqual(paint.field().values, painted);
  paint.undo();
  assert.equal(paint.count, 0);
  assert.equal(paint.canUndo, false);
  paint.begin();
  paint.dab(1, 0, 0);
  paint.dab(1, 0, null);
  assert.equal(paint.end(), false);
  assert.equal(paint.canUndo, false);
});
