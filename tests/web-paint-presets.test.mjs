import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {angleDefects, parseMesh, topology, torus} from '../web/mesh.mjs';
import {PAINT_PRESETS, applyPaintPreset} from '../web/paint-presets.mjs';
import {CurvaturePaint} from '../web/paint.mjs';
import {SHAPE_PRESETS} from '../web/shapes.mjs';
import {solve} from '../web/solver.mjs';

function editor(mesh) {
  const graph = topology(mesh);
  return new CurvaturePaint(mesh, graph, angleDefects(mesh.vertices, mesh.faces, graph.boundary));
}

function checkTargets(paint) {
  const {values, error} = paint.field();
  assert.equal(error, null);
  for (const ids of paint.components) {
    const deltas = ids.map(i => values[i] - paint.initial[i]);
    assert.ok(Math.abs(deltas.reduce((sum, delta) => sum + delta, 0)) < 1e-10);
    assert.ok(deltas.some(d => d > .001) && deltas.some(d => d < -.001));
    assert.ok(ids.every(i => values[i] > paint.lower[i] && values[i] < paint.upper[i]));
    assert.ok(ids.some(i => !paint.painted.has(i)));
  }
}

test('each preset builds a distinct balanced target and converges on a torus and a disk', () => {
  const dome = parseMesh(readFileSync(new URL('../web/meshes/dome5.ply', import.meta.url), 'utf8'), 'dome.ply');
  for (const mesh of [torus(14, 8), dome]) {
    const paint = editor(mesh), otherFields = [];
    for (const preset of PAINT_PRESETS) {
      assert.equal(applyPaintPreset(preset.id, mesh, paint), true);
      checkTargets(paint);
      const {values} = paint.field();
      for (const other of otherFields) assert.ok(values.some((v, i) => Math.abs(v - other[i]) > .1));
      const result = solve(mesh.vertices, mesh.faces, {targetCurvature: values});
      assert.ok(result.converged && result.curvatureError < 1e-5);
      otherFields.push(values);
    }
  }
});

test('long-wavelength paint on the trefoil retains a usable 3D reconstruction', () => {
  const mesh = SHAPE_PRESETS.find(preset => preset.id === 'trefoil').create();
  const paint = editor(mesh);
  applyPaintPreset('dipole', mesh, paint);
  const result = solve(mesh.vertices, mesh.faces, {targetCurvature: paint.field().values});
  assert.ok(result.converged && result.edgeError < .015);
});

test('presets apply independently to disconnected sheets and are invariant to rigid motion and scale', () => {
  const mesh = torus(18, 8);
  const moved = mesh.vertices.map(([x, y, z]) => [11 + 37 * (.8*x + .6*z), -9 + 37*y, 14 + 37*(-.6*x + .8*z)]);
  const joined = {vertices: [...mesh.vertices, ...moved], faces: [
    ...mesh.faces, ...mesh.faces.map(face => face.map(i => i + moved.length).reverse())
  ]};
  const paint = editor(joined);
  for (const preset of PAINT_PRESETS) {
    applyPaintPreset(preset.id, joined, paint);
    checkTargets(paint);
    const {values} = paint.field();
    for (let i = 0; i < moved.length; i++) {
      assert.equal(paint.painted.has(i), paint.painted.has(i + moved.length));
      assert.ok(Math.abs(values[i] - values[i + moved.length]) < 1e-12);
    }
  }
});

test('applying a preset is one reversible edit and identical applications add no undo step', () => {
  const mesh = torus(16, 8), paint = editor(mesh);
  paint.dab(7, 1, .15);
  const manual = new Map(paint.painted);
  applyPaintPreset('dipole', mesh, paint);
  const pattern = new Map(paint.painted);
  assert.equal(applyPaintPreset('dipole', mesh, paint), false);
  paint.dab(7, 1, -.2);
  assert.ok(paint.undo());
  assert.deepEqual(paint.painted, pattern);
  assert.ok(paint.undo());
  assert.deepEqual(paint.painted, manual);
  assert.throws(() => applyPaintPreset('missing', mesh, paint), /preset/);
  assert.deepEqual(paint.painted, manual);
  assert.ok(paint.undo());
  assert.equal(paint.count, 0);
});

test('minimal boundary meshes retain room for a subsequent manual stroke', () => {
  const mesh = {vertices: [[0,0,0], [1,0,0], [.5,Math.sqrt(3)/2,0]], faces: [[0,1,2]]};
  const paint = editor(mesh);
  for (const preset of PAINT_PRESETS) {
    applyPaintPreset(preset.id, mesh, paint);
    checkTargets(paint);
    assert.equal(paint.count, 2);
    const [i] = paint.indices;
    paint.dab(i, 0, paint.initial[i] + .05);
    assert.equal(paint.field().error, null);
  }
});
