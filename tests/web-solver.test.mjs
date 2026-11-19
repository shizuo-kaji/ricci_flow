import test from 'node:test';
import assert from 'node:assert/strict';
import {packingMetric, prepareMesh, reconstruct, solve} from '../web/solver.mjs';
import {HingeBending} from '../web/bending.mjs';

const tetra = [[1.4,1,1],[-1,-1,1],[-1,1,-1],[1,-1,-1]];
const shell = [[0,1,2],[1,2,3],[0,3,1],[0,2,3]];
const disk = [[0,0,0],[1,0,0],[0,1,0],[-1,0,0],[0,-1,0]];
const fan = [[0,1,2],[0,2,3],[0,3,4],[0,4,1]];
const maxDiff = (a, b) => Math.max(...Array.from(a, (v, i) => Math.abs(v - b[i])));

test('packing recovers the reference metric and differentiates angle defects', () => {
  const m = prepareMesh(tetra, shell), u = Float64Array.from([.05,-.07,.08,.03]);
  assert.ok(maxDiff(packingMetric(m, new Float64Array(m.n)).lengths, m.base) < 1e-14);
  const state = packingMetric(m, u), direction = [.3,-.8,-.7,.2];
  const delta = 1e-6;
  const plus = packingMetric(m, u.map((x,i) => x + delta*direction[i]));
  const minus = packingMetric(m, u.map((x,i) => x - delta*direction[i]));
  const numerical = plus.curvature.map((v,i) => (v - minus.curvature[i]) / (2*delta));
  const analytical = new Float64Array(m.n);
  m.edges.forEach(([i,j], e) => {
    const value = state.weights[e] * (direction[i] - direction[j]);
    analytical[i] += value;
    analytical[j] -= value;
  });
  assert.ok(maxDiff(numerical, analytical) < 1e-8);
});

test('previews reflect accepted steps and preserve the final fit', () => {
  const events = [], active = solve(tetra, shell, {previewEvery:2}, e => events.push(e));
  const silent = [];
  const final = solve(tetra, shell, {}, e => silent.push(e));
  assert.deepEqual(active, final);
  assert.equal(silent.filter(e => e.type === 'preview').length, 0);
  assert.ok(final.converged && final.curvatureError < 1e-5 && final.edgeError < 1e-6);
  assert.ok(final.history.slice(1).every((v,i) => v < final.history[i]));
  const expected = [];
  for (let i = 2; i < final.iteration; i += 2) expected.push(i);
  const frames = events.filter(e => e.type === 'preview');
  assert.deepEqual(frames.map(e => e.iteration), expected);
  for (const frame of frames) {
    assert.ok(frame.edgeError < 1e-6);
    const fittedMesh = prepareMesh(frame.vertices, shell);
    const fittedMetric = packingMetric(fittedMesh, new Float64Array(4));
    assert.ok(maxDiff(fittedMetric.curvature, frame.curvature) < 1e-5);
  }
});

test('a planar disk can lift out of plane to realize prescribed positive curvature', () => {
  const target = [.6, ...Array(4).fill((2*Math.PI - .6)/4)];
  const final = solve(disk, fan, {targetCurvature:target, tolerance:1e-7});
  assert.ok(final.converged);
  assert.ok(final.edgeError < 1e-6);
  assert.ok(Math.abs(final.vertices[0][2] - final.vertices[1][2]) > .1);
  assert.ok(Math.abs(final.curvature.reduce((a,b) => a+b) - 2*Math.PI) < 1e-10);
});

test('no-op targets preserve 3D input and zero accepted steps', () => {
  for (const opts of [{target:'initial'}, {strength:0}, {target:'flat-interior'}]) {
    const final = solve(disk, fan, opts);
    assert.deepEqual(final.vertices, disk);
    assert.equal(final.iteration, 0);
    assert.equal(final.reason, 'tolerance');
  }
});

test('component-wise Gauss–Bonnet prevents globally balanced impossible targets', () => {
  const vertices = [...tetra, ...tetra.map(p => p.map(v => v+5))];
  const faces = [...shell, ...shell.map(f => f.map(v => v+4))];
  const targetCurvature = [...Array(4).fill(Math.PI+.1), ...Array(4).fill(Math.PI-.1)];
  assert.throws(() => solve(vertices, faces, {targetCurvature}), /Gauss/);
  const final = solve(vertices, faces);
  assert.ok(final.converged && final.edgeError < 1e-6);
});

test('invalid meshes, tolerances and infeasible targets return actionable errors', () => {
  assert.throws(() => solve(tetra, [shell[0], shell[0]]), /Duplicate/);
  assert.throws(() => solve(disk, [[0,1,2],[0,3,4]]), /fans/);
  assert.throws(() => solve([[0,0,0],[1,0,0],[2,0,0]], [[0,1,2]]), /Degenerate/);
  assert.throws(() => solve(tetra, shell, {tolerance:0}), /Tolerance/);
  assert.throws(() => solve(tetra, shell, {targetCurvature:[1,1,1,1]}), /Gauss/);
  assert.throws(() => solve(disk, fan, {targetCurvature:[-8,0,0,0,0]}), /attainable/);
});

const folded = [[0,0,0], [1,0,0], [.2,1,0], [.1,-.6,.8]];
const panel = [[0,1,2], [1,0,3]];

test('bending can unfold an isometric hinge or restore its signed reference angle', () => {
  const mesh = prepareMesh(folded, panel), metric = packingMetric(mesh, new Float64Array(4));
  const flat = new HingeBending(mesh, metric.lengths, .02, 'flat');
  const unbent = solve(folded, panel, {target:'initial', bendingWeight:.02});
  assert.ok(flat.evaluate(folded.flat()) > .05);
  assert.ok(flat.evaluate(unbent.vertices.flat()) < 1e-12 && unbent.edgeError < 1e-7);
  const reflected = folded.map(([x,y,z]) => [x,y,-z]);
  const reference = new HingeBending(mesh, metric.lengths, .02, 'reference');
  const restored = reconstruct(mesh, metric, reflected, 100, {bendingWeight:.02, bendingTarget:'reference'});
  assert.ok(reference.evaluate(reflected.flat()) > .2);
  assert.ok(reference.evaluate(restored.vertices.flat()) < 1e-12 && restored.edgeError < 1e-7);
  assert.deepEqual(solve(folded, panel, {target:'initial', bendingWeight:0}).vertices, folded);
});

test('hinge gradient agrees with finite differences on a skewed 3D panel', () => {
  const mesh = prepareMesh(folded, panel), bending = new HingeBending(mesh, mesh.base, .02, 'flat');
  const x = folded.flat(), rhs = new Float64Array(x.length);
  bending.evaluate(x, {rhs, diagonal: new Float64Array(x.length)});
  for (let i = 0; i < x.length; i++) {
    const p = x.slice(), m = x.slice();
    p[i] += 1e-6; m[i] -= 1e-6;
    const numerical = (bending.evaluate(p) - bending.evaluate(m)) / 2e-6;
    assert.ok(Math.abs(numerical + 2*rhs[i]) < 1e-8);
  }
});
