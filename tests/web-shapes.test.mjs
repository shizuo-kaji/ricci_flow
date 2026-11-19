import test from 'node:test';
import assert from 'node:assert/strict';
import {SHAPE_PRESETS} from '../web/shapes.mjs';
import {prepareMesh, solve} from '../web/solver.mjs';

const expected = {
  trefoil: {chi: 0, rims: []},
  catenoid: {chi: 0, rims: [28, 28]},
  'ruffled-disk': {chi: 1, rims: [48]}
};

function boundaryLoops(mesh) {
  const links = new Map();
  mesh.edges.forEach(([a, b], index) => {
    if (mesh.edgeFaces[index].length !== 1) return;
    if (!links.has(a)) links.set(a, []);
    if (!links.has(b)) links.set(b, []);
    links.get(a).push(b);
    links.get(b).push(a);
  });
  const loops = [], seen = new Set();
  for (const start of links.keys()) {
    if (seen.has(start)) continue;
    let previous = null, at = start, size = 0;
    do {
      assert.ok(!seen.has(at));
      seen.add(at);
      size++;
      const neighbors = links.get(at);
      assert.equal(neighbors.length, 2);
      const next = neighbors.find(id => id !== previous);
      previous = at;
      at = next;
    } while (at !== start);
    loops.push(size);
  }
  return loops.sort((a, b) => a - b);
}

for (const {id, label, create} of SHAPE_PRESETS) {
  test(label + ' has consistently wound seams and the expected boundary loops', () => {
    const {vertices, faces} = create(), mesh = prepareMesh(vertices, faces);
    assert.equal(mesh.components.length, 1);
    assert.equal(vertices.length - mesh.edges.length + faces.length, expected[id].chi);
    assert.deepEqual(boundaryLoops(mesh), expected[id].rims);
    mesh.edges.forEach(([a, b], e) => {
      if (mesh.edgeFaces[e].length !== 2) return;
      const forwards = mesh.edgeFaces[e].map(index => {
        const f = faces[index];
        return f[(f.indexOf(a) + 1) % 3] === b;
      });
      assert.notEqual(forwards[0], forwards[1]);
    });
  });

  test(label + ' converges with the web lab’s default curvature target', () => {
    const {vertices, faces} = create();
    const result = solve(vertices, faces, {strength: 0.75});
    assert.ok(result.converged && result.iteration > 0);
    assert.ok(result.curvatureError < 1e-5);
    assert.ok(result.edgeError < 0.01);
    assert.ok(result.vertices.every(point => point.every(Number.isFinite)));
  });
}
