import test from 'node:test';
import assert from 'node:assert/strict';
import {diagnoseVertexStars} from '../web/immersion.mjs';

function windingFan(turns) {
  const vertices = [[0,0,0]], faces = [], n = 11;
  for (let i = 0; i < n; i++) {
    const phi = 2*Math.PI*i/n;
    vertices.push([Math.cos(turns*phi), Math.sin(turns*phi), 0.23*Math.sin(phi)]);
    const face = [0, i+1, (i+1)%n+1];
    faces.push(i%2 ? face.reverse() : face);
  }
  return {vertices, faces};
}

test('closed vertex-star spin parity and mixed face orientation', () => {
  for (const turns of [1, 2, 3]) {
    const {vertices, faces} = windingFan(turns);
    for (const scale of [1e-12, 1e12]) {
      const rotated = vertices.map(([x,y,z]) => [z+2, x-1, y+3].map(v => scale*v));
      const r = diagnoseVertexStars(rotated, faces);
      assert.deepEqual(r.pinchVertices, turns === 2 ? [0] : []);
      assert.equal(r.classifiedInteriorVertices, 1);
      assert.deepEqual(r.indeterminateVertices, []);
      assert.deepEqual(r.boundaryVertices, vertices.slice(1).map((_, i) => i+1));
    }
  }
});

test('half-turn, zero-area face and disconnected link give indeterminate stars', () => {
  const fan = [[0,1,2],[0,2,3],[0,3,4],[0,4,1]];
  for (const vertices of [
    [[0,0,0],[1,0,0],[0,1,0],[2,0,0],[0,-1,0]],
    [[0,0,0],[1,0,0],[2,0,0],[0,1,0],[-1,0,0]]
  ]) {
    const r = diagnoseVertexStars(vertices, fan);
    assert.deepEqual(r.indeterminateVertices, [0]);
    assert.equal(r.classifiedInteriorVertices, 0);
  }
  const bowtie = [[0,0,0],[1,0,0],[0,1,0],[-1,0,0],[0,-1,0],[0,0,1]];
  const r = diagnoseVertexStars(bowtie, [[0,1,2],[0,3,4]]);
  assert.deepEqual(r.indeterminateVertices, [0,5]);
  assert.deepEqual(r.boundaryVertices, [1,2,3,4]);
});
