import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {encodePly, parseMesh, topology} from '../web/mesh.mjs';

const triangle = {vertices: [[0, 0, 0], [2, 0, 0], [0, 3, 1.5]], faces: [[0, 1, 2]]};

function binaryTriangle(order, newline) {
  const header = [
    'ply', 'format binary_' + (order === 'LE' ? 'little' : 'big') + '_endian 1.0',
    'comment word end_header in comment', 'element edge 1', 'property int8 tag',
    'property list uint16 float32 weights', 'element face 1', 'property uchar red',
    'property list ushort uint32 vertex_index', 'property list char short materials',
    'element vertex 3', 'property float z', 'property int label',
    'property double x', 'property float64 y', 'end_header', ''
  ].join(newline);
  const chunks = [Buffer.from(header)];
  function put(method, size, value) {
    const data = Buffer.alloc(size);
    data[method + (size === 1 ? '' : order)](value);
    chunks.push(data);
  }
  put('writeInt8', 1, 10); // A valid payload byte equal to LF.
  put('writeUInt16', 2, 2);
  put('writeFloat', 4, -1.5);
  put('writeFloat', 4, 2.5);
  put('writeUInt8', 1, 220);
  put('writeUInt16', 2, 3);
  for (const id of triangle.faces[0]) put('writeUInt32', 4, id);
  put('writeInt8', 1, 1);
  put('writeInt16', 2, -129);
  for (const [x, y, z] of triangle.vertices) {
    put('writeFloat', 4, z);
    put('writeInt32', 4, 70000);
    put('writeDouble', 8, x);
    put('writeDouble', 8, y);
  }
  return Buffer.concat(chunks);
}

test('binary PLY respects byte order, view boundaries and declared property order', () => {
  for (const [order, newline] of [['LE', '\n'], ['BE', '\r\n'], ['LE', '\r']]) {
    const file = binaryTriangle(order, newline);
    const padded = Buffer.concat([Buffer.from('padding'), file, Buffer.from([0xff])]);
    const slice = padded.subarray(7, 7 + file.length);
    for (const source of [slice, new DataView(slice.buffer, slice.byteOffset, slice.byteLength), Uint8Array.from(slice).buffer]) {
      assert.deepEqual(parseMesh(source, 'Test.PLY'), triangle);
    }
    assert.throws(() => parseMesh(slice.subarray(0, -1), 'truncated.ply'), /ends before/);
  }
});

test('bundled PLY with extra properties round-trips through the editor export', () => {
  const file = readFileSync(new URL('../web/meshes/bunny.ply', import.meta.url));
  const mesh = parseMesh(file, 'bunny.ply');
  assert.equal(mesh.vertices.length, 102);
  assert.equal(mesh.faces.length, 200);
  assert.equal(topology(mesh).chi, 2);
  assert.deepEqual(parseMesh(file.toString('utf8'), 'bunny.ply'), mesh);
  assert.deepEqual(parseMesh(encodePly(mesh.vertices, mesh.faces), 'export.ply'), mesh);
});

test('point clouds, polygons, incomplete records and unsupported PLY declarations explain how to recover', () => {
  const ascii = encodePly(triangle.vertices, triangle.faces);
  const parse = source => parseMesh(source, 'input.ply');
  assert.throws(() => parse(ascii.replace('element face 1', 'element face 0')), /no faces/);
  assert.throws(() => parse(ascii.replace('3 0 1 2', '4 0 1 2 0')), /Triangulate/);
  assert.throws(() => parse(ascii.replace('3 0 1 2', '3 0 1')), /ends before/);
  assert.throws(() => parse(ascii.replace('3 0 1 2', '-1')), /negative item count/);
  assert.throws(() => parse(ascii.replace('3 0 1 2', '3 0 1 8')), /outside the mesh/);
  assert.throws(() => parse(ascii.replace('property double z\n', '')), /x, y and z/);
  assert.throws(() => parse(ascii.replace('end_header', 'missing')), /end_header/);
  assert.throws(() => parse(ascii.replace('format ascii 1.0', 'format ascii 2.0')), /PLY 1.0/);
  assert.throws(() => parse(ascii.replace('list uchar', 'list float')), /integer type/);
});

test('ASCII mesh strings and OBJ bytes retain the established import behaviour', () => {
  const obj = 'v 0 0 0\nv 2 0 0\nv 0 3 1.5\nf -3/1/1 -2/2/2 -1/3/3\n';
  assert.deepEqual(parseMesh(obj, 'triangle.obj'), triangle);
  assert.deepEqual(parseMesh(new TextEncoder().encode(obj), 'triangle.obj'), triangle);
  const ply = encodePly(triangle.vertices, triangle.faces);
  assert.deepEqual(parseMesh('\uFEFF\n' + ply, 'triangle'), triangle);
  assert.deepEqual(parseMesh(Buffer.from('\uFEFF\n' + ply), 'triangle.ply'), triangle);
});
