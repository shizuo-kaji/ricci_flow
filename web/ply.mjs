const decoder = new TextDecoder();
const types = new Map();
for (const [name, alias, width, getter] of [
  ['char', 'int8', 1, 'getInt8'], ['uchar', 'uint8', 1, 'getUint8'],
  ['short', 'int16', 2, 'getInt16'], ['ushort', 'uint16', 2, 'getUint16'],
  ['int', 'int32', 4, 'getInt32'], ['uint', 'uint32', 4, 'getUint32'],
  ['float', 'float32', 4, 'getFloat32'], ['double', 'float64', 8, 'getFloat64']
]) {
  const type = {width, getter, integer: !getter.startsWith('getFloat')};
  types.set(name, type);
  types.set(alias, type);
}

function scalarType(name) {
  const type = types.get(name);
  if (!type) throw Error('Unsupported PLY property type: ' + name + '.');
  return type;
}

function parseHeader(bytes) {
  const elements = [];
  let offset = 0, format, first = true, crOnly = false;
  while (offset < bytes.length) {
    const start = offset;
    while (offset < bytes.length && bytes[offset] !== 10 && bytes[offset] !== 13) offset++;
    const line = decoder.decode(bytes.subarray(start, offset)).trim();
    if (offset < bytes.length) {
      const carriageReturn = bytes[offset++] === 13;
      if (first) crOnly = carriageReturn && bytes[offset] !== 10;
      // With a CR-only header the next byte after end_header is already payload.
      if (carriageReturn && bytes[offset] === 10 && (!crOnly || line !== 'end_header')) offset++;
    }
    if (first) {
      if (!line) continue;
      if (line !== 'ply') throw Error('A PLY 1.0 file must begin with a ply header.');
      first = false;
      continue;
    }
    if (line === 'end_header') {
      if (!format) throw Error('The PLY header has no format declaration.');
      return {elements, format, offset};
    }
    const [key, ...fields] = line.split(/\s+/);
    if (key === 'format') {
      [format] = fields;
      if (!['ascii', 'binary_little_endian', 'binary_big_endian'].includes(format) || fields[1] !== '1.0') {
        throw Error('Save the mesh as an ASCII or binary PLY 1.0 file.');
      }
    } else if (key === 'element') {
      const [name, length] = fields, count = Number(length);
      if (!name || !Number.isSafeInteger(count) || count < 0) throw Error('Invalid PLY element declaration.');
      if (elements.some(el => el.name === name)) throw Error('Duplicate PLY element: ' + name + '.');
      elements.push({name, count, props: []});
    } else if (key === 'property') {
      const el = elements.at(-1);
      if (!el) throw Error('A PLY property appears before its element declaration.');
      const list = fields[0] === 'list';
      if (fields.length !== (list ? 4 : 2)) throw Error('Invalid PLY property declaration.');
      const name = fields.at(-1), type = scalarType(fields.at(-2));
      const countType = list ? scalarType(fields[1]) : null;
      if (list && !countType.integer) throw Error('A PLY list count must have an integer type.');
      if (el.props.some(p => p.name === name)) throw Error('Duplicate PLY property: ' + name + '.');
      el.props.push({name, type, countType});
    }
  }
  throw Error('The PLY header is missing its end_header line.');
}

const truncated = 'The PLY file ends before all declared property values are present.';

function binaryReader(bytes, littleEndian) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let offset = 0;
  function take(size) {
    const start = offset;
    offset += size;
    if (offset > view.byteLength) throw Error(truncated);
    return start;
  }
  return {
    scalar(type) { return view[type.getter](take(type.width), littleEndian); },
    skip(type, count) { take(type.width * count); }
  };
}

function asciiReader(bytes) {
  const lines = decoder.decode(bytes).split(/\r\n?|\n/);
  let line = 0, fields, at;
  return {
    nextRecord() {
      while (line < lines.length && !lines[line].trim()) line++;
      if (line >= lines.length) throw Error(truncated);
      fields = lines[line++].trim().split(/\s+/);
      at = 0;
    },
    scalar(type) {
      if (at >= fields.length) throw Error(truncated);
      const value = Number(fields[at++]);
      if (!Number.isFinite(value) || (type.integer && !Number.isSafeInteger(value))) {
        throw Error('A PLY property contains an invalid numeric value.');
      }
      return value;
    },
    skip(type, count) {
      at += count;
      if (at > fields.length) throw Error(truncated);
    },
    endRecord() {
      if (at !== fields.length) throw Error('A PLY row has more values than its declared properties.');
    }
  };
}

export function parsePly(bytes, textOnly = false) {
  const {elements, format, offset} = parseHeader(bytes);
  if (textOnly && format !== 'ascii') throw Error('Open binary PLY data as a file or byte buffer.');
  const vertex = elements.find(el => el.name === 'vertex'), face = elements.find(el => el.name === 'face');
  if (!face?.count) throw Error('This PLY has no faces. Export a triangle mesh with faces to use it here.');
  if (!vertex || vertex.count < 3) throw Error('The PLY mesh needs at least three vertices.');
  const axes = ['x', 'y', 'z'];
  for (const name of axes) {
    if (!vertex.props.some(p => p.name === name && !p.countType)) throw Error('Each PLY vertex needs scalar x, y and z coordinates.');
  }
  const indices = face.props.find(p => p.name === 'vertex_indices' || p.name === 'vertex_index');
  if (!indices?.countType || !indices.type.integer) throw Error('Each PLY face needs a vertex_indices list of integer vertex IDs.');
  const body = bytes.subarray(offset);
  const reader = format === 'ascii' ? asciiReader(body) : binaryReader(body, format === 'binary_little_endian');
  const vertices = [], faces = [];
  for (const el of elements) {
    // Empty element types have no payload, even for a nonzero declared count.
    if (!el.props.length) continue;
    for (let i = 0; i < el.count; i++) {
      reader.nextRecord?.();
      const point = el === vertex ? [] : null;
      for (const p of el.props) {
        const count = p.countType ? reader.scalar(p.countType) : 1;
        if (count < 0) throw Error('A PLY list has a negative item count.');
        if (p === indices) {
          if (count !== 3) throw Error('Triangulate the PLY faces before importing the mesh.');
          const ids = [reader.scalar(p.type), reader.scalar(p.type), reader.scalar(p.type)];
          if (ids.some(id => id < 0 || id >= vertex.count)) throw Error('A PLY face references a vertex outside the mesh.');
          faces.push(ids);
        } else if (point && axes.includes(p.name)) {
          const value = reader.scalar(p.type);
          if (!Number.isFinite(value)) throw Error('Each PLY vertex needs three finite coordinates.');
          point[axes.indexOf(p.name)] = value;
        } else {
          reader.skip(p.type, count);
        }
      }
      reader.endRecord?.();
      if (point) vertices.push(point);
    }
  }
  return {vertices, faces};
}
