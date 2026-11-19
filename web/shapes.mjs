const TAU = 2 * Math.PI;

// Connect two equally sampled rings. Each ring wraps to its first vertex.
function joinRings(faces, first, next, sides) {
  for (let j = 0; j < sides; j++) {
    const k = (j + 1) % sides;
    faces.push([first + j, next + j, next + k], [first + j, next + k, first + k]);
  }
}

function trefoil() {
  const vertices = [], faces = [];
  const rings = 56, sides = 8, tubeRadius = 0.23, minor = 0.53;
  for (let i = 0; i < rings; i++) {
    const t = TAU * i / rings;
    const x = Math.cos(2*t), y = Math.sin(2*t), c = Math.cos(3*t), s = Math.sin(3*t);
    const radius = 1.15 + minor*c;
    const center = [radius*x, radius*y, minor*s];
    // The normal of the underlying torus gives a periodic frame for the knot.
    const normal = [c*x, c*y, s];
    const dx = -3*minor*s*x - 2*radius*y, dy = -3*minor*s*y + 2*radius*x, dz = 3*minor*c;
    const speed = Math.hypot(dx, dy, dz);
    const binormal = [(normal[1]*dz - normal[2]*dy) / speed,
      (normal[2]*dx - normal[0]*dz) / speed, (normal[0]*dy - normal[1]*dx) / speed];
    for (let j = 0; j < sides; j++) {
      const angle = TAU*j/sides;
      vertices.push(center.map((v, k) => v + tubeRadius*(Math.cos(angle)*normal[k] + Math.sin(angle)*binormal[k])));
    }
    joinRings(faces, i*sides, ((i + 1) % rings)*sides, sides);
  }
  return {vertices, faces};
}

function catenoid() {
  const vertices = [], faces = [];
  const rings = 11, sides = 28, waist = 0.78;
  for (let i = 0; i < rings; i++) {
    const height = 1.15 * (1 - 2*i/(rings - 1));
    const radius = waist*Math.cosh(height);
    for (let j = 0; j < sides; j++) {
      const angle = TAU*j/sides;
      vertices.push([radius*Math.cos(angle), radius*Math.sin(angle), waist*height]);
    }
    if (i > 0) joinRings(faces, (i - 1)*sides, i*sides, sides);
  }
  return {vertices, faces};
}

function ruffledDisk() {
  const vertices = [[0, 0, 0]], faces = [];
  const rings = 8;
  let inner = [0];
  for (let i = 1; i <= rings; i++) {
    const radius = i/rings, outer = [];
    for (let j = 0; j < 6*i; j++) {
      const angle = TAU*j/(6*i);
      const phase = 5*angle - 0.65*radius;
      const r = radius*(1 + 0.08*radius**2*Math.cos(phase));
      const z = 0.30*radius**3*Math.cos(phase) + 0.10*radius**2;
      outer.push(vertices.length);
      vertices.push([r*Math.cos(angle), r*Math.sin(angle), z]);
    }
    // Each of six sectors gains one triangle along its outer edge per ring.
    // This keeps the central vertex valence at six as the sampling gets finer.
    for (let sector = 0; sector < 6; sector++) {
      for (let j = 0; j < i; j++) {
        const a = sector*(i - 1) + j, b = sector*i + j;
        const p = inner[a % inner.length], q = outer[(b + 1) % outer.length];
        faces.push([p, outer[b], q]);
        if (j < i - 1) faces.push([p, q, inner[(a + 1) % inner.length]]);
      }
    }
    inner = outer;
  }
  return {vertices, faces};
}

export const SHAPE_PRESETS = [
  {id: 'trefoil', label: 'Trefoil knot', description: 'A closed tube winds over and under itself in a three-lobed knot.', create: trefoil},
  {id: 'catenoid', label: 'Catenoid', description: 'Two circular rims frame a narrow, saddle-curved waist.', create: catenoid},
  {id: 'ruffled-disk', label: 'Ruffled disk', description: 'Five waves rise and fall around the edge of an open disk.', create: ruffledDisk}
];
