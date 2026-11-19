// Local thin-shell bending on the interior hinges of the target metric.
// Each row is sqrt(weight * length² / dualArea) * wrap(theta - rest).
function difference(x, p, q) {
  return [x[p] - x[q], x[p+1] - x[q+1], x[p+2] - x[q+2]];
}

const dot = (u, v) => u[0]*v[0] + u[1]*v[1] + u[2]*v[2];
const cross = (u, v) => [u[1]*v[2] - u[2]*v[1], u[2]*v[0] - u[0]*v[2], u[0]*v[1] - u[1]*v[0]];

// Hinge order is edge endpoints, then opposite vertices. Using the same
// ordering for the rest and current angles makes the penalty winding-independent.
function dihedral(x, hinge, gradient, offset) {
  const [a, b, c, d] = hinge;
  const edge = difference(x, b, a), left = difference(x, c, a), right = difference(x, d, a);
  const length = Math.hypot(...edge), leftLength = Math.hypot(...left), rightLength = Math.hypot(...right);
  if (![length, leftLength, rightLength].every(v => Number.isFinite(v) && v > 0)) return NaN;
  const e = edge.map(v => v / length);
  const n = cross(e, left.map(v => v / leftLength)), m = cross(right.map(v => v / rightLength), e);
  const sinL = Math.hypot(...n), sinR = Math.hypot(...m);
  if (Math.min(sinL, sinR) <= 32*Number.EPSILON) return NaN;
  for (let k = 0; k < 3; k++) { n[k] /= sinL; m[k] /= sinR; }
  if (gradient) {
    const tL = dot(left, e)/length, tR = dot(right, e)/length;
    for (let k = 0; k < 3; k++) {
      const gc = -n[k]/(leftLength*sinL), gd = -m[k]/(rightLength*sinR);
      const gb = -tL*gc - tR*gd;
      gradient[offset+k] = -gb-gc-gd;
      gradient[offset+3+k] = gb;
      gradient[offset+6+k] = gc;
      gradient[offset+9+k] = gd;
    }
  }
  return Math.atan2(dot(e, cross(n, m)), dot(n, m));
}

export class HingeBending {
  constructor(mesh, lengths, weight, target) {
    const areas = mesh.faceEdges.map(face => {
      const [a, b, c] = face.map(e => lengths[e]).sort((u, v) => v-u);
      return 0.25 * Math.sqrt((a+(b+c)) * (c-(a-b)) * (c+(a-b)) * (a+(b-c)));
    });
    const reference = target === 'reference' ? mesh.vertices.flat() : null;
    this.hinges = [];
    this.scales = [];
    this.rest = [];
    for (let e = 0; e < mesh.edges.length; e++) {
      const adjacent = mesh.edgeFaces[e];
      if (adjacent.length !== 2) continue;
      const [i, j] = mesh.edges[e];
      const [k, l] = adjacent.map(f => mesh.faces[f].find(v => v !== i && v !== j));
      const hinge = [3*i, 3*j, 3*k, 3*l];
      const dualArea = (areas[adjacent[0]] + areas[adjacent[1]])/3;
      const rest = reference ? dihedral(reference, hinge) : 0;
      if (!Number.isFinite(rest) || !(dualArea > 0)) {
        throw new Error('Bending requires nondegenerate triangles on each interior hinge.');
      }
      this.hinges.push(hinge);
      this.scales.push(Math.sqrt(weight) * lengths[e] / Math.sqrt(dualArea));
      this.rest.push(rest);
    }
    this.gradient = new Float64Array(12 * this.hinges.length);
  }

  // Return sum of squared weighted angle residuals and optionally linearize.
  evaluate(x, linearization) {
    let energy = 0;
    for (let h = 0; h < this.hinges.length; h++) {
      const hinge = this.hinges[h], start = 12*h, scale = this.scales[h];
      const angle = dihedral(x, hinge, linearization && this.gradient, start);
      if (!Number.isFinite(angle)) return Infinity;
      const change = angle - this.rest[h];
      const residual = scale * Math.atan2(Math.sin(change), Math.cos(change));
      energy += residual**2;
      if (linearization) {
        for (let i = 0; i < 4; i++) for (let k = 0; k < 3; k++) {
          const entry = start + 3*i + k, axis = hinge[i] + k;
          const g = this.gradient[entry] *= scale;
          linearization.rhs[axis] -= g*residual;
          linearization.diagonal[axis] += g*g;
        }
      }
    }
    return energy;
  }

  // Accumulate JᵀJv using the signed dihedral rows at the last linearization.
  addNormalProduct(v, out) {
    for (let h = 0; h < this.hinges.length; h++) {
      const hinge = this.hinges[h], start = 12*h;
      let projection = 0;
      for (let i = 0; i < 4; i++) for (let k = 0; k < 3; k++) {
        projection += this.gradient[start + 3*i+k] * v[hinge[i]+k];
      }
      for (let i = 0; i < 4; i++) for (let k = 0; k < 3; k++) {
        out[hinge[i]+k] += this.gradient[start + 3*i+k] * projection;
      }
    }
  }
}
