import {torus, parseMesh, topology, angleDefects, parseTarget, encodePly} from './mesh.mjs';
import {MeshViewer} from './viewer.mjs';
import {CurvaturePaint} from './paint.mjs';
import {SHAPE_PRESETS} from './shapes.mjs';
import {PAINT_PRESETS, applyPaintPreset} from './paint-presets.mjs';

const $ = id => document.getElementById(id);
const viewer = new MeshViewer($('view'));
const samples = [
  {id: 'torus', label: 'Torus', description: 'A ring with positive curvature on the outside and negative curvature on the inside.', create: torus},
  ...SHAPE_PRESETS,
  {id: 'dome5', label: 'Dome', description: 'A triangulated cap with an open rim.', file: 'dome5.ply'},
  {id: 'bunny', label: 'Bunny', description: 'A low-poly rabbit with long ears.', file: 'bunny.ply'}
];
for (const {id, label} of samples) $('sample').add(new Option(label, id));
for (const {id, label} of PAINT_PRESETS) $('paintPreset').add(new Option(label, id));
const state = {frames: [], history: [], index: 0, worker: null, busy: false, sample: 'torus',
  editing: false, brush: 'Paint'};

const paths = {
  cube: 'M12 2.5L20.5 7.2V16.8L12 21.5L3.5 16.8V7.2ZM3.8 7.5L12 12L20.2 7.5M12 12V21',
  book: 'M3 4.5Q8 3 12 6Q16 3 21 4.5V18Q16 16.5 12 19.5Q8 16.5 3 18ZM12 6V19',
  code: 'M8 6L2.5 12L8 18M16 6L21.5 12L16 18M14 4L10 20',
  plus: 'M12 5V19M5 12H19', minus: 'M5 12H19', close: 'M5 5L19 19M19 5L5 19',
  fit: 'M3 9V3H9M15 3H21V9M21 15V21H15M9 21H3V15',
  mesh: 'M3 19L7 5L19 4L21 18ZM3 19L13 12L19 4M7 5L13 12L21 18',
  settings: 'M3 6H8M13 6H21M3 12H14M19 12H21M3 18H6M11 18H21M12.5 6a2 2 0 1 1-4 0a2 2 0 1 1 4 0M18.5 12a2 2 0 1 1-4 0a2 2 0 1 1 4 0M10.5 18a2 2 0 1 1-4 0a2 2 0 1 1 4 0',
  activity: 'M2 13H7L10.5 4L14 19L17 11H22',
  upload: 'M12 16V3M7 8L12 3L17 8M3 15V20H21V15',
  download: 'M12 3V15M7 10L12 15L17 10M3 16V20H21V16', play: 'M7 3.5L20 12L7 20.5Z',
  stop: 'M5 5H19V19H5Z'
};

function icon(el, name = el.dataset.icon) {
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = 48;
  const c = canvas.getContext('2d');
  c.scale(2, 2);
  c.strokeStyle = c.fillStyle = getComputedStyle(el).color;
  c.lineWidth = 1.65; c.lineJoin = c.lineCap = 'round';
  const p = new Path2D(paths[name]);
  if (name === 'play' || name === 'stop') c.fill(p); else c.stroke(p);
  el.replaceChildren(canvas);
  el.setAttribute('aria-hidden', 'true');
}
document.querySelectorAll('[data-icon]').forEach(el => icon(el));
{
  const c = $('mark').getContext('2d');
  c.scale(2, 2);
  c.fillStyle = '#08796c'; c.beginPath(); c.roundRect(0, 0, 36, 36, 8); c.fill();
  const points = [[5,23],[10,14],[17,10],[24,8],[31,12],[26,22],[18,27],[11,28],[14,20],[20,15],[23,18],[18,21]];
  c.strokeStyle = '#abeadd'; c.lineWidth = .9; c.lineJoin = 'round';
  for (const face of [[0,1,8],[1,2,8],[2,9,8],[2,3,9],[3,4,9],[4,10,9],[4,5,10],[5,11,10],[5,6,11],[6,7,11],[7,8,11],[0,8,7]]) {
    c.beginPath(); c.moveTo(...points[face[0]]); c.lineTo(...points[face[1]]); c.lineTo(...points[face[2]]); c.closePath(); c.stroke();
  }
}

function status(label, kind = '') {
  $('status').textContent = label;
  $('status').dataset.state = kind;
}

function error(message = '') {
  $('error').textContent = message;
  $('error').hidden = !message;
}

function note(message = '') {
  $('runNote').textContent = message;
  $('runNote').hidden = !message;
}

function busy(on, working = false) {
  state.busy = on;
  $('runInputs').disabled = on;
  $('run').disabled = on && !working;
  $('run').dataset.running = String(working);
  $('runLabel').textContent = working ? 'Stop optimization' : 'Run optimization';
  icon($('runIcon'), working ? 'stop' : 'play');
  syncPaintTools();
}

function format(x) {
  if (!Number.isFinite(x)) return '—';
  return x === 0 ? '0' : (Math.abs(x) < .001 || Math.abs(x) >= 10000 ? x.toExponential(2) : Number(x.toPrecision(3)).toString());
}

function metrics(iteration, curvature, edge) {
  $('iteration').textContent = iteration;
  $('curvatureError').textContent = format(curvature);
  $('edgeError').textContent = format(edge);
}

function displayedFrame() {
  return state.frames[state.editing ? 0 : state.index];
}

function showLegendRange() {
  $('legendMin').textContent = viewer.limit ? '−' + format(viewer.limit) : '0';
  $('legendMax').textContent = format(viewer.limit);
}

function paintColors(field) {
  return state.colorMode === 'delta' ? field.values.map((target, i) => target - state.initial[i]) : field.values;
}

function showFrame(index, fit = false) {
  state.index = index;
  const frame = displayedFrame();
  if (!frame) return;
  $('starSummary').textContent = frame.starSummary || '';
  const {vertices, iteration, kind} = frame;
  const field = state.editing ? state.paint.field() : null;
  const values = field ? paintColors(field) : kind === 'initial' ? state.initial : angleDefects(vertices, state.mesh.faces, state.topology.boundary);
  viewer.setMesh(vertices, state.mesh.faces, values, fit);
  viewer.setPaintedVertices(state.editing ? state.paint.indices : null);
  $('frameTitle').textContent = state.editing ? 'Painted target · Input shape' : kind === 'initial' ? 'Input shape' : (kind === 'final' ? 'Final shape' : 'Preview') + ' · Iteration ' + iteration;
  showLegendRange();
  $('frameIndex').max = state.frames.length - 1;
  $('frameIndex').value = index;
  $('frameNumber').textContent = (index + 1) + ' / ' + state.frames.length;
  $('timeline').hidden = state.editing || state.frames.length < 2;
  $('plotRow').hidden = state.editing || state.history.length < 2;
  if (state.editing) {
    metrics(0); status('Editing target'); note();
    paintStatus(field);
    return;
  }
  metrics(iteration, frame.curvatureError, frame.edgeError);
  drawPlot();
  if (state.final) showOutcome();
}

function showOutcome() {
  const result = state.final;
  if (!result) return;
  const frame = state.frames[state.index], historical = frame.kind !== 'final';
  const end = result.converged ? 'Converged' : result.reason === 'line-search' ? 'Line search stopped' : 'Iteration limit reached';
  status(historical ? 'History · Iteration ' + frame.iteration : end, historical ? '' : result.converged ? 'done' : 'warning');
  if (historical) {
    note(`The values and chart marker show iteration ${frame.iteration}. The solver finished at iteration ${result.iteration} with a curvature RMS error of ${format(result.curvatureError)} rad (${end.toLowerCase()}).`);
  } else {
    note(result.converged ? 'The intrinsic curvature error reached the selected tolerance.' : 'The solver reconstructed the shape from its last accepted metric. The curvature error shows the remaining difference from the target.');
  }
}

function resetResults(fit = false) {
  state.final = null;
  state.frames = [{vertices: state.mesh.vertices, iteration: 0, kind: 'initial'}];
  state.history = [];
  status('Ready'); note(); error();
  showFrame(0, fit);
  updateEditButton();
}

function updateEditButton() {
  $('editPaint').hidden = $('target').value !== 'paint' || state.busy || (state.editing && state.frames.length < 2);
  $('editPaint').textContent = state.editing ? 'View last shape' : 'Edit painted target';
}

function paintStatus(field) {
  const {count, canUndo} = state.paint;
  const countText = count + (count === 1 ? ' vertex painted' : ' vertices painted');
  $('paintStatus').textContent = field.error ? countText + '. ' + field.error : countText;
  $('paintStatus').dataset.state = field.error ? 'warning' : '';
  $('undoPaint').disabled = !canUndo;
  $('clearPaint').disabled = count === 0;
  $('run').disabled = state.busy || Boolean(field.error);
}

let paintUpdate = 0;
function queuePaintUpdate() {
  if (paintUpdate) return;
  paintUpdate = requestAnimationFrame(() => {
    paintUpdate = 0;
    if (!state.editing || state.busy) return;
    const field = state.paint.field();
    viewer.setValues(paintColors(field));
    viewer.setPaintedVertices(state.paint.indices);
    showLegendRange();
    paintStatus(field);
  });
}

function finishEdit(changed) {
  if (changed && (state.history.length || state.frames.length > 1)) resetResults();
  else queuePaintUpdate();
}

const brushTool = {
  begin() { state.paint.begin(); },
  dab(vertex) {
    const erase = state.brush === 'Erase';
    const value = $('brushValue').valueAsNumber;
    if (!erase && !Number.isFinite(value)) {
      $('brushValue').reportValidity();
      return;
    }
    if (state.paint.dab(vertex, Number($('brushRadius').value), erase ? null : value)) queuePaintUpdate();
  },
  end(cancelled) { finishEdit(state.paint.end(cancelled) && !cancelled); }
};

function syncPaintTools() {
  const active = state.editing && !state.busy;
  $('paintControls').hidden = !active;
  updateEditButton();
  for (const tool of ['Paint', 'Erase', 'Orbit']) {
    $('brush' + tool).setAttribute('aria-pressed', state.brush === tool);
  }
  $('brushValue').disabled = state.brush !== 'Paint';
  $('brushRadius').disabled = state.brush === 'Orbit';
  const drawing = active && state.brush !== 'Orbit';
  viewer.setPaintTool(drawing ? brushTool : null);
  $('view').setAttribute('aria-label', drawing
    ? '3D target editor. Drag to ' + state.brush.toLowerCase() + ' curvature. Shift-drag or use arrow keys to rotate. Scroll or pinch to zoom.'
    : '3D mesh. Drag or use the arrow keys to rotate. Scroll, pinch, or use the plus and minus keys to zoom.');
  if (active) paintStatus(state.paint.field());
  else if (!state.busy) $('run').disabled = false;
}

function setEditing(editing) {
  if (editing !== state.editing) {
    if (editing) state.savedMode = viewer.mode;
    state.editing = editing;
    setMode(editing ? 'curvature' : state.savedMode);
  }
  $('colorMode').textContent = editing ? 'Target' : 'Curvature';
  syncPaintTools();
}

function setTargetMode() {
  const painting = $('target').value === 'paint';
  $('customTarget').hidden = $('target').value !== 'custom';
  const strength = $('strength');
  if (painting && !strength.disabled) {
    state.savedStrength = strength.value;
    strength.value = 100;
  } else if (!painting && strength.disabled) {
    strength.value = state.savedStrength;
  }
  strength.disabled = painting;
  strength.oninput();
  setEditing(painting);
}

for (const tool of ['Paint', 'Erase', 'Orbit']) {
  $('brush' + tool).onclick = () => { state.brush = tool; syncPaintTools(); };
}
$('undoPaint').onclick = () => finishEdit(state.paint.undo());
$('clearPaint').onclick = () => finishEdit(state.paint.clear());
function revealEditor() {
  if (state.editing && window.matchMedia('(max-width: 720px)').matches) {
    $('viewportTitle').closest('.viewport-panel').scrollIntoView({block: 'start'});
  }
}

$('editPaint').onclick = () => {
  setEditing(!state.editing);
  error();
  showFrame(state.index);
  if (!state.editing && !state.final) status('Stopped', 'warning');
  revealEditor();
};

$('paintPreset').onchange = () => {
  const preset = PAINT_PRESETS.find(p => p.id === $('paintPreset').value);
  $('patternDescription').textContent = preset?.description ?? '';
  $('patternDescription').hidden = !preset;
  $('applyPaintPreset').disabled = !preset;
};
$('applyPaintPreset').onclick = () => {
  const id = $('paintPreset').value;
  if (!id || state.busy) return;
  try {
    const changed = applyPaintPreset(id, state.mesh, state.paint);
    const wasPaint = $('target').value === 'paint';
    $('target').value = 'paint';
    setTargetMode();
    setMode('delta');
    error();
    if (changed || !wasPaint) resetResults();
    else showFrame(state.index);
    revealEditor();
  } catch (e) { error(e.message); }
};

function installMesh(mesh, name, sample, description = '') {
  const structure = topology(mesh), initial = angleDefects(mesh.vertices, mesh.faces, structure.boundary);
  Object.assign(state, {mesh, topology: structure, initial, name, sample, customTarget: null,
    paint: new CurvaturePaint(mesh, structure, initial)});
  $('meshName').textContent = name;
  $('shapeDescription').textContent = description;
  $('shapeDescription').hidden = !description;
  $('vertexCount').textContent = mesh.vertices.length.toLocaleString('en-US');
  $('faceCount').textContent = mesh.faces.length.toLocaleString('en-US');
  $('chi').textContent = structure.chi;
  if ($('target').value === 'custom') $('target').value = 'uniform';
  setTargetMode();
  if (state.editing) setMode('curvature');
  $('targetFilename').textContent = 'Choose CSV';
  resetResults(true);
}

async function loadSample(sample) {
  busy(true);
  try {
    if (sample === 'custom') { installMesh(state.imported.mesh, state.imported.name, sample); return; }
    const preset = samples.find(entry => entry.id === sample);
    let mesh;
    if (preset.create) mesh = preset.create();
    else {
      const response = await fetch(new URL('./meshes/' + preset.file, import.meta.url));
      if (!response.ok) throw Error(`Could not load the sample mesh (HTTP ${response.status}).`);
      mesh = parseMesh(await response.arrayBuffer(), preset.file);
    }
    installMesh(mesh, preset.label, sample, preset.description);
  } catch (e) { $('sample').value = state.sample; error(e.message); }
  finally { busy(false); }
}

$('sample').onchange = () => loadSample($('sample').value);
$('importMesh').onclick = () => $('meshFile').click();
$('meshFile').onchange = async () => {
  const file = $('meshFile').files[0];
  if (!file) return;
  busy(true);
  try {
    const mesh = parseMesh(await file.arrayBuffer(), file.name);
    installMesh(mesh, file.name, 'custom');
    state.imported = {mesh, name: file.name};
    let option = $('sample').querySelector('[value=custom]');
    if (!option) { option = document.createElement('option'); option.value = 'custom'; $('sample').append(option); }
    option.textContent = file.name; $('sample').value = 'custom';
  } catch (e) { error(e.message); }
  finally { busy(false); $('meshFile').value = ''; }
};
$('target').onchange = () => { setTargetMode(); resetResults(); revealEditor(); };
$('importTarget').onclick = () => $('targetFile').click();
$('targetFile').onchange = async () => {
  const file = $('targetFile').files[0];
  if (!file) return;
  busy(true);
  try {
    const target = parseTarget(await file.text(), state.initial, state.topology.adjacent);
    state.customTarget = target;
    $('targetFilename').textContent = file.name;
    resetResults();
  } catch (e) { error(e.message); }
  finally { busy(false); $('targetFile').value = ''; }
};
$('strength').oninput = () => {
  $('strengthValue').textContent = $('strength').value + '%';
  $('strength').style.setProperty('--strength', $('strength').value + '%');
};
$('strength').onchange = () => resetResults();
$('preview').onchange = () => { $('intervalRow').hidden = !$('preview').checked; };
$('bendingWeight').oninput = () => { $('bendingTarget').disabled = !($('bendingWeight').valueAsNumber > 0); };
$('bendingWeight').onchange = $('bendingTarget').onchange = () => resetResults();
$('frameIndex').oninput = () => showFrame(Number($('frameIndex').value));

$('checkStars').onclick = async () => {
  const frame = displayedFrame();
  const faces = state.mesh.faces;
  $('checkStars').disabled = true;
  try {
    const {diagnoseVertexStars} = await import('./immersion.mjs');
    const result = diagnoseVertexStars(frame.vertices, faces);
    const classified = result.classifiedInteriorVertices;
    const boundary = result.boundaryVertices.length;
    frame.starSummary = `${classified} interior ${classified === 1 ? 'star' : 'stars'} classified (${result.pinchVertices.length} pinched), ${result.indeterminateVertices.length} indeterminate. `
      + `${boundary} boundary ${boundary === 1 ? 'vertex' : 'vertices'} skipped. This local check excludes global self-intersections.`;
  } catch (err) {
    frame.starSummary = 'Vertex star check failed. ' + err.message;
  } finally {
    $('checkStars').disabled = false;
    if (displayedFrame() === frame) $('starSummary').textContent = frame.starSummary;
  }
};

function setMode(mode) {
  state.colorMode = mode;
  viewer.mode = mode === 'plain' ? 'plain' : 'curvature';
  $('colorMode').setAttribute('aria-pressed', mode === 'curvature');
  $('deltaMode').setAttribute('aria-pressed', mode === 'delta');
  $('plainMode').setAttribute('aria-pressed', mode === 'plain');
  $('deltaMode').hidden = !state.editing;
  $('legend').hidden = mode === 'plain';
  $('legendTitle').textContent = mode === 'delta' ? 'Target − input' : state.editing ? 'Target angle defect' : 'Angle defect';
  if (state.editing) {
    viewer.setValues(paintColors(state.paint.field()));
    showLegendRange();
  }
  viewer.redraw();
}
$('colorMode').onclick = () => setMode('curvature');
$('deltaMode').onclick = () => setMode('delta');
$('plainMode').onclick = () => setMode('plain');
$('wire').onclick = () => {
  viewer.wire = !viewer.wire; $('wire').setAttribute('aria-pressed', viewer.wire); viewer.redraw();
};
$('fit').onclick = () => viewer.reset();
$('zoomIn').onclick = () => viewer.zoomBy(1.2);
$('zoomOut').onclick = () => viewer.zoomBy(1 / 1.2);
$('helpOpen').onclick = () => $('help').showModal();
$('helpClose').onclick = () => $('help').close();
$('help').addEventListener('click', e => { if (e.target === $('help') && !e.target.matches(':hover')) $('help').close(); });
$('export').onclick = () => {
  const f = displayedFrame();
  const blob = new Blob([encodePly(f.vertices, state.mesh.faces)], {type: 'text/plain;charset=utf-8'});
  const url = URL.createObjectURL(blob), link = document.createElement('a');
  link.href = url;
  link.download = state.name.replace(/\.(obj|ply)$/i, '') + '-' + f.kind + '-' + f.iteration + '.ply';
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};

function drawPlot() {
  const el = $('plot'), samples = state.history;
  if (samples.length < 2 || $('plotRow').hidden) return;
  const c = el.getContext('2d'), w = el.clientWidth, h = el.clientHeight;
  if (!w || !h) return;
  const px = Math.min(2, devicePixelRatio || 1);
  el.width = w*px; el.height = h*px; c.scale(px, px);
  const logs = samples.map(v => Math.log10(Math.max(1e-16, v)));
  let lo = Math.floor(Math.min(...logs)), hi = Math.ceil(Math.max(...logs));
  if (lo === hi) hi++;
  const left = 45, right = w-12, bottom = h-21, top = 11;
  const x = i => left + i/(logs.length-1)*(right-left);
  const y = v => bottom - (v-lo)/(hi-lo)*(bottom-top);
  c.font = '11px Consolas, monospace'; c.fillStyle = '#516971';
  for (const log of [hi, lo]) {
    c.strokeStyle = '#d8e3e1'; c.beginPath(); c.moveTo(left, y(log)); c.lineTo(right, y(log)); c.stroke();
    c.fillText('1e' + log, 1, y(log)+3);
  }
  c.fillText('0', left, h-4); c.textAlign = 'right'; c.fillText('Iteration ' + (logs.length-1), right, h-4);
  c.beginPath(); logs.forEach((log, i) => i ? c.lineTo(x(i),y(log)) : c.moveTo(x(i),y(log)));
  c.strokeStyle = '#008476'; c.lineWidth = 2; c.stroke();
  const i = Math.min(logs.length-1, state.frames[state.index]?.iteration ?? 0);
  c.beginPath(); c.arc(x(i), y(logs[i]), 3.5, 0, Math.PI*2); c.fillStyle = '#007267'; c.fill();
}
new ResizeObserver(drawPlot).observe($('plot'));

function stopWorker() {
  state.worker?.terminate();
  state.worker = null;
  busy(false);
}

function failed(message) {
  stopWorker();
  status('Error', 'warning'); note(); error(message);
}

function receive(message) {
  if (message.type === 'error') { failed(message.message); return; }
  const {iteration, curvatureError, edgeError, type, phase} = message;
  if (iteration === 0) state.frames[0].curvatureError = curvatureError;
  state.history[iteration] = curvatureError;
  if (state.history.length > 1) $('plotRow').hidden = false;
  if (type === 'preview' || type === 'done') {
    state.frames.push({vertices: message.vertices, iteration, curvatureError, edgeError, kind: type === 'done' ? 'final' : 'preview'});
    showFrame(state.frames.length - 1);
  } else {
    metrics(iteration, curvatureError, null);
    drawPlot();
  }
  if (type === 'done') {
    stopWorker();
    state.final = {iteration, curvatureError, reason: message.reason, converged: message.converged};
    showOutcome();
  } else {
    status(phase === 'shape' ? 'Reconstructing shape' : 'Optimizing metric', 'running');
  }
}

$('run').onclick = () => {
  if (state.worker) {
    stopWorker(); status('Stopped', 'warning'); note('Optimization stopped. You can still inspect the available shapes.'); return;
  }
  error();
  const target = $('target').value;
  if (target === 'custom' && !state.customTarget) { error('Choose a CSV file with the target curvatures.'); return; }
  const painted = target === 'paint' ? state.paint.field() : null;
  if (painted?.error) { error(painted.error); return; }
  const inputs = [
    [$('maxIter'), 'Enter an iteration limit from 1 to 500 in whole steps.'],
    [$('bendingWeight'), 'Enter a finite bending weight of zero or greater.']
  ];
  if ($('preview').checked) inputs.push([$('interval'), 'Enter a preview interval from 1 to 500 in whole steps.']);
  for (const [input, message] of inputs) {
    if (!input.checkValidity()) {
      if (input.closest('details')) $('advanced').open = true;
      input.focus();
      error(message);
      return;
    }
  }
  const options = {target, strength: Number($('strength').value)/100, maxIterations: Number($('maxIter').value),
    tolerance: Number($('tolerance').value), previewEvery: $('preview').checked ? Number($('interval').value) : 0,
    bendingWeight: $('bendingWeight').valueAsNumber, bendingTarget: $('bendingTarget').value};
  if (target === 'custom') options.targetCurvature = state.customTarget;
  if (painted) options.targetCurvature = painted.values;
  try {
    const worker = new Worker(new URL('./solver-worker.mjs', import.meta.url), {type:'module'});
    // A second touch can start the solver before the brush pointer lifts.
    if (painted) state.paint.end();
    setEditing(false);
    resetResults();
    state.worker = worker;
    busy(true, true); status('Optimizing metric', 'running');
    worker.onmessage = e => {
      if (worker !== state.worker) return;
      try { receive(e.data); } catch (err) { failed(err.message); }
    };
    worker.onerror = e => { if (worker === state.worker) failed(e.message || 'The solver worker could not be loaded.'); };
    worker.postMessage({type:'run', vertices:state.mesh.vertices, faces:state.mesh.faces, options});
  } catch (e) { failed(e.message); }
};

loadSample('torus');
