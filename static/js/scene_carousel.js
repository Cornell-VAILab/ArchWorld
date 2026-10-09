// ArchWorld scene carousel: metadata, photos, and an interactive point cloud
// viewer with metric distance measurement (Shift+click two points).
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const ROOT_ID = 'scene-carousel';
const DATA_URL = 'static/scenes/scenes.json';
const CLICK_SLOP_PX = 5;

// Region keys as stored in scenes.json, with display labels, in taxonomy order.
const REGION_ORDER = [
  ['Africa', 'Africa'],
  ['East Asia', 'East Asia'],
  ['Europe', 'Europe'],
  ['Latin America', 'Latin America'],
  ['North America', 'North America'],
  ['South Asia', 'South Asia'],
  ['SE Asia', 'Southeast Asia'],
  ['West/Central Asia', 'West/Central Asia'],
];

const META_FIELDS = [
  ['Landmark Name', (s) => s.name],
  ['Country', (s) => s.country],
  ['Region', (s) => s.region],
  ['WikiData Architecture Type <span class="aw-meta-sub">(instanceof)</span>', (s) => s.wikidata_type],
  ['Coarse Architecture Type', (s) => s.coarse_type],
  ['Construction Year', (s) => s.year],
  ['Scene Extent', (s, v) => `${s.versions[v].extent_m.toFixed(2)} m`],
];

function el(tag, cls, html) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html !== undefined) e.innerHTML = html;
  return e;
}

// ---------------------------------------------------------------------------
// Point cloud viewer
// ---------------------------------------------------------------------------
class PointViewer {
  constructor(container, readout) {
    this.container = container;
    this.readout = readout;
    this.cache = new Map();          // glb url -> THREE.Points
    this.current = null;             // { scene meta, points }
    this.measurements = [];
    this.pending = null;             // first point of an in-progress measurement
    this.measureMode = false;        // touch-friendly alternative to Shift
    this.visible = true;
    this.loadToken = 0;
    this.showBox = false;           // bounding box + extent diagonal overlay
    this.box = null;                // { objects, label, a, b }

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xf5f5f4);
    this.camera = new THREE.PerspectiveCamera(50, 1, 0.001, 1000);
    this._makeControls();

    this.raycaster = new THREE.Raycaster();
    this.overlay = el('div', 'aw-viewer-labels');
    container.appendChild(this.overlay);
    this.status = el('div', 'aw-viewer-status');
    container.appendChild(this.status);

    this._bindPointer();
    new ResizeObserver(() => this._resize()).observe(container);
    new IntersectionObserver(([e]) => { this.visible = e.isIntersecting; }).observe(container);
    this._resize();
    this.renderer.setAnimationLoop(() => this._frame());
  }

  // OrbitControls caches camera.up when constructed, so rebuild it whenever
  // a scene sets a different up vector.
  _makeControls() {
    if (this.controls) this.controls.dispose();
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.12;
    this.controls.zoomSpeed = 0.35;
  }

  _resize() {
    const w = this.container.clientWidth, h = this.container.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  _frame() {
    if (!this.visible) return;
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    this._updateLabels();
  }

  setStatus(text) {
    this.status.textContent = text || '';
    this.status.style.display = text ? 'flex' : 'none';
  }

  // Show `version` ('landmark' | 'full') of a scene. Switching versions of the
  // same scene keeps the camera and measurements, since both clouds share one
  // coordinate frame; switching scenes resets them.
  async show(meta, version) {
    const token = ++this.loadToken;
    const sameScene = this.current && this.current.meta === meta;
    if (!sameScene) this.clearMeasurements();
    if (this.current) this.scene.remove(this.current.points);
    this.current = null;

    const url = meta.versions[version].glb;
    let points = this.cache.get(url);
    if (!points) {
      this.setStatus('Loading point cloud…');
      try {
        points = await this._load(meta, url, (pct) => {
          if (token === this.loadToken) this.setStatus(`Loading point cloud… ${pct}%`);
        });
      } catch (err) {
        console.error(err);
        if (token === this.loadToken) this.setStatus('Could not load the point cloud.');
        return;
      }
      this.cache.set(url, points);
    }
    if (token !== this.loadToken) return; // user moved on meanwhile

    this.setStatus('');
    this.scene.add(points);
    this.current = { meta, points, version };
    this._updateBox();
    const v = meta.view;
    // Pick radius follows point spacing, so it stays tight however far the camera starts.
    this.pickThreshold = v.point_size ? Math.max(v.point_size * 1.5, v.dist * 0.002) : v.dist * 0.006;
    this.markerRadius = v.dist * 0.005;
    if (!sameScene) this.resetView();
  }

  _load(meta, url, onPct) {
    return new Promise((resolve, reject) => {
      new GLTFLoader().load(url, (gltf) => {
        let pts = null;
        gltf.scene.traverse((o) => { if (o.isPoints && !pts) pts = o; });
        if (!pts) return reject(new Error('No point primitive in ' + url));
        pts.material = new THREE.PointsMaterial({
          size: meta.view.point_size ?? meta.view.dist * 0.0028,
          vertexColors: !!pts.geometry.getAttribute('color'),
          sizeAttenuation: true,
        });
        // Keep the node's transform so world coordinates match the file;
        // measurements are converted back to local units in _pick().
        gltf.scene.updateMatrixWorld(true);
        const group = new THREE.Group();
        group.add(gltf.scene);
        group.userData.points = pts;
        resolve(group);
      }, (xhr) => {
        if (xhr.total) onPct(Math.round((xhr.loaded / xhr.total) * 100));
      }, reject);
    });
  }

  resetView() {
    if (!this.current) return;
    const { center, up, dir, dist } = this.current.meta.view;
    const c = new THREE.Vector3(...center);
    const u = new THREE.Vector3(...up).normalize();
    const d = new THREE.Vector3(...dir).normalize();
    this.camera.up.copy(u);
    this._makeControls();
    this.controls.minDistance = dist * 0.02;
    this.controls.maxDistance = dist * 10;
    this.camera.position.copy(c).addScaledVector(d, dist).addScaledVector(u, dist * 0.2);
    this.camera.near = dist / 500;
    this.camera.far = dist * 100;
    this.camera.updateProjectionMatrix();
    this.controls.target.copy(c);
    this.controls.update();
  }

  // ---- picking & measurement ------------------------------------------------
  _bindPointer() {
    const dom = this.renderer.domElement;
    let down = null;
    dom.addEventListener('pointerdown', (e) => { down = { x: e.clientX, y: e.clientY }; });
    dom.addEventListener('pointerup', (e) => {
      if (!down) return;
      const moved = Math.hypot(e.clientX - down.x, e.clientY - down.y);
      down = null;
      if (moved > CLICK_SLOP_PX) return;
      if (!(e.shiftKey || this.measureMode)) return;
      const hit = this._pick(e);
      if (hit) this._addPoint(hit);
    });
  }

  _pick(e) {
    if (!this.current) return null;
    const pts = this.current.points.userData.points;
    const rect = this.renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((e.clientX - rect.left) / rect.width) * 2 - 1,
      -((e.clientY - rect.top) / rect.height) * 2 + 1,
    );
    this.raycaster.setFromCamera(ndc, this.camera);
    this.raycaster.params.Points.threshold = this.pickThreshold;
    const hits = this.raycaster.intersectObject(pts, false);
    if (!hits.length) return null;
    // Among hits near the front surface, take the one closest to the ray.
    const front = hits[0].distance + this.pickThreshold * 4;
    let best = hits[0];
    for (const h of hits) {
      if (h.distance > front) break;
      if (h.distanceToRay < best.distanceToRay) best = h;
    }
    const pos = pts.geometry.getAttribute('position');
    const world = new THREE.Vector3().fromBufferAttribute(pos, best.index).applyMatrix4(pts.matrixWorld);
    const local = new THREE.Vector3().fromBufferAttribute(pos, best.index);
    return { world, local };
  }

  _marker(p) {
    const m = new THREE.Mesh(
      new THREE.SphereGeometry(this.markerRadius, 16, 12),
      new THREE.MeshBasicMaterial({ color: 0xdc2626, depthTest: false }),
    );
    m.position.copy(p);
    m.renderOrder = 10;
    this.scene.add(m);
    return m;
  }

  _addPoint(hit) {
    if (!this.pending) {
      this.pending = { hit, marker: this._marker(hit.world) };
      this._renderReadout();
      return;
    }
    const a = this.pending.hit, b = hit;
    const line = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([a.world, b.world]),
      new THREE.LineBasicMaterial({ color: 0xdc2626, depthTest: false }),
    );
    line.renderOrder = 9;
    this.scene.add(line);
    const meters = a.local.distanceTo(b.local) * this.current.meta.scale;
    const label = el('div', 'aw-measure-label', `${meters.toFixed(2)} m`);
    this.overlay.appendChild(label);
    this.measurements.push({
      a: a.world, b: b.world, meters, label,
      objects: [this.pending.marker, this._marker(b.world), line],
    });
    this.pending = null;
    this._renderReadout();
  }

  _updateLabels() {
    const items = this.box ? [...this.measurements, this.box] : this.measurements;
    if (!items.length) return;
    const w = this.container.clientWidth, h = this.container.clientHeight;
    const v = new THREE.Vector3();
    for (const m of items) {
      v.copy(m.a).add(m.b).multiplyScalar(0.5).project(this.camera);
      const behind = v.z > 1;
      m.label.style.display = behind ? 'none' : 'block';
      m.label.style.transform = `translate(${(v.x * 0.5 + 0.5) * w}px, ${(-v.y * 0.5 + 0.5) * h}px) translate(-50%, -130%)`;
    }
  }

  // ---- bounding box showing the scene extent ----------------------------------
  setShowBox(on) {
    this.showBox = on;
    this._updateBox();
  }

  _updateBox() {
    if (this.box) {
      this.box.objects.forEach((o) => { this.scene.remove(o); o.geometry.dispose(); o.material.dispose(); });
      this.box.label.remove();
      this.box = null;
    }
    if (!this.showBox || !this.current) return;
    const { meta, points, version } = this.current;
    const info = meta.versions[version];
    // Prefer the bbox of the original (pre-downsampling) cloud, which is what
    // the extent was computed from; fall back to the loaded cloud's bbox.
    let min, max;
    if (info.bbox) {
      min = new THREE.Vector3(...info.bbox.min);
      max = new THREE.Vector3(...info.bbox.max);
    } else {
      const geo = points.userData.points.geometry;
      geo.computeBoundingBox();
      min = geo.boundingBox.min.clone();
      max = geo.boundingBox.max.clone();
    }
    const box = new THREE.Box3(min, max);
    const edges = new THREE.Box3Helper(box, 0x2563eb);
    edges.material.depthTest = false;
    edges.renderOrder = 8;
    const diag = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([min, max]),
      new THREE.LineDashedMaterial({ color: 0x2563eb, dashSize: meta.view.dist * 0.03, gapSize: meta.view.dist * 0.015, depthTest: false }),
    );
    diag.computeLineDistances();
    diag.renderOrder = 8;
    this.scene.add(edges, diag);
    const meters = info.extent_m ?? min.distanceTo(max) * meta.scale;
    const label = el('div', 'aw-measure-label aw-extent-label', `Extent: ${meters.toFixed(2)} m`);
    this.overlay.appendChild(label);
    this.box = { objects: [edges, diag], label, a: min, b: max };
  }

  clearMeasurements() {
    for (const m of this.measurements) {
      m.objects.forEach((o) => { this.scene.remove(o); o.geometry.dispose(); o.material.dispose(); });
      m.label.remove();
    }
    if (this.pending) {
      this.scene.remove(this.pending.marker);
      this.pending.marker.geometry.dispose();
      this.pending.marker.material.dispose();
    }
    this.measurements = [];
    this.pending = null;
    this._renderReadout();
  }

  _renderReadout() {
    if (!this.readout) return;
    if (!this.measurements.length && !this.pending) {
      this.readout.innerHTML = '<span class="aw-muted">No measurements yet.</span>';
      return;
    }
    const items = this.measurements.map((m, i) => `<li><span>#${i + 1}</span><strong>${m.meters.toFixed(2)} m</strong></li>`);
    if (this.pending) items.push('<li class="aw-muted">Pick a second point…</li>');
    this.readout.innerHTML = `<ol>${items.join('')}</ol>`;
  }
}

// ---------------------------------------------------------------------------
// Carousel UI
// ---------------------------------------------------------------------------
function buildLightbox() {
  const box = el('div', 'aw-lightbox');
  const img = el('img');
  box.appendChild(img);
  box.addEventListener('click', () => box.classList.remove('is-open'));
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') box.classList.remove('is-open'); });
  document.body.appendChild(box);
  return (src, alt) => { img.src = src; img.alt = alt; box.classList.add('is-open'); };
}

function init(root, scenes) {
  root.innerHTML = '';
  const openLightbox = buildLightbox();

  // Group scenes by region, in taxonomy order; within a region, scenes.json order.
  const regions = REGION_ORDER
    .map(([key, label]) => ({ key, label, scenes: scenes.filter((s) => s.region === key) }))
    .filter((r) => r.scenes.length);

  // Region slider: one clickable segment per region, with a sliding highlight.
  const nav = el('div', 'aw-region-nav');
  const prev = el('button', 'button is-small is-rounded aw-region-arrow', '<span class="icon"><i class="fas fa-chevron-left"></i></span>');
  const next = el('button', 'button is-small is-rounded aw-region-arrow', '<span class="icon"><i class="fas fa-chevron-right"></i></span>');
  prev.setAttribute('aria-label', 'Previous region');
  next.setAttribute('aria-label', 'Next region');
  const track = el('div', 'aw-region-track');
  track.setAttribute('role', 'tablist');
  const indicator = el('span', 'aw-region-indicator');
  track.appendChild(indicator);
  const regionBtns = regions.map((r, i) => {
    const b = el('button', 'aw-region', r.label);
    b.setAttribute('role', 'tab');
    b.addEventListener('click', () => go(i));
    track.appendChild(b);
    return b;
  });
  nav.append(prev, track, next);

  const head = el('div', 'aw-scene-head');
  const title = el('div', 'aw-scene-name');
  const candidates = el('div', 'aw-candidates');
  head.append(title, candidates);

  const grid = el('div', 'aw-scene-grid');
  const viewerCol = el('div', 'aw-viewer-col');
  const viewerBox = el('div', 'aw-viewer');
  const toolbar = el('div', 'aw-viewer-toolbar');
  // Slider switch: left = before segmentation (full cloud), right = landmark only.
  const versionGroup = el('label', 'aw-switch',
    '<span class="aw-switch-label" data-side="full">Before Segmentation</span>' +
    '<input type="checkbox" role="switch" checked aria-label="Show landmark only">' +
    '<span class="aw-switch-track"><span class="aw-switch-thumb"></span></span>' +
    '<span class="aw-switch-label" data-side="landmark">Landmark only</span>');
  const versionInput = versionGroup.querySelector('input');
  const measureBtn = el('button', 'button is-small', '<span class="icon"><i class="fas fa-ruler"></i></span><span>Measure</span>');
  const clearBtn = el('button', 'button is-small', '<span>Clear</span>');
  const resetBtn = el('button', 'button is-small', '<span class="icon"><i class="fas fa-undo"></i></span><span>Reset view</span>');
  const boxBtn = el('button', 'button is-small aw-box-btn', '<span class="icon"><i class="fas fa-cube"></i></span><span>Extent box</span>');
  toolbar.append(versionGroup, boxBtn, measureBtn, clearBtn, resetBtn);
  const hint = el('p', 'aw-viewer-hint',
    '<strong>Shift+click</strong> two points to measure the distance between them in meters. On touch screens, turn on <em>Measure</em> and tap.');
  viewerCol.append(viewerBox, toolbar, hint);

  const metaCol = el('div', 'aw-meta-col');
  const metaTable = el('table', 'aw-meta');
  const readoutWrap = el('div', 'aw-readout');
  readoutWrap.append(el('div', 'aw-readout-title', 'Measurements'));
  const readout = el('div', 'aw-readout-body');
  readoutWrap.append(readout);
  metaCol.append(metaTable, readoutWrap);
  grid.append(viewerCol, metaCol);

  const thumbs = el('div', 'aw-thumbs');
  root.append(nav, head, grid, thumbs);

  let viewer = null;
  try {
    viewer = new PointViewer(viewerBox, readout);
  } catch (err) {
    console.error(err);
    viewerBox.append(el('div', 'aw-viewer-status', 'WebGL is not available in this browser.'));
  }

  measureBtn.addEventListener('click', () => {
    if (!viewer) return;
    viewer.measureMode = !viewer.measureMode;
    measureBtn.classList.toggle('is-active', viewer.measureMode);
  });
  clearBtn.addEventListener('click', () => viewer && viewer.clearMeasurements());
  resetBtn.addEventListener('click', () => viewer && viewer.resetView());
  boxBtn.addEventListener('click', () => {
    if (!viewer) return;
    viewer.setShowBox(!viewer.showBox);
    boxBtn.classList.toggle('is-active', viewer.showBox);
  });

  let regionIdx = 0;
  const candidateIdx = regions.map(() => 0);   // remembered per region
  const currentScene = () => regions[regionIdx].scenes[candidateIdx[regionIdx]];
  let version = 'landmark';
  function renderMeta() {
    const s = currentScene();
    metaTable.innerHTML = META_FIELDS.map(([k, f]) => `<tr><th>${k}</th><td>${f(s, version)}</td></tr>`).join('');
  }
  function syncSwitch() {
    versionInput.checked = version === 'landmark';
    versionGroup.querySelectorAll('.aw-switch-label').forEach((l) => l.classList.toggle('is-active', l.dataset.side === version));
  }
  function setVersion(v) {
    version = v;
    syncSwitch();
    renderMeta();
    if (viewer) viewer.show(currentScene(), version);
  }
  versionInput.addEventListener('change', () => setVersion(versionInput.checked ? 'landmark' : 'full'));
  syncSwitch();

  function moveIndicator() {
    const b = regionBtns[regionIdx];
    indicator.style.width = `${b.offsetWidth}px`;
    indicator.style.transform = `translateX(${b.offsetLeft}px)`;
    b.scrollIntoView({ block: 'nearest', inline: 'nearest' });
  }
  new ResizeObserver(moveIndicator).observe(track);

  function go(r, c) {
    regionIdx = (r + regions.length) % regions.length;
    if (c !== undefined) candidateIdx[regionIdx] = c;
    const region = regions[regionIdx];
    const s = currentScene();
    regionBtns.forEach((b, k) => {
      b.classList.toggle('is-active', k === regionIdx);
      b.setAttribute('aria-selected', k === regionIdx);
    });
    moveIndicator();
    title.textContent = s.name;
    // While several candidates exist for a region, let people switch between them.
    candidates.innerHTML = '';
    if (region.scenes.length > 1) {
      candidates.append(el('span', 'aw-candidates-label', 'Candidates:'));
      region.scenes.forEach((cand, k) => {
        const chip = el('button', 'aw-chip' + (k === candidateIdx[regionIdx] ? ' is-active' : ''), cand.name);
        chip.addEventListener('click', () => go(regionIdx, k));
        candidates.append(chip);
      });
    }
    renderMeta();
    thumbs.innerHTML = '';
    s.images.forEach((im, k) => {
      const t = el('button', 'aw-thumb');
      const img = el('img');
      img.src = im.src;
      img.loading = 'lazy';
      img.alt = `${s.name}, photo ${k + 1}`;
      t.appendChild(img);
      t.addEventListener('click', () => openLightbox(im.src, img.alt));
      thumbs.appendChild(t);
    });
    if (viewer) viewer.show(s, version);
  }
  prev.addEventListener('click', () => go(regionIdx - 1));
  next.addEventListener('click', () => go(regionIdx + 1));
  go(0);
}

async function main() {
  const root = document.getElementById(ROOT_ID);
  if (!root) return;
  root.innerHTML = '<div class="aw-viewer-status" style="position:static">Loading scenes…</div>';
  let data;
  try {
    data = await (await fetch(DATA_URL)).json();
  } catch (err) {
    console.error(err);
    root.innerHTML = '<div class="aw-viewer-status" style="position:static">Could not load scene data.</div>';
    return;
  }
  // Don't download the point clouds until the carousel is near the viewport.
  const io = new IntersectionObserver((entries) => {
    if (entries.some((e) => e.isIntersecting)) {
      io.disconnect();
      init(root, data.scenes);
    }
  }, { rootMargin: '300px' });
  io.observe(root);
}

main();
