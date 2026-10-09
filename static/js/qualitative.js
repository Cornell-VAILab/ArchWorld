// Qualitative results: looping videos of the GT, the ICP-aligned pi3 prediction, and its
// normalized-CD error overlay. Data and videos come from static/qualitative/ (built by
// build_qualitative_spins.py): one scene per (region, building type) median of pi3's
// normalized CD.
// The median scatter drives the video: it auto-advances through the medians (a ring on the
// active median counts down), and hovering a median holds its scene until the mouse leaves.
(function () {
  const REGION_LABEL = {
    'Africa': 'Africa', 'East Asia': 'E Asia', 'Europe': 'Europe', 'Latin America': 'L Amer',
    'North America': 'N Amer', 'SE Asia': 'SE Asia', 'South Asia': 'S Asia', 'West/Central Asia': 'WC Asia',
  };
  // Same colors as the paper's scatter (plot_medians.py).
  const REGION_COLOR = {
    'Africa': '#E69F00', 'East Asia': '#56B4E9', 'Europe': '#009E73', 'Latin America': '#F0E442',
    'North America': '#0072B2', 'SE Asia': '#D55E00', 'South Asia': '#CC79A7', 'West/Central Asia': '#8172B3',
  };
  const TYPE_LABEL = { 'Mosque': 'Mosque', 'Church/Cathedral': 'Church', 'Temple/Shrine': 'Temple/Shrine' };
  const CDHAT = '\\(\\widehat{\\mathrm{CD}}\\)';
  const CYCLE_MS = 7000;   // one sway cycle of the videos at 1x
  const SPEEDS = [1, 1.5, 2];
  const RING_R = 15;
  const SVGNS = 'http://www.w3.org/2000/svg';

  function star(r) {
    const pts = [];
    for (let i = 0; i < 10; i++) {
      const a = -Math.PI / 2 + (i * Math.PI) / 5;
      const rr = i % 2 ? r * 0.45 : r;
      pts.push(`${(rr * Math.cos(a)).toFixed(2)},${(rr * Math.sin(a)).toFixed(2)}`);
    }
    return `M${pts.join('L')}Z`;
  }
  function plus(r) {
    const t = r * 0.38;
    return `M${-t},${-r}H${t}V${-t}H${r}V${t}H${t}V${r}H${-t}V${t}H${-r}V${-t}H${-t}Z`;
  }
  function circle(r) {
    return `M${-r},0A${r},${r} 0 1,0 ${r},0A${r},${r} 0 1,0 ${-r},0Z`;
  }
  const SHAPE = { 'Mosque': star(10), 'Church/Cathedral': plus(8), 'Temple/Shrine': circle(7.5) };
  const marker = (type, color) =>
    `<svg class="aw-mk" viewBox="-11 -11 22 22" aria-hidden="true"><path d="${SHAPE[type]}" fill="${color}"/></svg>`;

  function typeset(el) {
    if (window.renderMathInElement) {
      window.renderMathInElement(el, { delimiters: [{ left: '\\(', right: '\\)', display: false }] });
    }
  }

  function build(root, data) {
    const order = data.region_order.filter((r) => data.points.some((p) => p.region === r));
    // Visit the medians top to bottom, left to right.
    const pts = [...data.points].sort((a, b) =>
      order.indexOf(a.region) - order.indexOf(b.region) || a.median_cdnorm - b.median_cdnorm);
    const cb = data.error_colorbar;
    const rgb = (c) => `rgb(${c.join(',')})`;

    root.innerHTML = `
      <div class="aw-qual-video">
        <div class="aw-qual-track"></div>
        <span class="lbl" style="left:0.6rem">Ground truth</span>
        <span class="lbl" style="left:calc(100% / 3 + 0.6rem)">π<sup>3</sup> (aligned)</span>
        <span class="lbl" style="left:calc(200% / 3 + 0.6rem)">Error overlay</span>
        <div class="aw-qual-cbar">
          <div class="bar" style="background:linear-gradient(to right, ${rgb(cb.low_rgb)}, ${rgb(cb.high_rgb)})"></div>
          <div class="ticks"><span>${cb.lo.toFixed(2)}</span><span>${CDHAT}</span><span>${cb.hi.toFixed(1)}</span></div>
        </div>
      </div>
      <div class="aw-qual-meta"><div class="name"></div><div class="sub"></div></div>
      <div class="aw-qsc">
        <p class="aw-qsc-hint">Each point on the scatter plot below is the median ${CDHAT} of one region and architecture
          type. The scene above shows the ground truth and prediction for the selected point.</p>
        <svg role="img" aria-label="Median π3 normalized CD by region and building type"></svg>
        <div class="aw-qsc-xlabel">${CDHAT}</div>
        <div class="aw-qsc-foot">
          <div class="aw-qsc-legend"></div>
          <div class="aw-qsc-ctrl">
            <div class="aw-speed" role="group" aria-label="Playback speed">
              ${SPEEDS.map((s) => `<button type="button" data-speed="${s}"${s === 1 ? ' class="is-active"' : ''}>${s}×</button>`).join('')}
            </div>
            <label class="aw-qcar-auto"><input type="checkbox" checked> Auto-advance</label>
          </div>
        </div>
      </div>`;
    const tile = root.querySelector('.aw-qual-video');
    const track = tile.querySelector('.aw-qual-track');
    let video = null;
    const nameEl = root.querySelector('.aw-qual-meta .name');
    const subEl = root.querySelector('.aw-qual-meta .sub');
    const svg = root.querySelector('.aw-qsc svg');
    typeset(tile);
    typeset(root.querySelector('.aw-qsc-hint'));
    typeset(root.querySelector('.aw-qsc-xlabel'));

    // ---- scatter ----
    const W = 560, LABEL_W = 84, ROW_H = 40, TOP = 4, AXIS_H = 22;
    const H = TOP + order.length * ROW_H + AXIS_H;
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    const xMin = 1;
    const xMax = Math.ceil(Math.max(...pts.map((p) => p.median_cdnorm)) * 4 + 0.5) / 4;
    const x = (v) => LABEL_W + 22 + ((v - xMin) / (xMax - xMin)) * (W - LABEL_W - 44);
    const rowY = (r) => TOP + order.indexOf(r) * ROW_H + ROW_H / 2;
    const el = (tag, attrs, parent = svg) => {
      const n = document.createElementNS(SVGNS, tag);
      Object.entries(attrs).forEach(([k, v]) => n.setAttribute(k, v));
      parent.appendChild(n);
      return n;
    };
    const plotH = order.length * ROW_H;
    el('rect', { class: 'row-bg', x: 0, y: TOP, width: LABEL_W, height: plotH, rx: 4 });
    el('rect', { class: 'row-bg', x: LABEL_W + 6, y: TOP, width: W - LABEL_W - 6, height: plotH, rx: 4 });
    for (let t = Math.ceil(xMin * 4) / 4; t <= xMax + 1e-9; t += 0.25) {
      el('line', { class: 'grid', x1: x(t), x2: x(t), y1: TOP, y2: TOP + plotH });
      el('text', { class: 'axis-text', x: x(t), y: TOP + plotH + 16, 'text-anchor': 'middle' }).textContent = t.toFixed(2);
    }
    order.forEach((r, i) => {
      if (i) el('line', { class: 'grid', x1: 0, x2: W, y1: TOP + i * ROW_H, y2: TOP + i * ROW_H });
      el('text', { class: 'region-text', x: LABEL_W / 2, y: rowY(r) + 5, 'text-anchor': 'middle' }).textContent = REGION_LABEL[r];
    });

    const ring = el('circle', { class: 'ring', r: RING_R, pathLength: 100 });
    const nodes = pts.map((p, i) => {
      const g = el('g', { class: 'pt', tabindex: 0, transform: `translate(${x(p.median_cdnorm)},${rowY(p.region)})`,
        'aria-label': `${REGION_LABEL[p.region]} ${TYPE_LABEL[p.type]}: ${p.name}` });
      el('circle', { class: 'hit', r: 14 }, g);
      el('path', { d: SHAPE[p.type], fill: REGION_COLOR[p.region] }, g);
      el('title', {}, g).textContent = `${p.name} (${REGION_LABEL[p.region]} · ${TYPE_LABEL[p.type]})`;
      // Hold while hovered (or touched), then count down from this scene once released.
      g.addEventListener('pointerenter', () => { show(i); hold(true); });
      g.addEventListener('pointerleave', () => hold(false));
      g.addEventListener('focus', () => { show(i); hold(true); });
      g.addEventListener('blur', () => hold(false));
      return g;
    });

    const legend = root.querySelector('.aw-qsc-legend');
    Object.keys(SHAPE).forEach((t) => {
      legend.insertAdjacentHTML('beforeend', `<span>${marker(t, '#555')}${TYPE_LABEL[t]}</span>`);
    });

    // ---- playback ----
    let idx = 0, auto = true, visible = false, held = false, speed = 1, timer = null;

    // Slide the new scene's video in from the right (forward) or left (backward).
    function slideTo(p, dir) {
      const next = document.createElement('video');
      Object.assign(next, { muted: true, loop: true, playsInline: true, preload: 'auto', poster: p.poster, src: p.video });
      next.setAttribute('muted', '');
      next.setAttribute('playsinline', '');   // iOS needs the attributes to autoplay inline
      next.playbackRate = speed;
      track.querySelectorAll('video.leaving').forEach((v) => v.remove());
      const old = video;
      video = next;
      if (old && dir) next.style.transform = `translateX(${dir * 100}%)`;
      track.appendChild(next);
      if (visible) next.play().catch(() => {});
      if (!old) return;
      if (!dir) { old.remove(); return; }
      void next.offsetWidth;   // commit the start position before transitioning
      old.classList.add('leaving');
      next.style.transform = 'translateX(0)';
      old.style.transform = `translateX(${-dir * 100}%)`;
      old.addEventListener('transitionend', () => old.remove(), { once: true });
      setTimeout(() => old.remove(), 1000);
    }

    function show(i) {
      const dir = video ? Math.sign(i - idx) : 0;
      idx = (i + pts.length) % pts.length;
      const p = pts[idx];
      const color = REGION_COLOR[p.region];
      nodes.forEach((n, j) => n.classList.toggle('is-active', j === idx));
      ring.setAttribute('cx', x(p.median_cdnorm));
      ring.setAttribute('cy', rowY(p.region));
      ring.style.stroke = color;
      root.style.setProperty('--aw-active', color);
      if (!video || video.getAttribute('src') !== p.video) slideTo(p, dir);
      nameEl.textContent = p.name;
      subEl.innerHTML = `${marker(p.type, color)}${p.region} · ${TYPE_LABEL[p.type]}
        <span class="sep">·</span> Group Median ${CDHAT} <b>${p.median_cdnorm.toFixed(2)}</b>`;
      typeset(subEl);
    }
    // Countdown to the next scene. `done` is the fraction already elapsed, so a speed
    // change keeps the progress instead of restarting it.
    let startedAt = 0, durationMs = 0;
    function schedule(done = 0) {
      clearTimeout(timer);
      ring.classList.remove('is-counting');
      ring.classList.toggle('is-held', held);   // empty while a hovered scene is held
      ring.style.animationDelay = '';
      void ring.getBBox();
      if (!(auto && visible && !held)) return;
      durationMs = CYCLE_MS / speed;
      startedAt = performance.now() - done * durationMs;
      ring.style.setProperty('--aw-auto-ms', `${durationMs}ms`);
      ring.style.animationDelay = `${-done * durationMs}ms`;
      ring.classList.add('is-counting');
      timer = setTimeout(() => { show(idx + 1); schedule(); }, (1 - done) * durationMs);
    }
    function elapsed() {
      return ring.classList.contains('is-counting')
        ? Math.min((performance.now() - startedAt) / durationMs, 1) : 0;
    }
    function hold(on) {
      held = on;
      schedule();
    }

    root.querySelector('.aw-qcar-auto input').addEventListener('change', (e) => {
      auto = e.target.checked;
      schedule();
    });
    root.querySelectorAll('.aw-speed button').forEach((b) => b.addEventListener('click', () => {
      root.querySelectorAll('.aw-speed button').forEach((o) => o.classList.toggle('is-active', o === b));
      const done = elapsed();
      speed = Number(b.dataset.speed);
      video.playbackRate = speed;
      schedule(done);
    }));
    new IntersectionObserver(([entry]) => {
      visible = entry.isIntersecting;
      if (visible) video.play().catch(() => {}); else video.pause();
      schedule();
    }, { threshold: 0.3 }).observe(root);

    show(0);
  }

  function init() {
    const root = document.getElementById('qual-figure');
    if (!root) return;
    fetch('static/qualitative/qualitative.json')
      .then((r) => r.json())
      .then((data) => build(root, data))
      .catch((e) => console.error('qualitative results:', e));
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
