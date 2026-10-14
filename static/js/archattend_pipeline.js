// Figure 5 of the paper, assembled one step at a time.
//
// The whole figure is drawn once; every element belongs to a stage, and
// stepping forward only ever reveals more. Nothing that has appeared is taken
// away again, so the last stage is the complete figure.
(function () {
  const NS = 'http://www.w3.org/2000/svg';
  const IMG = 'static/images/method/';

  // sampled from the figure's own vector source
  const ARCH = [['eaves', '#a52a2a'], ['pagoda', '#ffb90f'], ['window', '#e9967a']];
  const NON = [['sky', '#0047ab'], ['trees', '#48d1cc']];
  const TOKENS = ['#0047ab', '#a52a2a', '#ffb90f', '#e9967a', '#48d1cc', '#2e8b57'];
  const ANCHOR = '#ffb90f';
  const TONE = { a: '#7a0000', b: '#28448d', c: '#3b6c1e', d: '#5c4080' };
  const RULE = '#e5e7eb', INK = '#1f2937', MUTED = '#6b7280', FAINT = '#9ca3af';

  const W = 1000, H = 724, TILE = 100;

  const STAGES = [
    { label: 'Input', tone: TONE.a,
      caption: 'Every view of the scene is segmented with SAM 3.' },
    { label: 'Elements', tone: TONE.a,
      caption: 'The segments split into architectural elements ℰarch and non-architectural ones ℰnon.' },
    { label: 'Encode', tone: TONE.b,
      caption: 'The views go through the 3D foundation model; we read out the token representation at block 23.' },
    { label: 'Score', tone: TONE.b,
      caption: 'For each element, sum the attention its tokens in one view pay to its tokens in the other.' },
    { label: 'Anchor', tone: TONE.b,
      caption: 'The highest-scoring architectural element becomes the anchor e∗. We ship the anchors, so a run starts at Pass 2.' },
    { label: 'Steer', tone: TONE.c,
      caption: 'In every global attention block, one gradient-ascent step moves the tokens toward the anchor. Weights stay frozen.' },
    { label: 'Before', tone: TONE.d,
      caption: 'Before the intervention, attention spreads over sky and trees and the reconstruction breaks up.' },
    { label: 'After', tone: TONE.d,
      caption: 'After it, attention gathers on the building and the reconstruction fills in.' },
  ];

  // ---------------------------------------------------------------- helpers
  const el = (n, a = {}, t) => {
    const e = document.createElementNS(NS, n);
    for (const k in a) e.setAttribute(k, a[k]);
    if (t != null) e.textContent = t;
    return e;
  };
  function layer(svg) {
    const groups = STAGES.map((_, i) => {
      const g = el('g', { 'data-stage': String(i), class: 'aw-pipe-op' });
      svg.appendChild(g);
      return g;
    });
    return (i, n, a, t) => { const e = el(n, a, t); groups[i].appendChild(e); return e; };
  }
  const lab = (add, s, x, y, t, anchor = 'start', cls = 'aw-pipe-lab') =>
    add(s, 'text', { x, y, 'text-anchor': anchor, class: cls }, t);

  function tile(add, s, x, y, file, alt) {
    add(s, 'image', {
      x, y, width: TILE, height: TILE, href: IMG + file,
      preserveAspectRatio: 'xMidYMid slice', 'aria-label': alt,
    });
    add(s, 'rect', { x: x + 0.5, y: y + 0.5, width: TILE - 1, height: TILE - 1,
      rx: 4, fill: 'none', stroke: RULE });
  }
  function chips(add, s, x, y, n, color, round) {
    for (let i = 0; i < n; i++) {
      const cy = y + i * 14;
      if (round) add(s, 'circle', { cx: x + 5.5, cy: cy + 5.5, r: 5.5, fill: color });
      else add(s, 'rect', { x, y: cy, width: 11, height: 11, rx: 2, fill: color });
    }
    return y + n * 14 - 3;
  }
  const arrow = (add, s, x1, x2, y, color = FAINT) =>
    add(s, 'path', { d: `M ${x1} ${y} L ${x2} ${y}`, stroke: color,
      'stroke-width': 1.4, 'marker-end': 'url(#aw-tip)' });

  // ------------------------------------------------------------------ build
  function build() {
    const svg = el('svg', {
      viewBox: `0 0 ${W} ${H}`, class: 'aw-pipe-svg', role: 'img',
      'aria-label': 'The ArchAttend pipeline, from input views to the steered reconstruction',
    });
    const defs = el('defs');
    [['aw-tip', FAINT], ['aw-tip-i', INK]].forEach(([id, fill]) => {
      const m = el('marker', { id, viewBox: '0 0 10 10', refX: '9', refY: '5',
        markerWidth: '5.5', markerHeight: '5.5', orient: 'auto-start-reverse' });
      m.appendChild(el('path', { d: 'M 0 1 L 9 5 L 0 9 z', fill }));
      defs.appendChild(m);
    });
    svg.appendChild(defs);
    const add = layer(svg);
    const band = (s, x1, x2, y, tone, text) => {
      lab(add, s, x1, y, text, 'start', 'aw-pipe-head');
      add(s, 'line', { x1, x2, y1: y + 10, y2: y + 10, stroke: tone, 'stroke-width': 2 });
    };

    // ===== section A: input, y 30..300 ====================================
    band(0, 24, 420, 44, TONE.a, 'INPUT IMAGES AND SEMANTIC ELEMENTS');
    lab(add, 0, 94, 70, 'IMAGES', 'middle');
    lab(add, 0, 163, 70, 'SAM 3', 'middle');
    lab(add, 0, 230, 70, 'MASKS', 'middle');
    [['1', 80], ['2', 196]].forEach(([n, y]) => {
      add(0, 'text', { x: 26, y: y + 54, class: 'aw-pipe-vlab' }, 'I');
      add(0, 'text', { x: 32, y: y + 58, class: 'aw-pipe-sub' }, n);
      tile(add, 0, 44, y, `i${n}.jpg`, `Input view ${n}`);
      tile(add, 0, 180, y, `i${n}_mask.jpg`, `Segmentation of view ${n}`);
      arrow(add, 0, 150, 174, y + TILE / 2);
    });

    // ----- elements -------------------------------------------------------
    add(1, 'path', { d: 'M 296 86 q 9 0 9 9 l 0 44 q 0 9 9 9 q -9 0 -9 9 l 0 44 q 0 9 -9 9',
      fill: 'none', stroke: FAINT, 'stroke-width': 1.2 });
    lab(add, 1, 322, 96, '\u2130arch', 'start', 'aw-pipe-sym');
    ARCH.forEach(([name, c], i) => {
      const y = 124 + i * 24;
      add(1, 'rect', { x: 322, y: y - 10, width: 12, height: 12, rx: 2, fill: c });
      add(1, 'text', { x: 342, y, class: 'aw-pipe-el' }, `\u201c${name}\u201d`);
    });
    lab(add, 1, 322, 218, '\u2130non', 'start', 'aw-pipe-sym');
    NON.forEach(([name, c], i) => {
      const y = 246 + i * 24;
      add(1, 'rect', { x: 322, y: y - 10, width: 12, height: 12, rx: 2, fill: c });
      add(1, 'text', { x: 342, y, class: 'aw-pipe-el' }, `\u201c${name}\u201d`);
    });

    // ===== section B: pass 1 ==============================================
    band(2, 452, 976, 44, TONE.b, 'PASS 1 \u00b7 DETERMINE THE ANCHOR ELEMENT');
    add(2, 'rect', { x: 504, y: 82, width: 42, height: 196, rx: 6, fill: '#fff3c2', stroke: '#e0ad3a' });
    add(2, 'text', { x: 525, y: 180, 'text-anchor': 'middle', class: 'aw-pipe-el is-bb',
      transform: 'rotate(-90 525 180)' }, '3DFM Backbone');
    lab(add, 2, 581, 70, 'T\u00b2\u00b3', 'middle', 'aw-pipe-sym');
    [['1', 130], ['2', 246]].forEach(([n, cy]) => {
      add(2, 'text', { x: 456, y: cy + 5, class: 'aw-pipe-vlab' }, 'I');
      add(2, 'text', { x: 462, y: cy + 9, class: 'aw-pipe-sub' }, n);
      arrow(add, 2, 478, 500, cy);
      arrow(add, 2, 550, 570, cy);
      TOKENS.forEach((c, i) =>
        add(2, 'rect', { x: 576, y: cy - 42 + i * 14, width: 11, height: 11, rx: 2, fill: c }));
    });

    // ----- score ----------------------------------------------------------
    const SX = 636;
    ARCH.forEach(([name, c], i) => {
      const y = 120 + i * 46;
      add(3, 'text', { x: SX, y, class: 'aw-pipe-sum' }, '\u2211');
      add(3, 'text', { x: SX + 20, y, class: 'aw-pipe-fn' }, 'attn');
      add(3, 'text', { x: SX + 58, y: y + 3, class: 'aw-pipe-paren' }, '(');
      add(3, 'rect', { x: SX + 68, y: y - 11, width: 13, height: 13, rx: 2, fill: c });
      add(3, 'text', { x: SX + 86, y, class: 'aw-pipe-paren' }, ',');
      add(3, 'circle', { cx: SX + 104, cy: y - 4.5, r: 6.5, fill: c });
      add(3, 'text', { x: SX + 114, y: y + 3, class: 'aw-pipe-paren' }, ')');
      add(3, 'text', { x: SX + 130, y, class: 'aw-pipe-el' }, '=');
      add(3, 'text', { x: SX + 148, y, class: 'aw-pipe-sc' }, 'S');
      add(3, 'text', { x: SX + 160, y: y - 8, class: 'aw-pipe-sup' }, '23');
      add(3, 'text', { x: SX + 178, y: y + 3, class: 'aw-pipe-paren' }, '(');
      add(3, 'text', { x: SX + 188, y, class: 'aw-pipe-el is-name', fill: c }, name);
      add(3, 'text', { x: SX + 192 + name.length * 8.6, y: y + 3, class: 'aw-pipe-paren' }, ')');
    });

    // ----- anchor ---------------------------------------------------------
    add(4, 'rect', { x: SX - 12, y: 148, width: 316, height: 32, rx: 8,
      fill: 'none', stroke: ANCHOR, 'stroke-width': 2 });
    lab(add, 4, SX, 268, 'MAX', 'start');
    arrow(add, 4, SX + 40, SX + 70, 263, INK);
    add(4, 'text', { x: SX + 84, y: 268, class: 'aw-pipe-estar' }, 'e\u2217 =');
    add(4, 'text', { x: SX + 124, y: 268, class: 'aw-pipe-estar is-name', fill: ANCHOR }, 'pagoda');

    // ===== section C: pass 2, y 312..481 ==================================
    add(5, 'line', { x1: 24, x2: 976, y1: 312, y2: 312, stroke: RULE });
    band(5, 24, 976, 344, TONE.c, 'PASS 2 \u00b7 ARCHITECTURE-CENTRIC INTERVENTION');
    const tokenCols = (s, x0, color) => {
      [[x0, false], [x0 + 36, true]].forEach(([x, round]) => {
        add(s, 'rect', { x: x - 5, y: 374, width: 24, height: 107, rx: 4,
          fill: 'none', stroke: INK, 'stroke-width': 1.4 });
        chips(add, s, x, 379, 7, color, round);
      });
    };
    lab(add, 5, 67, 366, 'T\u2113', 'middle', 'aw-pipe-sym');
    tokenCols(5, 40, '#a6a6a6');
    arrow(add, 5, 112, 146, 428, INK);
    add(5, 'rect', { x: 158, y: 384, width: 466, height: 88, rx: 8, fill: '#fafafa', stroke: RULE });
    add(5, 'text', { x: 391, y: 420, 'text-anchor': 'middle', class: 'aw-pipe-rule' },
      'T\u2113  \u2190  T\u2113 + \u03b7 \u00b7 \u2207S\u2113(e\u2217)');
    add(5, 'text', { x: 391, y: 450, 'text-anchor': 'middle', class: 'aw-pipe-note' },
      'one step per block \u00b7 model weights frozen');
    arrow(add, 5, 638, 672, 428, INK);
    lab(add, 5, 711, 366, 'T\u2113\u207a\u00b9', 'middle', 'aw-pipe-sym');
    tokenCols(5, 684, ANCHOR);
    lab(add, 5, 780, 422, '\u00d7 24 BLOCKS', 'start');
    lab(add, 5, 780, 442, 'ONE PASS, NO RETRAINING', 'start');

    // ===== section D: output, y 504..708 ==================================
    add(6, 'line', { x1: 24, x2: 976, y1: 504, y2: 504, stroke: RULE });
    band(6, 24, 976, 536, TONE.d, 'OUTPUT RECONSTRUCTION AND ATTENTION MAPS');
    const COLS = ['Reconstruction', 'Close-up', 'Attention I\u2081', 'Attention I\u2082'];
    const FILES = (k) => [`recon_${k}.jpg`, `zoom_${k}.jpg`, `attn_${k}_1.jpg`, `attn_${k}_2.jpg`];
    [['before', 6, 40, '#ff4444'], ['after', 7, 530, '#ff9800']].forEach(([k, s, x0, tone]) => {
      add(s, 'rect', { x: x0 - 10, y: 556, width: 450, height: 22, rx: 11, fill: tone, opacity: 0.14 });
      lab(add, s, x0, 571, k.toUpperCase(), 'start', 'aw-pipe-side');
      FILES(k).forEach((f, i) => {
        const x = x0 + i * 110;
        tile(add, s, x, 590, f, `${COLS[i]} ${k} the intervention`);
        lab(add, s, x + TILE / 2, 708, COLS[i], 'middle');
      });
    });
    return svg;
  }

  // ------------------------------------------------------------------- wire
  // Per-stage dwell when playing, in ms, hand-tuned to how much each stage adds.
  const DWELL = [2400, 2000, 2600, 3200, 2400, 3400, 2800, 3000];

  function init() {
    const root = document.getElementById('archattend-pipeline');
    if (!root) return;
    const figure = root.querySelector('.aw-pipe-figure');
    const strip = root.querySelector('.aw-pipe-steps');
    const cap = root.querySelector('.aw-pipe-caption');
    const playBtn = root.querySelector('.aw-pipe-play');
    const resetBtn = root.querySelector('.aw-pipe-reset');
    const slow = window.matchMedia
      ? window.matchMedia('(prefers-reduced-motion: reduce)')
      : { matches: false };

    figure.appendChild(build());
    const ops = [...figure.querySelectorAll('.aw-pipe-op')];
    // the caption the page ships with doubles as the complete-figure caption
    const overviewCaption = cap.innerHTML;

    const buttons = STAGES.map((st, i) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.dataset.step = String(i);
      b.dataset.state = 'overview';
      b.title = st.label;
      b.setAttribute('aria-label', `Stage ${i + 1}: ${st.label}`);
      b.innerHTML = `<span class="aw-pipe-step-l">${st.label}</span>`;
      b.addEventListener('click', () => { overview = false; at = i; pause(); });
      strip.appendChild(b);
      return b;
    });

    // Opening on the complete figure, the way the printed figure is read.
    let overview = true, at = STAGES.length - 1, running = false, raf = 0, last = 0, t = 1;
    const done = () => at === STAGES.length - 1 && t >= 1;

    function render() {
      ops.forEach((g, n) => g.classList.toggle('is-on', overview || n <= at));
      buttons.forEach((b, n) => {
        b.dataset.state = overview ? 'overview' : n < at ? 'complete' : n === at ? 'current' : 'pending';
        if (!overview && n === at) b.setAttribute('aria-current', 'step');
        else b.removeAttribute('aria-current');
      });
      if (!overview) {
        const b = buttons[at];
        strip.scrollTo({ left: Math.max(0, b.offsetLeft - (strip.clientWidth - b.offsetWidth) / 2),
          behavior: slow.matches ? 'instant' : 'smooth' });
      } else {
        strip.scrollTo({ left: 0, behavior: 'instant' });
      }
      const mode = running ? 'pause' : (!overview && done()) ? 'replay' : 'play';
      playBtn.querySelector('use').setAttribute('href', `#aw-i-${mode}`);
      const text = mode === 'pause' ? 'Pause' : mode === 'replay' ? 'Replay' : 'Play';
      playBtn.querySelector('span').textContent = text;
      playBtn.setAttribute('aria-label', `${text} the pipeline`);
      playBtn.title = `${text} the pipeline`;
      if (overview) cap.innerHTML = overviewCaption;
      else cap.textContent = `${at + 1} / ${STAGES.length} \u00b7 ${STAGES[at].caption}`;
      root.style.setProperty('--tone', overview ? 'var(--aw-accent)' : STAGES[at].tone);
    }

    function pause() { running = false; cancelAnimationFrame(raf); last = 0; render(); }
    function tick(now) {
      if (!running) return;
      if (last) t += Math.min(now - last, 100) / DWELL[at];
      last = now;
      if (t >= 1) {
        t = 1;
        if (at === STAGES.length - 1) { pause(); return; }
        at += 1; t = 0; render();
      }
      raf = requestAnimationFrame(tick);
    }
    playBtn.addEventListener('click', () => {
      if (running) { pause(); return; }
      if (overview || done()) { at = 0; t = 0; } else if (t >= 1) t = 0;
      overview = false;
      if (slow.matches) { at = STAGES.length - 1; t = 1; render(); return; }
      running = true; last = 0; render();
      raf = requestAnimationFrame(tick);
    });
    resetBtn.addEventListener('click', () => { overview = true; at = STAGES.length - 1; t = 1; pause(); });

    strip.addEventListener('keydown', (e) => {
      const i = buttons.indexOf(e.target.closest('[data-step]'));
      if (i < 0 || !['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(e.key)) return;
      e.preventDefault();
      const n = e.key === 'Home' ? 0 : e.key === 'End' ? buttons.length - 1
        : (i + (e.key === 'ArrowRight' ? 1 : -1) + buttons.length) % buttons.length;
      buttons[n].focus({ preventScroll: true });
      buttons[n].click();
    });

    // playing off-screen or in a hidden tab just burns frames
    if ('IntersectionObserver' in window) {
      new IntersectionObserver((es) => { if (!es[0].isIntersecting && running) pause(); },
        { threshold: 0.08 }).observe(root);
    }
    document.addEventListener('visibilitychange', () => { if (document.hidden && running) pause(); });

    render();
  }
  document.readyState === 'loading' ? document.addEventListener('DOMContentLoaded', init) : init();
})();
