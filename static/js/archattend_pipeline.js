// Figure 5 of the paper, assembled one step at a time.
//
// The whole figure is drawn once; every element belongs to a stage, and
// stepping forward only ever reveals more, so the last stage is the complete
// figure. The canvas is cropped to the deepest stage on screen, so an early
// step is short and wide rather than a band of content over an empty page.
(function () {
  const NS = 'http://www.w3.org/2000/svg';
  const IMG = 'static/images/method/';

  // sampled from the figure's own vector source
  const ARCH = [['eaves', '#a52a2a'], ['pagoda', '#ffb90f'], ['window', '#e9967a']];
  const NON = [['sky', '#0047ab'], ['trees', '#48d1cc']];
  const TOKENS = ['#0047ab', '#a52a2a', '#ffb90f', '#e9967a', '#48d1cc', '#2e8b57'];
  const ANCHOR = '#ffb90f';
  const TONE = { a: '#7a0000', b: '#28448d', c: '#3b6c1e', d: '#5c4080' };
  // x, y, w, h as fractions of the reconstruction tile
  const CALLOUT = {
    before: [0.2055, 0.2344, 0.3625, 0.3668],
    after: [0.2002, 0.2243, 0.3686, 0.3604],
  };
  const RULE = '#e5e7eb', INK = '#1f2937', FAINT = '#9ca3af';

  const W = 1200, TILE = 96, OUT = 112;
  // how far down the canvas each stage reaches
  // bottom y each stage reaches; the output tiles end at 744 (632 + 88 + border)
  const DEPTH = [312, 312, 312, 312, 312, 496, 752, 752];
  const FULL = DEPTH[DEPTH.length - 1];
  const TOP = 26;   // crop the air above the first band title

  const STAGES = [
    { label: 'Input', tone: TONE.a,
      caption: 'Every view of the scene is segmented with SAM 3.' },
    { label: 'Elements', tone: TONE.a,
      caption: 'The segments divide into architectural elements and non-architectural ones — sky, water, trees.' },
    { label: 'Encode', tone: TONE.b,
      caption: 'Both views go through the 3D foundation model; we read the token representation out of block 23.' },
    { label: 'Score', tone: TONE.b,
      caption: 'For each element, sum the attention its tokens in one view pay to its tokens in the other.' },
    { label: 'Anchor', tone: TONE.b,
      caption: 'The architectural element that scores highest becomes the anchor. We ship the anchors, so a run of the released code starts at Pass 2.' },
    { label: 'Steer', tone: TONE.c,
      caption: 'Run the model again: in every global attention block, one gradient-ascent step moves the tokens toward the anchor. The weights never change.' },
    { label: 'Before', tone: TONE.d,
      caption: 'Before the intervention, attention spreads over sky and trees, and the reconstruction breaks up.' },
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
  // Mathematics is set with KaTeX inside a foreignObject: the page already
  // loads KaTeX, and hand-placing superscripts in <tspan> had drifted out of
  // alignment and relied on glyphs the page font does not carry.
  // The box is given a fixed width and the content aligned inside it, so
  // nothing needs measuring. The plain-text body is the no-KaTeX fallback.
  function tex(add, s, x, y, w, latex, plain, align = 'start', size = 17) {
    const h = size * 2.4;
    const fo = add(s, 'foreignObject', {
      x: align === 'middle' ? x - w / 2 : align === 'end' ? x - w : x,
      y: y - h / 2, width: w, height: h, style: 'overflow:visible',
    });
    const d = document.createElement('div');
    d.className = 'aw-pipe-tex';
    d.style.cssText = 'font-size:' + size + 'px;height:' + h + 'px;display:flex;'
      + 'align-items:center;justify-content:'
      + (align === 'middle' ? 'center' : align === 'end' ? 'flex-end' : 'flex-start');
    d.dataset.tex = latex;
    d.textContent = plain;
    fo.appendChild(d);
    return fo;
  }
  function typeset(root) {
    if (!window.katex) return false;
    root.querySelectorAll('[data-tex]').forEach((d) => {
      try {
        window.katex.render(d.dataset.tex, d, { throwOnError: false, displayMode: false });
      } catch (e) { /* the plain-text fallback stays */ }
      d.removeAttribute('data-tex');
    });
    return true;
  }
  function whenKatex(fn) {
    if (fn()) return;
    const iv = setInterval(() => { if (fn()) clearInterval(iv); }, 150);
    setTimeout(() => clearInterval(iv), 20000);
  }
  const lab = (add, s, x, y, t, anchor = 'start', cls = 'aw-pipe-lab') =>
    add(s, 'text', { x, y, 'text-anchor': anchor, class: cls }, t);
  function tile(add, s, x, y, size, file, alt) {
    add(s, 'image', { x, y, width: size, height: size, href: IMG + file,
      preserveAspectRatio: 'xMidYMid meet', 'aria-label': alt });
    add(s, 'rect', { x: x + 0.5, y: y + 0.5, width: size - 1, height: size - 1,
      rx: 4, fill: 'none', stroke: RULE });
  }
  const arrow = (add, s, x1, x2, y, color = FAINT) =>
    add(s, 'path', { d: `M ${x1} ${y} L ${x2} ${y}`, stroke: color,
      'stroke-width': 1.4, 'marker-end': 'url(#aw-tip)' });

  // ------------------------------------------------------------------ build
  function build() {
    const svg = el('svg', { viewBox: `0 ${TOP} ${W} ${FULL - TOP}`, class: 'aw-pipe-svg', role: 'img',
      'aria-label': 'The ArchAttend pipeline, from input views to the steered reconstruction' });
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
      add(s, 'line', { x1, x2, y1: y + 11, y2: y + 11, stroke: tone, 'stroke-width': 2 });
    };

    // ===== A. input, x 24..470 ============================================
    const PX = 96, MX = 256, TY = [84, 196];
    band(0, 24, 470, 44, TONE.a, 'INPUT IMAGES AND SEMANTIC ELEMENTS');
    lab(add, 0, PX + TILE / 2, 76, 'IMAGES', 'middle');
    lab(add, 0, (PX + TILE + MX) / 2, 76, 'SAM 3', 'middle');
    lab(add, 0, MX + TILE / 2, 76, 'MASKS', 'middle');
    TY.forEach((y, i) => {
      const n = String(i + 1), cy = y + TILE / 2;
      tex(add, 0, 50, cy, 40, `I_{${n}}`, `I${n}`, 'end', 17);
      tile(add, 0, PX, y, TILE, `i${n}.jpg`, `Input view ${n}`);
      tile(add, 0, MX, y, TILE, `i${n}_mask.jpg`, `Segmentation of view ${n}`);
      arrow(add, 0, PX + TILE + 10, MX - 10, cy);
    });

    // ----- elements -------------------------------------------------------
    const LX = 392;
    add(1, 'path', { d: 'M 372 92 q 9 0 9 9 l 0 74 q 0 9 9 9 q -9 0 -9 9 l 0 74 q 0 9 -9 9',
      fill: 'none', stroke: FAINT, 'stroke-width': 1.2 });
    tex(add, 1, LX, 100, 130, '\\mathcal{E}_{\\mathrm{arch}}', 'E arch', 'start', 19);
    ARCH.forEach(([name, c], i) => {
      const y = 130 + i * 25;
      add(1, 'rect', { x: LX, y: y - 10, width: 12, height: 12, rx: 2, fill: c });
      add(1, 'text', { x: LX + 20, y, class: 'aw-pipe-el' }, `“${name}”`);
    });
    tex(add, 1, LX, 222, 130, '\\mathcal{E}_{\\mathrm{non}}', 'E non', 'start', 19);
    NON.forEach(([name, c], i) => {
      const y = 252 + i * 25;
      add(1, 'rect', { x: LX, y: y - 10, width: 12, height: 12, rx: 2, fill: c });
      add(1, 'text', { x: LX + 20, y, class: 'aw-pipe-el' }, `“${name}”`);
    });

    // ===== B. pass 1, x 510..1176 =========================================
    const BBX = 592, BBW = 44, SX = 684, QX = 752;
    band(2, 510, 1004, 44, TONE.b, 'PASS 1 · DETERMINE THE ANCHOR ELEMENT');
    add(2, 'rect', { x: BBX, y: 80, width: BBW, height: 212, rx: 6, fill: '#fff3c2', stroke: '#e0ad3a' });
    add(2, 'text', { x: BBX + BBW / 2, y: 186, 'text-anchor': 'middle', class: 'aw-pipe-el is-bb',
      transform: `rotate(-90 ${BBX + BBW / 2} 186)` }, '3DFM Backbone');
    tex(add, 2, SX + 6.5, 64, 90, '\\mathbf{T}^{23}', 'T23', 'middle', 17);
    TY.forEach((y, i) => {
      const cy = y + TILE / 2;
      tex(add, 2, 542, cy, 40, `I_{${i + 1}}`, `I${i + 1}`, 'end', 17);
      arrow(add, 2, 546, BBX - 8, cy);
      arrow(add, 2, BBX + BBW + 8, SX - 12, cy);
      TOKENS.forEach((c, k) =>
        add(2, 'rect', { x: SX, y: cy - 46 + k * 16, width: 13, height: 13, rx: 2, fill: c }));
    });

    // ----- score: fixed slots, and the one variable-width item is last -----
    const ROW = [112, 158, 204];
    ARCH.forEach(([name, c], i) => {
      tex(add, 3, QX, ROW[i], 430,
        `\\sum \\operatorname{attn}\\!\\left(\\textcolor{${c}}{\\blacksquare},\\;`
        + `{\\large\\textcolor{${c}}{\\bullet}}\\right)\\;=\\;`
        + `S^{23}\\!\\left(\\textcolor{${c}}{\\textbf{${name}}}\\right)`,
        `S23(${name})`, 'start', 18);
    });

    // ----- anchor ---------------------------------------------------------
    add(4, 'rect', { x: QX - 14, y: ROW[1] - 21, width: 392, height: 34, rx: 8,
      fill: 'none', stroke: ANCHOR, 'stroke-width': 2 });
    lab(add, 4, QX, 262, 'MAX', 'start');
    arrow(add, 4, QX + 46, QX + 82, 257, INK);
    tex(add, 4, QX + 96, 262, 300,
      `e^{*} \\;=\\; \\textcolor{${ANCHOR}}{\\textbf{pagoda}}`, 'e* = pagoda', 'start', 19);

    // ===== C. pass 2, y 324..476 ==========================================
    const CY = 432;
    add(5, 'line', { x1: 24, x2: 1176, y1: 324, y2: 324, stroke: RULE });
    band(5, 24, 1176, 352, TONE.c, 'PASS 2 · ARCHITECTURE-CENTRIC INTERVENTION');
    const cols = (s, cx, color) => {
      [-19, 19].forEach((off, k) => {
        const x = cx + off - 5.5;
        add(s, 'rect', { x: x - 5, y: CY - 42, width: 23, height: 84, rx: 4,
          fill: 'none', stroke: INK, 'stroke-width': 1.4 });
        for (let i = 0; i < 5; i++) {
          const cy = CY - 36 + i * 15;
          if (k) add(s, 'circle', { cx: x + 5.5, cy: cy + 5.5, r: 5.5, fill: color });
          else add(s, 'rect', { x, y: cy, width: 11, height: 11, rx: 2, fill: color });
        }
      });
    };
    // One measure across the full width: tokens, the update, tokens, the note.
    const TL = 86, BOX0 = 186, BOXW = 600, TR = 900, NOTE = 1000;
    const BOXC = BOX0 + BOXW / 2;
    tex(add, 5, TL + 10, CY - 58, 120, '\\mathbf{T}^{\\ell}', 'T l', 'middle', 17);
    cols(5, TL, '#a6a6a6');
    arrow(add, 5, TL + 46, BOX0 - 14, CY, INK);
    add(5, 'rect', { x: BOX0, y: CY - 42, width: BOXW, height: 84, rx: 8, fill: '#fafafa', stroke: RULE });
    tex(add, 5, BOXC, CY - 10, BOXW - 60,
      '\\mathbf{T}^{\\ell} \\;\\leftarrow\\; \\mathbf{T}^{\\ell} '
      + '+ \\eta \\cdot \\nabla S^{\\ell}(e^{*})',
      'T^l <- T^l + eta grad S^l(e*)', 'middle', 21);
    add(5, 'text', { x: BOXC, y: CY + 28, 'text-anchor': 'middle', class: 'aw-pipe-note' },
      'one step per block · model weights frozen');
    arrow(add, 5, BOX0 + BOXW + 14, TR - 46, CY, INK);
    tex(add, 5, TR + 10, CY - 58, 140, '\\mathbf{T}^{\\ell+1}', 'T l+1', 'middle', 17);
    cols(5, TR, ANCHOR);
    lab(add, 5, NOTE, CY - 6, '× 24 BLOCKS', 'start');
    lab(add, 5, NOTE, CY + 14, 'ONE PASS, NO RETRAINING', 'start');

    // ===== D. output, y 494..700 ==========================================
    add(6, 'line', { x1: 24, x2: 1176, y1: 522, y2: 522, stroke: RULE });
    band(6, 24, 1176, 550, TONE.d, 'OUTPUT RECONSTRUCTION AND ATTENTION MAPS');
    const COLS = ['Reconstruction', 'Close-up', 'Attention I₁', 'Attention I₂'];
    const FILES = (k) => [`recon_${k}.jpg`, `zoom_${k}.jpg`, `attn_${k}_1.jpg`, `attn_${k}_2.jpg`];
    // Two groups of four, mirrored about the canvas centre so the row reads as
    // one comparison rather than a cluster adrift on the left.
    const GAP = 12, GW = 4 * OUT + 3 * GAP, SPLIT = 72;
    const BX = (W - 2 * GW - SPLIT) / 2, AX = BX + GW + SPLIT;
    [['before', 6, BX, '#ff4444'], ['after', 7, AX, '#ff9800']].forEach(([k, s, x0, tone]) => {
      add(s, 'rect', { x: x0, y: 584, width: GW, height: 22, rx: 11, fill: tone, opacity: 0.14 });
      lab(add, s, x0 + 14, 599, k.toUpperCase(), 'start', 'aw-pipe-side');
      FILES(k).forEach((f, i) => {
        const x = x0 + i * (OUT + GAP);
        lab(add, s, x + OUT / 2, 624, COLS[i], 'middle');
        tile(add, s, x, 632, OUT, f, `${COLS[i]} ${k} the intervention`);
      });
      const [cx, cy, cw, ch] = CALLOUT[k];
      add(s, 'rect', { x: x0 + cx * OUT, y: 632 + cy * OUT, width: cw * OUT, height: ch * OUT,
        rx: 2, fill: 'none', stroke: tone, 'stroke-width': 1.6 });
      add(s, 'rect', { x: x0 + OUT + GAP + 0.5, y: 632.5, width: OUT - 1, height: OUT - 1,
        rx: 4, fill: 'none', stroke: tone, 'stroke-width': 2 });
    });
    return svg;
  }

  // ------------------------------------------------------------------- wire
  const DWELL = [1500, 1200, 1500, 1900, 1400, 1900, 1600, 1800];

  function init() {
    const root = document.getElementById('archattend-pipeline');
    if (!root) return;
    const figure = root.querySelector('.aw-pipe-figure');
    const strip = root.querySelector('.aw-pipe-steps');
    const cap = root.querySelector('.aw-pipe-caption');
    const playBtn = root.querySelector('.aw-pipe-play');
    const resetBtn = root.querySelector('.aw-pipe-reset');
    const slow = window.matchMedia
      ? window.matchMedia('(prefers-reduced-motion: reduce)') : { matches: false };

    const svg = build();
    figure.appendChild(svg);
    whenKatex(() => typeset(svg));
    const ops = [...figure.querySelectorAll('.aw-pipe-op')];
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

    let overview = true, at = STAGES.length - 1, running = false, raf = 0, last = 0, t = 1;
    const done = () => at === STAGES.length - 1 && t >= 1;

    function render() {
      ops.forEach((g, n) => g.classList.toggle('is-on', overview || n <= at));
      const h = (overview ? FULL : DEPTH[at]) - TOP;
      svg.setAttribute('viewBox', `0 ${TOP} ${W} ${h}`);
      figure.style.setProperty('--figure-ratio', `${W}/${h}`);
      buttons.forEach((b, n) => {
        b.dataset.state = overview ? 'overview' : n < at ? 'complete' : n === at ? 'current' : 'pending';
        if (!overview && n === at) b.setAttribute('aria-current', 'step');
        else b.removeAttribute('aria-current');
      });
      if (!overview) {
        const b = buttons[at];
        strip.scrollTo({ left: Math.max(0, b.offsetLeft - (strip.clientWidth - b.offsetWidth) / 2),
          behavior: slow.matches ? 'instant' : 'smooth' });
      } else strip.scrollTo({ left: 0, behavior: 'instant' });
      const mode = running ? 'pause' : (!overview && done()) ? 'replay' : 'play';
      playBtn.querySelector('use').setAttribute('href', `#aw-i-${mode}`);
      const text = mode === 'pause' ? 'Pause' : mode === 'replay' ? 'Replay' : 'Play';
      playBtn.querySelector('span').textContent = text;
      playBtn.setAttribute('aria-label', `${text} the pipeline`);
      playBtn.title = `${text} the pipeline`;
      if (overview) cap.innerHTML = overviewCaption;
      else cap.textContent = `${at + 1} / ${STAGES.length} · ${STAGES[at].caption}`;
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

    if ('IntersectionObserver' in window) {
      new IntersectionObserver((es) => { if (!es[0].isIntersecting && running) pause(); },
        { threshold: 0.08 }).observe(root);
    }
    document.addEventListener('visibilitychange', () => { if (document.hidden && running) pause(); });
    render();
  }

  // SVG <image> has no loading="lazy", so building eagerly would pull all
  // twelve tiles on first paint for a figure well below the fold.
  function boot() {
    const root = document.getElementById('archattend-pipeline');
    if (!root) return;
    if (!('IntersectionObserver' in window)) { init(); return; }
    const io = new IntersectionObserver((entries) => {
      if (!entries.some((e) => e.isIntersecting)) return;
      io.disconnect();
      init();
    }, { rootMargin: '400px' });
    io.observe(root);
  }
  document.readyState === 'loading' ? document.addEventListener('DOMContentLoaded', boot) : boot();
})();
