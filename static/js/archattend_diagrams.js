/* One compact diagram, with a shared native animation timeline. */
(function () {
  'use strict';
  const root = document.getElementById('archattend-pipeline');
  if (!root) return;
  const IMG = 'static/images/method/';
  const colors = { eaves: '#a52a2a', pagoda: '#ffb90f', window: '#e9967a', sky: '#0047ab', trees: '#48d1cc', water: '#2e8b57' };
  const esc = value => String(value).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const math = (source, cls = '') => '<span class="awcr-math ' + cls + '" data-tex="' + esc(source) + '">' + esc(source) + '</span>';
  const arrow = cls => '<svg class="awcr-arrow ' + (cls || '') + '" viewBox="0 0 24 16" aria-hidden="true"><path pathLength="1" d="M1 8h22m-5-5 5 5-5 5"/></svg>';
  const brace = cls => '<svg class="awcr-brace ' + cls + '" viewBox="0 0 12 130" preserveAspectRatio="none" aria-hidden="true"><path pathLength="1" d="M2 1Q8 1 8 10V53Q8 65 11 65Q8 65 8 77V120Q8 129 2 129"/></svg>';
  const marks = (round, cls) => '<div class="awcr-semantic-strip ' + (round ? 'is-round ' : '') + cls + '">' +
    Object.entries(colors).map(([name, color]) => '<i data-element="' + name + '" style="--mark-color:' + color + '"></i>').join('') + '</div>';
  const pair = (body, cls = '') => '<div class="awcr-pair ' + cls + '">' + body(1) + body(2) + '</div>';
  const vector = (round, cls = '') => '<span class="awcr-vector ' + cls + (round ? ' is-round' : '') + '">' + '<i></i>'.repeat(5) + '</span>';
  const bank = updated => '<div class="awcr-bank ' + (updated ? 'is-updated' : 'is-input') + '">' +
    pair(n => '<div class="awcr-vector-row">' + vector(n === 2) + vector(n === 2) + '</div>') + '</div>';
  const icons = {
    play: '<path d="m8 5 11 7-11 7Z" fill="currentColor" stroke="none"/>',
    pause: '<path d="M8 5v14M16 5v14" stroke-width="3"/>',
    replay: '<path d="M4 10a8 8 0 1 1 .8 7M4 4v6h6"/>'
  };
  const icon = name => '<svg viewBox="0 0 24 24" aria-hidden="true">' + icons[name] + '</svg>';
  function inputs() {
    return '<section id="awcr-input" class="awcr-panel awcr-inputs" aria-label="Input images and semantic elements">' +
      '<div class="awcr-input-grid"><span></span><span class="awcr-subhead">Images</span><span class="awcr-subhead">Masks</span>' +
      math('\\mathcal{E}_{\\mathrm{arch}}', 'awcr-set-label') +
      [1, 2].map(n => '<div class="awcr-input-label">' + math('I_' + n) + '</div>' +
        '<div class="awcr-photo awcr-view"><img src="' + IMG + 'i' + n + '_paper.jpg" alt="Input view ' + n + '" width="480" height="360"></div>' +
        '<div class="awcr-photo awcr-mask"><img class="awcr-mask-source" src="' + IMG + 'i' + n + '_paper.jpg" alt="" width="480" height="360"><img class="awcr-mask-image" src="' + IMG + 'i' + n + '_mask_paper.png" alt="Semantic mask of view ' + n + '" width="480" height="360"><i class="awcr-scan"></i></div>').join('') +
      '<div class="awcr-elements">' + brace('awcr-element-brace') + '<div class="awcr-legend">' +
      ['eaves', 'pagoda', 'window'].map(name => '<span class="awcr-legend-item" data-element="' + name + '"><i style="background:' + colors[name] + '"></i><span>' + name + '</span></span>').join('') +
      '</div></div></div><p class="awcr-caption">Segment input views into semantic elements.</p></section>';
  }
  function anchor() {
    return '<section id="awcr-anchor" class="awcr-panel awcr-anchor" aria-label="Pass 1: determine the architectural anchor">' +
      '<div class="awcr-anchor-grid"><div class="awcr-token-heading">' + math('\\mathbf{T}^{23}') + '</div><div class="awcr-attention-heading awcr-subhead">Semantic Cross-View Attention</div>' +
      pair(n => math('I_' + n), 'awcr-source-views') + pair(() => arrow('awcr-encode-arrow'), 'awcr-enter-arrows') +
      '<div class="awcr-backbone"><span>3DFM Backbone</span><i class="awcr-encoder-light"></i></div>' +
      pair(() => arrow('awcr-encode-arrow'), 'awcr-exit-arrows') +
      '<div class="awcr-semantic-tokens">' + marks(false, 'view-one') + marks(true, 'view-two') + '</div>' +
      pair(() => arrow('awcr-attend-arrow'), 'awcr-score-arrows') +
      '<div class="awcr-scores">' + ['eaves', 'pagoda', 'window'].map(name => {
        const ink = name === 'pagoda' ? '#a77608' : name === 'window' ? '#bc674a' : colors[name];
        return '<div data-element="' + name + '" class="awcr-score ' + (name === 'pagoda' ? 'is-anchor' : '') + '">' +
          math('\\sum\\mathrm{attn}\\!\\left(\\textcolor{' + colors[name] + '}{\\blacksquare},\\textcolor{' + colors[name] + '}{\\bullet}\\right)=S^{23}(\\textcolor{' + ink + '}{\\text{' + name + '}})') + '</div>';
      }).join('') + '</div>' + brace('awcr-score-brace') +
      '<div class="awcr-max">' + math('\\max') + arrow('awcr-anchor-arrow') + '</div>' +
      '<div class="awcr-anchor-value">' + math('e^{*}=\\textcolor{#a77608}{\\text{pagoda}}') + '</div></div>' +
      '<p class="awcr-caption">Select the anchor with the strongest cross-view attention.</p></section>';
  }
  function update() {
    return '<section id="awcr-intervene" class="awcr-panel awcr-update" aria-label="Pass 2: architecture-centric intervention">' +
      '<div class="awcr-update-grid"><div class="awcr-before-heading">' + math('\\mathbf{T}^{l}') + '</div>' +
      '<div class="awcr-gradient-heading awcr-subhead">Gradient-Ascent Update</div>' +
      '<div class="awcr-after-heading">' + math('\\mathbf{T}^{l+1}') + '</div>' +
      bank(false) + pair(() => arrow('awcr-update-in'), 'awcr-update-in-arrows') +
      '<div class="awcr-gradients">' + [1, 2].map(n => '<div class="awcr-gradient-row">' + vector(n === 2, 'is-latent') +
        math('+\\eta\\,\\nabla S^{l}(\\textcolor{#a77608}{\\text{pagoda}})', 'awcr-gradient-formula') + '</div>').join('') + '</div>' +
      pair(() => arrow('awcr-update-out'), 'awcr-update-out-arrows') + bank(true) + '</div>' +
      '<p class="awcr-caption">Update tokens to strengthen anchor attention.</p></section>';
  }
  const stages = [
    { label: 'Input', id: 'input', start: 0, end: 6800 },
    { label: 'Anchor', id: 'anchor', start: 6800, end: 16000 },
    { label: 'Intervene', id: 'intervene', start: 16000, end: 28000 }
  ];
  root.insertAdjacentHTML('afterbegin', '<div class="awcr-toolbar"><button class="awcr-toggle awcr-control" type="button" aria-label="Play diagram animation">' + icon('play') + '</button>' +
    '<div class="awcr-stages" role="group" aria-label="Animation stages">' + stages.map((stage, i) => {
      const outline = i === 0 ? 'M1 1H286L299 20L286 39H1Z' : 'M1 1H286L299 20L286 39H1L14 20Z';
      return '<button class="awcr-stage" type="button" data-stage="' + i + '" aria-controls="awcr-' + stage.id + '" aria-label="Animate ' + stage.label + '">' +
        '<svg viewBox="0 0 300 40" preserveAspectRatio="none" aria-hidden="true"><path class="awcr-stage-outline" d="' + outline + '"/><path class="awcr-stage-fill" d="' + outline + '"/></svg><span>' + stage.label + '</span></button>';
    }).join('') + '</div><button class="awcr-replay awcr-control" type="button" aria-label="Replay diagram animation">' + icon('replay') + '</button></div>');
  root.querySelector('.awcr-row').innerHTML = inputs() + anchor() + update();
  const toggle = root.querySelector('.awcr-toggle');
  const replay = root.querySelector('.awcr-replay');
  const stageButtons = [...root.querySelectorAll('.awcr-stage')];
  const viewport = root.querySelector('.awcr-viewport');
  const diagram = root.querySelector('.awcr-row');
  const motionLayer = document.createElement('div');
  motionLayer.className = 'awcr-motion-layer';
  motionLayer.setAttribute('aria-hidden', 'true');
  diagram.appendChild(motionLayer);
  // Scale the complete canvas, so its three stages never crop or reflow.
  // Resizing leaves every animation at its existing position on the timeline.
  let fittedWidth = -1;
  function fitDiagram() {
    const width = viewport.getBoundingClientRect().width;
    if (width <= 0 || width === fittedWidth) return;
    fittedWidth = width;
    const scale = width / diagram.offsetWidth;
    diagram.style.setProperty('--diagram-scale', scale);
    viewport.style.height = (diagram.offsetHeight * scale) + 'px';
  }
  fitDiagram();
  const resizeObserver = new ResizeObserver(() => requestAnimationFrame(fitDiagram));
  resizeObserver.observe(viewport);
  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)');
  const DURATION = stages[stages.length - 1].end;
  let animations = [], elapsed = DURATION, startedAt = 0, raf = 0;
  let running = false, ready = false, inView = false, hasStarted = false, resumeWhenVisible = false;
  let currentStage = -1;
  function typeset() {
    if (!window.katex) return false;
    root.querySelectorAll('[data-tex]:not([data-math-ready])').forEach(node => {
      window.katex.render(node.dataset.tex, node, { throwOnError: true, output: 'htmlAndMathml' });
      if (node.closest('.awcr-score')) {
        node.querySelectorAll('.katex-html .mord').forEach(glyph => {
          if (glyph.children.length || !['■', '∙'].includes(glyph.textContent)) return;
          glyph.classList.add('awcr-token-glyph');
          if (glyph.textContent === '∙') glyph.classList.add('is-round');
        });
      }
      node.dataset.mathReady = 'true';
    });
    return true;
  }
  // Every track shares one duration and start time, including its hold frames.
  // Pause, resume and replay therefore keep all elements synchronized.
  function track(selector, frames, at, duration, stagger = 0) {
    const elements = typeof selector === 'string' ? root.querySelectorAll(selector) : [selector];
    elements.forEach((el, index) => {
      const start = at + stagger * index;
      const keys = [{ ...frames[0], offset: 0 }, ...frames.map((frame, i) => ({
        ...frame, offset: (start + (frame.offset == null ? i / (frames.length - 1) : frame.offset) * duration) / DURATION
      })), { ...frames[frames.length - 1], offset: 1 }];
      const animation = el.animate(keys, { duration: DURATION, fill: 'both', easing: 'linear' });
      animation.pause();
      animation.currentTime = 0;
      animations.push(animation);
    });
  }
  function reveal(selector, at, duration = 420, stagger = 0) {
    track(selector, [
      { opacity: 0, transform: 'translateY(4px)', easing: 'cubic-bezier(.2,.8,.2,1)' },
      { opacity: 1, transform: 'translateY(0)' }
    ], at, duration, stagger);
  }
  function draw(selector, at, duration = 420, stagger = 0) {
    track(selector, [{ strokeDasharray: '1', strokeDashoffset: 1 }, { strokeDasharray: '1', strokeDashoffset: 0 }], at, duration, stagger);
  }
  function createTimeline() {
    animations.forEach(a => a.cancel());
    animations = [];
    motionLayer.replaceChildren();
    // Measure the static diagram in its original coordinate system. All paths
    // scale with the canvas, including when the window changes during playback.
    const canvas = diagram.getBoundingClientRect();
    const scale = canvas.width / diagram.offsetWidth;
    const measured = new Map([...diagram.querySelectorAll(
      '.awcr-mask, .awcr-legend-item i, .awcr-semantic-strip i, .awcr-score, .awcr-score .awcr-math, .awcr-anchor-value, .awcr-gradient-formula, .awcr-vector, .awcr-vector i'
    )].map(el => [el, el.getBoundingClientRect()]));
    const point = (el, x = .5, y = .5) => {
      const rect = measured.get(el);
      return [(rect.left - canvas.left + rect.width * x) / scale,
        (rect.top - canvas.top + rect.height * y) / scale];
    };
    const svgNS = 'http://www.w3.org/2000/svg';
    const trails = document.createElementNS(svgNS, 'svg');
    trails.classList.add('awcr-motion-trails');
    trails.setAttribute('viewBox', '0 0 ' + diagram.offsetWidth + ' ' + diagram.offsetHeight);
    motionLayer.appendChild(trails);
    function flight(from, to, options) {
      const { color, at, duration, round = false, size = 9, bend = 0, trail = false, kind } = options;
      const dot = document.createElement('i');
      dot.className = 'awcr-particle' + (round ? ' is-round' : '');
      dot.dataset.motion = kind;
      dot.style.cssText = 'width:' + size + 'px;height:' + size + 'px;background:' + color + ';--particle-color:' + color;
      motionLayer.appendChild(dot);
      const middle = [(from[0] + to[0]) / 2, (from[1] + to[1]) / 2 + bend];
      const transform = (p, s = 1) => 'translate(' + (p[0] - size / 2) + 'px,' + (p[1] - size / 2) + 'px) scale(' + s + ')';
      const frames = [{ opacity: 0, transform: transform(from, .7), offset: 0 }];
      for (let i = 0; i <= 14; i++) {
        const u = i / 14;
        const t = u * u * (3 - 2 * u);
        const p = [0, 1].map(axis => (1 - t) * (1 - t) * from[axis] + 2 * (1 - t) * t * middle[axis] + t * t * to[axis]);
        frames.push({ opacity: 1, transform: transform(p), offset: .08 + .84 * u });
      }
      frames.push({ opacity: 0, transform: transform(to, .7), offset: 1 });
      track(dot, frames, at, duration);
      if (trail) {
        const path = document.createElementNS(svgNS, 'path');
        path.setAttribute('d', 'M' + from.join(',') + ' Q' + middle.join(',') + ' ' + to.join(','));
        path.setAttribute('pathLength', '1');
        path.setAttribute('fill', 'none');
        path.setAttribute('stroke', color);
        path.setAttribute('stroke-width', '1.2');
        path.style.opacity = '0';
        trails.appendChild(path);
        track(path, [
          { opacity: 0, strokeDasharray: '1', strokeDashoffset: 1, offset: 0 },
          { opacity: .5, strokeDasharray: '1', strokeDashoffset: .95, offset: .1 },
          { opacity: .35, strokeDasharray: '1', strokeDashoffset: 0, offset: .8 },
          { opacity: 0, strokeDasharray: '1', strokeDashoffset: 0, offset: 1 }
        ], at, duration);
      }
    }
    function glow(selector, color, at, duration = 1100) {
      track(selector, [
        { boxShadow: '0 0 0 0 ' + color + '00' },
        { boxShadow: '0 0 0 3px ' + color + '55, 0 0 12px ' + color + '44', offset: .45 },
        { boxShadow: '0 0 0 0 ' + color + '00' }
      ], at, duration);
    }
    reveal('.awcr-inputs .awcr-caption', 0, 500);
    reveal('.awcr-anchor .awcr-caption', 6800, 500);
    reveal('.awcr-update .awcr-caption', 16000, 500);

    // Input: views, masks, then matching colors travel into the element legend.
    reveal('.awcr-view, .awcr-input-label', 200, 900, 180);
    reveal('.awcr-mask', 900, 500, 250);
    track('.awcr-mask-image', [{ clipPath: 'inset(0 0 100% 0)' }, { clipPath: 'inset(0 0 0% 0)' }], 1700, 1450, 300);
    const masks = [...root.querySelectorAll('.awcr-mask')];
    masks.forEach((mask, i) => {
      track(mask.querySelector('.awcr-scan'), [
        { opacity: 0, transform: 'translateY(0)' },
        { opacity: .85, transform: 'translateY(0)', offset: .03 },
        { opacity: .85, transform: 'translateY(' + (mask.clientHeight - 2) + 'px)', offset: .96 },
        { opacity: 0, transform: 'translateY(' + mask.clientHeight + 'px)' }
      ], 1700 + i * 300, 1450);
    });
    reveal('.awcr-set-label', 3300, 600);
    draw('.awcr-element-brace path', 3400, 900);
    const maskPoints = [
      { eaves: [.483, .328], pagoda: [.542, .583], window: [.469, .422] },
      { eaves: [.504, .422], pagoda: [.494, .581], window: [.565, .367] }
    ];
    ['eaves', 'pagoda', 'window'].forEach((name, i) => {
      const item = root.querySelector('.awcr-legend-item[data-element="' + name + '"]');
      const chip = item.querySelector('i');
      const at = 3400 + i * 520;
      masks.forEach((mask, view) => flight(point(mask, ...maskPoints[view][name]), point(chip), {
        color: colors[name], at: at + view * 180, duration: 1400,
        round: view === 1, bend: view ? 12 : -12, trail: true, kind: 'semantic'
      }));
      reveal(item, at + 1200, 650);
      glow(chip, colors[name], at + 1380, 900);
    });

    // Anchor: let every score settle before drawing attention to the winner.
    reveal('.awcr-source-views', 6800, 650);
    draw('.awcr-enter-arrows path', 7100, 650, 160);
    reveal('.awcr-backbone', 7500, 650);
    track('.awcr-encoder-light', [
      { opacity: 0, transform: 'translateY(-110%)' },
      { opacity: .8, transform: 'translateY(-90%)', offset: .15 },
      { opacity: .7, transform: 'translateY(90%)', offset: .85 },
      { opacity: 0, transform: 'translateY(110%)' }
    ], 8000, 1400);
    draw('.awcr-exit-arrows path', 8500, 650, 160);
    reveal('.awcr-token-heading', 8600, 650);
    track('.awcr-semantic-strip i', [
      { opacity: 0, transform: 'translateX(-9px) scale(.5)', easing: 'cubic-bezier(.16,1,.3,1)' },
      { opacity: 1, transform: 'translateX(0) scale(1)' }
    ], 8700, 650, 65);
    draw('.awcr-score-arrows path', 10000, 600, 160);
    reveal('.awcr-attention-heading', 10000, 600);
    reveal('.awcr-score', 10500, 750, 700);
    track('.awcr-score.is-anchor', [
      { backgroundColor: 'rgba(255,247,221,0)', borderColor: 'rgba(232,207,139,0)' },
      { backgroundColor: '#fff7dd', borderColor: '#e8cf8b' }
    ], 13000, 1000);
    // Use emphasis, not invented numeric scores, to show the argmax choice.
    track('.awcr-score:not(.is-anchor) .awcr-math', [
      { opacity: 1 }, { opacity: .45, offset: .25 },
      { opacity: .45, offset: .75 }, { opacity: 1 }
    ], 13000, 2500);
    const winningFormula = root.querySelector('.awcr-score.is-anchor .awcr-math');
    [...root.querySelectorAll('.awcr-semantic-strip [data-element="pagoda"]')].forEach((token, view) => {
      glow(token, colors.pagoda, 12800 + view * 180, 1800);
      flight(point(token), point(winningFormula, view ? .46 : .34, .5), {
        color: colors.pagoda, at: 13250 + view * 200, duration: 1400,
        round: view === 1, bend: view ? 18 : -18, trail: true, kind: 'anchor'
      });
    });
    glow('.awcr-score.is-anchor', colors.pagoda, 14100, 1800);
    draw('.awcr-score-brace path', 12800, 850);
    track('.awcr-max .awcr-math', [{ opacity: 0 }, { opacity: 1 }], 14200, 500);
    draw('.awcr-anchor-arrow path', 14700, 650);
    flight(point(root.querySelector('.awcr-score.is-anchor'), .96, .5),
      point(root.querySelector('.awcr-anchor-value'), .18, .5), {
        color: colors.pagoda, at: 14800, duration: 950, size: 8, kind: 'anchor'
      });
    track('.awcr-anchor-value', [
      { opacity: 0, transform: 'scale(.9)', easing: 'cubic-bezier(.16,1,.3,1)' },
      { opacity: 1, transform: 'scale(1)' }
    ], 15200, 700);

    // Intervene: show gray output slots first. Gold gradient packets enter the
    // latent tokens, then travel into the output cells one by one.
    reveal('.awcr-before-heading, .awcr-gradient-heading, .awcr-after-heading', 16000, 650, 140);
    reveal('.awcr-bank.is-input .awcr-vector', 16000, 700, 140);
    draw('.awcr-update-in path', 16800, 700, 400);
    reveal('.awcr-vector.is-latent', 16900, 650, 400);
    reveal('.awcr-gradient-formula', 17700, 850, 400);
    reveal('.awcr-bank.is-updated .awcr-vector', 18300, 600, 120);
    draw('.awcr-update-out path', 22000, 800, 1500);
    const outputRows = [...root.querySelectorAll('.awcr-bank.is-updated .awcr-vector-row')];
    [...root.querySelectorAll('.awcr-gradient-row')].forEach((row, view) => {
      const formula = row.querySelector('.awcr-gradient-formula');
      const latentCells = [...row.querySelectorAll('.awcr-vector i')];
      glow(formula, colors.pagoda, 18800 + view * 1500, 2000);
      latentCells.forEach((cell, i) => {
        const at = 19000 + view * 1500 + i * 140;
        flight(point(formula, .8, .5), point(cell), {
          color: colors.pagoda, at, duration: 1100, size: 10,
          round: view === 1, bend: (i - 2) * 8, kind: 'gradient'
        });
        track(cell, [
          { backgroundColor: '#aeb9c8', boxShadow: '0 0 0 0 #ffb90f00' },
          { backgroundColor: '#ffb90f', boxShadow: '0 0 0 3px #ffb90f44', offset: .12 },
          { backgroundColor: '#ffb90f', boxShadow: '0 0 0 0 #ffb90f00', offset: .65 },
          { backgroundColor: '#aeb9c8', boxShadow: '0 0 0 0 #ffb90f00' }
        ], at + 1100, 4500);
      });
      [...outputRows[view].querySelectorAll('.awcr-vector')].forEach((vector, column) => {
        [...vector.querySelectorAll('i')].forEach((cell, i) => {
          const at = 22000 + view * 1500 + column * 180 + i * 140;
          const from = point(latentCells[i]);
          const to = point(cell);
          const lane = point(formula)[1] + (view ? -32 : 32);
          flight(from, to, {
            color: colors.pagoda, at, duration: 900, size: 9,
            round: view === 1, bend: 2 * (lane - (from[1] + to[1]) / 2), kind: 'updated'
          });
          track(cell, [
            { backgroundColor: '#aeb9c8', transform: 'scale(1)', boxShadow: '0 0 0 0 #ffb90f00' },
            { backgroundColor: '#efb72e', transform: 'scale(1.28)', boxShadow: '0 0 0 2px #ffb90f55', offset: .55 },
            { backgroundColor: '#efb72e', transform: 'scale(1)', boxShadow: '0 0 0 0 #ffb90f00' }
          ], at + 900, 650);
        });
        glow(vector, colors.pagoda, 23800 + view * 1500 + column * 120, 1200);
      });
    });
  }
  function stageProgress() {
    const active = elapsed < stages[1].start ? 0 : elapsed < stages[2].start ? 1 : 2;
    if (active !== currentStage) {
      currentStage = active;
      root.dataset.stage = stages[active].id;
      stageButtons.forEach((button, i) => {
        if (i === active) button.setAttribute('aria-current', 'step');
        else button.removeAttribute('aria-current');
      });
    }
    stageButtons.forEach((button, i) => {
      const progress = Math.max(0, Math.min(1, (elapsed - stages[i].start) / (stages[i].end - stages[i].start)));
      button.style.setProperty('--stage-progress', (progress * 100).toFixed(1) + '%');
      button.classList.toggle('is-complete', progress === 1);
    });
  }
  function controls() {
    toggle.innerHTML = icon(running ? 'pause' : 'play');
    toggle.setAttribute('aria-label', running ? 'Pause diagram animation' : elapsed >= DURATION ? 'Play diagram animation' : 'Resume diagram animation');
    toggle.setAttribute('aria-pressed', String(running));
    root.dataset.state = running ? 'playing' : elapsed >= DURATION ? 'complete' : 'paused';
    stageProgress();
  }
  function stop(manual = true) {
    if (running) elapsed = Math.min(DURATION, performance.now() - startedAt);
    running = false;
    cancelAnimationFrame(raf);
    animations.forEach(a => { a.pause(); a.currentTime = elapsed; });
    if (manual) resumeWhenVisible = false;
    controls();
  }
  function tick() {
    if (!running) return;
    elapsed = Math.min(DURATION, performance.now() - startedAt);
    stageProgress();
    const phase = elapsed < 6800 ? 'inputs' : elapsed < 13000 ? 'attention' : elapsed < 16000 ? 'anchor' : elapsed < 22000 ? 'gradient' : 'updated';
    if (root.dataset.phase !== phase) root.dataset.phase = phase;
    if (elapsed >= DURATION) { stop(); return; }
    raf = requestAnimationFrame(tick);
  }
  function play(fromStart = false) {
    if (!ready || reduced.matches) return;
    hasStarted = true;
    if (!animations.length || fromStart || elapsed >= DURATION) { createTimeline(); elapsed = 0; }
    cancelAnimationFrame(raf);
    const timelineStart = document.timeline.currentTime - elapsed;
    animations.forEach(a => { a.currentTime = elapsed; a.play(); a.startTime = timelineStart; });
    running = true;
    resumeWhenVisible = false;
    startedAt = performance.now() - elapsed;
    controls();
    raf = requestAnimationFrame(tick);
  }
  function seek(ms) {
    stop();
    if (!animations.length) createTimeline();
    elapsed = Math.max(0, Math.min(DURATION, ms));
    animations.forEach(a => { a.pause(); a.currentTime = elapsed; });
    controls();
  }
  function staticView() {
    stop();
    animations.forEach(a => a.cancel());
    animations = [];
    motionLayer.replaceChildren();
    elapsed = DURATION;
    controls();
  }
  toggle.addEventListener('click', () => running ? stop() : play());
  replay.addEventListener('click', () => play(true));
  function selectStage(index) {
    if (!ready) return;
    hasStarted = true;
    if (reduced.matches) {
      staticView();
      currentStage = index;
      root.dataset.stage = stages[index].id;
      stageButtons.forEach((button, i) => {
        if (i === index) button.setAttribute('aria-current', 'step');
        else button.removeAttribute('aria-current');
      });
    } else {
      seek(stages[index].start);
      play();
    }
  }
  stageButtons.forEach((button, index) => {
    button.addEventListener('click', () => selectStage(index));
    button.addEventListener('keydown', event => {
      let target = index;
      if (event.key === 'ArrowRight') target = Math.min(index + 1, stages.length - 1);
      else if (event.key === 'ArrowLeft') target = Math.max(index - 1, 0);
      else if (event.key === 'Home') target = 0;
      else if (event.key === 'End') target = stages.length - 1;
      else return;
      event.preventDefault();
      stageButtons[target].focus();
      selectStage(target);
    });
  });
  function autoPlay() {
    if (ready && inView && !hasStarted && !reduced.matches && !document.hidden) play(true);
  }
  const observer = new IntersectionObserver(entries => {
    inView = entries[0].intersectionRatio >= .5;
    if (!inView && running) { stop(false); resumeWhenVisible = true; }
    else if (inView && resumeWhenVisible && !document.hidden) play();
    else autoPlay();
  }, { threshold: [0, .5] });
  observer.observe(root);
  document.addEventListener('visibilitychange', () => {
    if (document.hidden && running) { stop(false); resumeWhenVisible = true; }
    else if (!document.hidden && inView && resumeWhenVisible) play();
  });
  reduced.addEventListener('change', () => { if (reduced.matches) staticView(); });
  root.archAttend = { play, pause: stop, seek, finish: () => seek(DURATION), duration: DURATION };
  controls();
  let checkingMath = false;
  const mathPoll = window.setInterval(async () => {
    if (checkingMath || !window.katex) return;
    checkingMath = true;
    try {
      typeset();
      await document.fonts.ready;
      ready = true;
      root.dataset.ready = 'true';
      window.clearInterval(mathPoll);
      autoPlay();
    } catch (error) {
      console.error('ArchAttend LaTeX:', error);
      window.clearInterval(mathPoll);
    }
  }, 150);
  window.addEventListener('pagehide', () => { window.clearInterval(mathPoll); stop(); });
})();
