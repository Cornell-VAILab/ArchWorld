// Interactive version of Figure 7: DA3 median normalized Chamfer distance per
// bucket, before and after ArchAttend. Colours, ordering and the gap brackets
// follow the printed figure; hovering a bucket is what the page adds.
(function () {
  const DATA = {
    region: [
      { label: 'N. Amer.', n: 44, before: 18.83, after: 12.82 },
      { label: 'E. Asia',  n: 43, before: 15.52, after: 13.54 },
      { label: 'WC Asia',  n: 50, before: 11.08, after: 8.53 },
      { label: 'Africa',   n: 37, before: 10.41, after: 9.73 },
      { label: 'S. Asia',  n: 53, before: 9.99,  after: 8.29 },
      { label: 'Europe',   n: 51, before: 8.73,  after: 9.85 },
      { label: 'SE Asia',  n: 50, before: 7.88,  after: 6.59 },
      { label: 'L. Amer.', n: 50, before: 7.57,  after: 6.81 },
    ],
    arch: [
      { label: 'Tower',    n: 30,  before: 21.04, after: 16.29 },
      { label: 'Monum.',   n: 32,  before: 12.42, after: 9.41 },
      { label: 'Temple',   n: 38,  before: 10.80, after: 11.00 },
      { label: 'Tomb',     n: 15,  before: 9.53,  after: 9.03 },
      { label: 'Church',   n: 114, before: 9.20,  after: 7.82 },
      { label: 'P. Bldg.', n: 38,  before: 9.08,  after: 9.32 },
      { label: 'Mosque',   n: 27,  before: 8.24,  after: 7.90 },
      { label: 'Arch',     n: 6,   before: 7.29,  after: 7.15 },
    ],
  };
  const OVERALL = { before: 10.36, after: 8.92 };
  // Quoted from the paper rather than recomputed: differencing the rounded bar
  // labels loses a hundredth (13.54 - 6.59 = 6.95, the unrounded gap is 6.96).
  const GAPS = {
    region: { before: 11.26, after: 6.96 },
    arch: { before: 13.75, after: 9.14 },
  };
  // sampled from the figure
  const C = {
    before: { fill: '#ff6060', edge: '#ff4444' },
    after: { fill: '#ffa726', edge: '#ff9800' },
    grid: '#eceff2', axis: '#9ca3af', ink: '#1f2937', muted: '#6b7280',
  };
  const NS = 'http://www.w3.org/2000/svg';

  const W = 920, H = 430;
  const PAD = { t: 22, r: 148, b: 58, l: 56 };
  const plotW = W - PAD.l - PAD.r, plotH = H - PAD.t - PAD.b;

  const el = (name, attrs = {}, text) => {
    const n = document.createElementNS(NS, name);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (text != null) n.textContent = text;
    return n;
  };
  const fmt = (v) => v.toFixed(2);
  const pct = (b, a) => `${a < b ? '−' : '+'}${Math.abs((a - b) / b * 100).toFixed(1)}%`;

  function render(root, key) {
    const rows = DATA[key];
    const yMax = Math.max(...rows.flatMap((r) => [r.before, r.after])) * 1.12;
    const y = (v) => PAD.t + plotH - (v / yMax) * plotH;
    const band = plotW / rows.length;
    const bw = Math.min(30, band * 0.33);

    root.textContent = '';
    const svg = el('svg', {
      viewBox: `0 0 ${W} ${H}`, class: 'aw-ba-svg', role: 'img',
      'aria-label': `Median normalized Chamfer distance by ${key === 'region' ? 'geographic region' : 'architecture type'}, before and after ArchAttend`,
    });

    const step = yMax > 18 ? 5 : 2;
    for (let v = 0; v <= yMax; v += step) {
      svg.appendChild(el('line', { x1: PAD.l, x2: PAD.l + plotW, y1: y(v), y2: y(v), stroke: C.grid, 'stroke-width': 1 }));
      svg.appendChild(el('text', { x: PAD.l - 12, y: y(v) + 4, 'text-anchor': 'end', class: 'aw-ba-tick' }, v));
    }
    svg.appendChild(el('line', { x1: PAD.l, x2: PAD.l + plotW, y1: y(0), y2: y(0), stroke: C.axis, 'stroke-width': 1 }));
    svg.appendChild(el('text', {
      x: 16, y: PAD.t + plotH / 2, 'text-anchor': 'middle', class: 'aw-ba-axis',
      transform: `rotate(-90 16 ${PAD.t + plotH / 2})`,
    }, 'Median CD̂'));

    const tip = root.parentElement.querySelector('.aw-ba-tip');
    const show = (cx, cy, r) => {
      tip.innerHTML = `<strong>${r.label}</strong> <span class="n">n=${r.n}</span><br>` +
        `before <b>${fmt(r.before)}</b> → after <b>${fmt(r.after)}</b> ` +
        `<span class="${r.after < r.before ? 'good' : 'bad'}">${pct(r.before, r.after)}</span>`;
      tip.classList.add('is-on');
      const box = root.getBoundingClientRect();
      tip.style.left = `${cx - box.left}px`;
      tip.style.top = `${cy - box.top - 12}px`;
    };
    const hide = () => tip.classList.remove('is-on');

    rows.forEach((r, i) => {
      const cx = PAD.l + band * (i + 0.5);
      const g = el('g', { class: 'aw-ba-group', tabindex: '0',
        'aria-label': `${r.label}: before ${fmt(r.before)}, after ${fmt(r.after)}, ${pct(r.before, r.after)}, n equals ${r.n}` });
      g.appendChild(el('rect', { x: PAD.l + band * i, y: PAD.t, width: band, height: plotH, fill: 'transparent' }));
      [['before', -1], ['after', 1]].forEach(([which, side]) => {
        const v = r[which];
        const x = cx + side * bw * 0.54 - bw / 2;
        g.appendChild(el('rect', { x, y: y(v), width: bw, height: y(0) - y(v), rx: 1,
          fill: C[which].fill, stroke: C[which].edge, 'stroke-width': 1, class: `aw-ba-bar is-${which}` }));
        g.appendChild(el('text', { x: x + bw / 2, y: y(v) - 7, 'text-anchor': 'middle', class: 'aw-ba-val' }, fmt(v)));
      });
      g.appendChild(el('text', { x: cx, y: y(0) + 22, 'text-anchor': 'middle', class: 'aw-ba-cat' }, r.label));
      g.appendChild(el('text', { x: cx, y: y(0) + 38, 'text-anchor': 'middle', class: 'aw-ba-n' }, `n=${r.n}`));
      const at = (e) => show(e.clientX, e.clientY, r);
      g.addEventListener('pointermove', at);
      g.addEventListener('pointerleave', hide);
      g.addEventListener('focus', () => {
        const b = g.getBoundingClientRect();
        show(b.left + b.width / 2, b.top + 40, r);
      });
      g.addEventListener('blur', hide);
      svg.appendChild(g);
    });

    // worst-to-best spread, with the figure's dotted leaders back to the bars
    [['before', 54], ['after', 110]].forEach(([which, dx]) => {
      const vals = rows.map((r) => r[which]);
      const lo = Math.min(...vals), hi = Math.max(...vals);
      const x = PAD.l + plotW + dx;
      const g = el('g', { class: 'aw-ba-bracket' });
      [hi, lo].forEach((v) => {
        const i = vals.indexOf(v);
        const from = PAD.l + band * (i + 0.5) + (which === 'before' ? -1 : 1) * bw * 1.04;
        g.appendChild(el('line', { x1: from, x2: x, y1: y(v), y2: y(v),
          stroke: C[which].edge, 'stroke-width': 1, 'stroke-dasharray': '2 3', opacity: 0.75 }));
        g.appendChild(el('line', { x1: x - 6, x2: x + 6, y1: y(v), y2: y(v), stroke: C[which].edge, 'stroke-width': 2.5 }));
      });
      g.appendChild(el('line', { x1: x, x2: x, y1: y(hi), y2: y(lo), stroke: C[which].edge, 'stroke-width': 2.5 }));
      g.appendChild(el('text', {
        x: x - 8, y: (y(lo) + y(hi)) / 2, fill: C[which].edge, class: 'aw-ba-gap',
        transform: `rotate(-90 ${x - 8} ${(y(lo) + y(hi)) / 2})`, 'text-anchor': 'middle',
      }, `gap = ${fmt(GAPS[key][which])}`));
      svg.appendChild(g);
    });

    root.appendChild(svg);
  }

  function init() {
    const root = document.getElementById('before-after-chart');
    if (!root) return;
    const toggle = document.getElementById('ba-toggle');
    const summary = document.getElementById('ba-summary');
    if (summary) {
      summary.innerHTML = `Across all scenes the median falls from <b>${fmt(OVERALL.before)}</b> ` +
        `to <b>${fmt(OVERALL.after)}</b> <span class="good">${pct(OVERALL.before, OVERALL.after)}</span>. ` +
        `Hover a bucket for its numbers.`;
    }
    render(root, 'region');
    toggle?.addEventListener('click', (e) => {
      const btn = e.target.closest('button[data-bucket]');
      if (!btn) return;
      toggle.querySelectorAll('button').forEach((b) => b.classList.toggle('is-active', b === btn));
      render(root, btn.dataset.bucket);
    });
  }
  document.readyState === 'loading' ? document.addEventListener('DOMContentLoaded', init) : init();
})();
