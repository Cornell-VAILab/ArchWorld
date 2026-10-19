// Main and supplementary intervention figures: DA3, VGGT, and pi3.
// Values reproduced with the paper's figure-generation scripts (gen_main_da3_norm.py
// and gen_supp_norm_two.py): baseline averaged over runs, best step per scene.
// Keep full precision so gaps and percentage changes agree with the paper.
(function () {
  const MODELS = {
    da3: {
      label: "DA3",
      n: 378,
      overall: {"before": 10.360402199922381, "after": 8.919853259020199},
      gaps: {"region": {"before": 11.259743223396312, "after": 6.955552511745884}, "arch": {"before": 13.74952767919702, "after": 9.141169410745443}},
      region: [
        {"label": "N. Amer.", "n": 44, "before": 18.82806992358137, "after": 12.815349814089892},
        {"label": "E. Asia", "n": 43, "before": 15.523871459479283, "after": 13.543441706743971},
        {"label": "WC Asia", "n": 50, "before": 11.080354415803255, "after": 8.526894777832634},
        {"label": "Africa", "n": 37, "before": 10.409027936817477, "after": 9.731527934641878},
        {"label": "S. Asia", "n": 53, "before": 9.993283729413797, "after": 8.293325145742553},
        {"label": "Europe", "n": 51, "before": 8.73127388707949, "after": 9.84670739096559},
        {"label": "SE Asia", "n": 50, "before": 7.881327439418965, "after": 6.587889194998088},
        {"label": "L. Amer.", "n": 50, "before": 7.568326700185059, "after": 6.809980079901202},
      ],
      arch: [
        {"label": "Tower", "n": 30, "before": 21.040645180223958, "after": 16.293416670333094},
        {"label": "Monum.", "n": 32, "before": 12.422286212117726, "after": 9.410561313896789},
        {"label": "Temple", "n": 38, "before": 10.802479603563176, "after": 11.00364141782666},
        {"label": "Tomb", "n": 15, "before": 9.534415079417842, "after": 9.028631773538866},
        {"label": "Church", "n": 114, "before": 9.196824390011157, "after": 7.822441495493381},
        {"label": "P. Bldg.", "n": 38, "before": 9.076243282526814, "after": 9.320005687233802},
        {"label": "Mosque", "n": 27, "before": 8.235759984376847, "after": 7.9047459929766655},
        {"label": "Arch", "n": 6, "before": 7.29111750102694, "after": 7.152247259587652},
      ],
    },
    vggt: {
      label: "VGGT",
      n: 380,
      overall: {"before": 3.2793412074999564, "after": 3.1195948564318288},
      gaps: {"region": {"before": 4.549568346151872, "after": 3.8718269829708944}, "arch": {"before": 5.413076059950473, "after": 5.234919707145142}},
      region: [
        {"label": "E. Asia", "n": 43, "before": 6.634427756955592, "after": 5.874782577690411},
        {"label": "N. Amer.", "n": 44, "before": 4.354185526780993, "after": 3.9971577001288034},
        {"label": "S. Asia", "n": 53, "before": 3.762856099139863, "after": 3.6931974747864236},
        {"label": "Africa", "n": 38, "before": 3.7520160750884988, "after": 3.654993884698426},
        {"label": "Europe", "n": 51, "before": 3.033201258554099, "after": 2.9110981795891795},
        {"label": "SE Asia", "n": 50, "before": 2.847091940369939, "after": 2.7200017473093396},
        {"label": "L. Amer.", "n": 50, "before": 2.1457144519550964, "after": 2.0029555947195163},
        {"label": "WC Asia", "n": 51, "before": 2.0848594108037193, "after": 2.043839148153338},
      ],
      arch: [
        {"label": "Tower", "n": 30, "before": 7.6876623239677775, "after": 7.4693372707203505},
        {"label": "Monum.", "n": 32, "before": 4.878073231303134, "after": 4.801185320242247},
        {"label": "Tomb", "n": 15, "before": 4.498429194714691, "after": 4.026421560423903},
        {"label": "Temple", "n": 38, "before": 3.7447902497298458, "after": 3.242008409625582},
        {"label": "Mosque", "n": 27, "before": 3.5107139754005106, "after": 3.359915246749164},
        {"label": "P. Bldg.", "n": 38, "before": 3.2162907748402607, "after": 3.0276590249570505},
        {"label": "Arch", "n": 6, "before": 3.1924347693050086, "after": 3.140444901305024},
        {"label": "Church", "n": 115, "before": 2.2745862640173042, "after": 2.234417563575209},
      ],
    },
    pi3: {
      label: "π³",
      n: 380,
      overall: {"before": 1.7533699967480574, "after": 1.7064059991531146},
      gaps: {"region": {"before": 1.4030668577752956, "after": 1.3446438143784767}, "arch": {"before": 1.4978414161041087, "after": 1.348821152239311}},
      region: [
        {"label": "E. Asia", "n": 43, "before": 2.7469960247985306, "after": 2.673305950750901},
        {"label": "N. Amer.", "n": 44, "before": 2.2657891732016973, "after": 2.240925721897367},
        {"label": "Africa", "n": 38, "before": 2.241532287551519, "after": 2.103851295243147},
        {"label": "S. Asia", "n": 53, "before": 1.9703735437497154, "after": 1.8052363779131022},
        {"label": "Europe", "n": 51, "before": 1.6813049146749541, "after": 1.6123509505815237},
        {"label": "L. Amer.", "n": 50, "before": 1.4806617040657812, "after": 1.4439500264570326},
        {"label": "WC Asia", "n": 51, "before": 1.4304563640139678, "after": 1.411322497053889},
        {"label": "SE Asia", "n": 50, "before": 1.343929167023235, "after": 1.3286621363724245},
      ],
      arch: [
        {"label": "Tower", "n": 30, "before": 2.9282977801180765, "after": 2.749798631467364},
        {"label": "Monum.", "n": 32, "before": 2.4003064290402314, "after": 2.2689615741650098},
        {"label": "Temple", "n": 38, "before": 2.155746801723953, "after": 1.9618668619795514},
        {"label": "Arch", "n": 6, "before": 2.1291860111090024, "after": 1.9594923710546113},
        {"label": "Tomb", "n": 15, "before": 1.9703735437497154, "after": 1.8052363779131022},
        {"label": "Mosque", "n": 27, "before": 1.910159125333955, "after": 1.818113748194183},
        {"label": "P. Bldg.", "n": 38, "before": 1.6630027386294497, "after": 1.577893560443648},
        {"label": "Church", "n": 115, "before": 1.4304563640139678, "after": 1.400977479228053},
      ],
    },
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

  function render(root, model, key) {
    const rows = model[key];
    const maxValue = Math.max(...rows.flatMap((r) => [r.before, r.after]));
    const step = [0.25, 0.5, 1, 2, 5, 10, 20, 50].find((s) => s >= maxValue * 1.12 / 5) || 50;
    const yMax = Math.ceil(maxValue * 1.12 / step) * step;
    const y = (v) => PAD.t + plotH - (v / yMax) * plotH;
    const band = plotW / rows.length;
    const bw = Math.min(30, band * 0.33);

    root.textContent = '';
    const svg = el('svg', {
      viewBox: `0 0 ${W} ${H}`, class: 'aw-ba-svg', role: 'img',
      'aria-label': `${model.label}: median normalized Chamfer distance in thousandths by ${key === 'region' ? 'geographic region' : 'architecture type'}, before and after ArchAttend`,
    });

    for (let v = 0; v <= yMax; v += step) {
      svg.appendChild(el('line', { x1: PAD.l, x2: PAD.l + plotW, y1: y(v), y2: y(v), stroke: C.grid, 'stroke-width': 1 }));
      svg.appendChild(el('text', { x: PAD.l - 12, y: y(v) + 4, 'text-anchor': 'end', class: 'aw-ba-tick' }, v));
    }
    svg.appendChild(el('line', { x1: PAD.l, x2: PAD.l + plotW, y1: y(0), y2: y(0), stroke: C.axis, 'stroke-width': 1 }));
    svg.appendChild(el('text', {
      x: 16, y: PAD.t + plotH / 2, 'text-anchor': 'middle', class: 'aw-ba-axis',
      transform: `rotate(-90 16 ${PAD.t + plotH / 2})`,
    }, 'Median error'));

    const tip = root.parentElement.querySelector('.aw-ba-tip');
    tip.classList.remove('is-on');
    const show = (cx, cy, r) => {
      tip.innerHTML = `<strong>${r.label}</strong> <span class="n">n=${r.n}</span><br>` +
        `before <b>${fmt(r.before)}</b> → after <b>${fmt(r.after)}</b> ` +
        `<span class="${r.after < r.before ? 'good' : 'bad'}">${pct(r.before, r.after)}</span>`;
      tip.classList.add('is-on');
      const wrap = root.parentElement;
      const box = wrap.getBoundingClientRect();
      const half = tip.offsetWidth / 2;
      const left = Math.max(half + 8, Math.min(wrap.clientWidth - half - 8, cx - box.left));
      tip.style.left = `${left + wrap.scrollLeft}px`;
      tip.style.top = `${Math.max(tip.offsetHeight + 12, cy - box.top) + wrap.scrollTop - 12}px`;
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
      }, `gap = ${fmt(model.gaps[key][which])}`));
      svg.appendChild(g);
    });

    root.appendChild(svg);
  }

  function init() {
    const panel = document.getElementById('ba-charts');
    if (!panel) return;
    const modelTabs = document.getElementById('ba-model-tabs');
    const tabs = [...modelTabs.querySelectorAll('[role="tab"]')];
    const toggle = document.getElementById('ba-toggle');
    let modelKey = 'da3';
    let bucketKey = 'region';

    function update() {
      tabs.forEach((tab) => {
        const active = tab.dataset.model === modelKey;
        tab.classList.toggle('is-active', active);
        tab.setAttribute('aria-selected', String(active));
        tab.tabIndex = active ? 0 : -1;
        if (active) panel.setAttribute('aria-labelledby', tab.id);
      });
      toggle.querySelectorAll('button').forEach((button) => {
        const active = button.dataset.bucket === bucketKey;
        button.classList.toggle('is-active', active);
        button.setAttribute('aria-pressed', String(active));
      });
      const model = MODELS[modelKey];
      render(panel.querySelector('.aw-ba-plot'), model, bucketKey);
      const gap = model.gaps[bucketKey];
      panel.querySelector('.aw-ba-summary').innerHTML =
        `${bucketKey === 'region' ? 'Regional' : 'Architectural'} disparity: ` +
        `<span class="aw-ba-before">${fmt(gap.before)}</span> → <span class="aw-ba-after">${fmt(gap.after)}</span> ` +
        `<span class="good">(${pct(gap.before, gap.after)})</span>`;
    }

    update();
    modelTabs.addEventListener('click', (e) => {
      const tab = e.target.closest('[role="tab"]');
      if (!tab || tab.dataset.model === modelKey) return;
      modelKey = tab.dataset.model;
      update();
    });
    modelTabs.addEventListener('keydown', (e) => {
      const index = tabs.indexOf(e.target.closest('[role="tab"]'));
      if (index < 0) return;
      const next = { ArrowRight: (index + 1) % tabs.length, ArrowLeft: (index + tabs.length - 1) % tabs.length,
        Home: 0, End: tabs.length - 1 }[e.key];
      if (next === undefined) return;
      e.preventDefault();
      modelKey = tabs[next].dataset.model;
      update();
      tabs[next].focus();
    });
    toggle.addEventListener('click', (e) => {
      const btn = e.target.closest('button[data-bucket]');
      if (!btn || btn.dataset.bucket === bucketKey) return;
      bucketKey = btn.dataset.bucket;
      update();
    });
  }
  document.readyState === 'loading' ? document.addEventListener('DOMContentLoaded', init) : init();
})();
