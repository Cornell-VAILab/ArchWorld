// Interactive version of Table 2: 3DFM median error by geographic region.
// Values are medians with interquartile ranges; dCD is scaled by 1e3.
(function () {
  const REGIONS = {
    sea: { name: 'SE Asia', n: 50 },
    wca: { name: 'WC Asia', n: 51 },
    sa: { name: 'S Asia', n: 53 },
    la: { name: 'L Amer', n: 50 },
    af: { name: 'Africa', n: 38 },
    eu: { name: 'Europe', n: 51 },
    ea: { name: 'E Asia', n: 43 },
    na: { name: 'N Amer', n: 44 },
  };

  // Column order follows pi3, the best-performing model.
  const DATA = {
    cd: {
      order: ['sea', 'wca', 'sa', 'la', 'af', 'eu', 'ea', 'na'],
      rows: {
        'DA3':  { sea: [0.39, 1.15], wca: [0.72, 1.54], sa: [0.86, 1.17], la: [0.77, 1.19], af: [0.87, 1.55], eu: [0.99, 1.99], ea: [0.93, 1.53], na: [2.34, 5.01] },
        'VGGT': { sea: [0.14, 0.27], wca: [0.18, 0.47], sa: [0.26, 0.53], la: [0.22, 0.31], af: [0.26, 0.38], eu: [0.35, 0.59], ea: [0.48, 1.78], na: [0.65, 0.93] },
        'π<sup>3</sup>': { sea: [0.08, 0.08], wca: [0.13, 0.23], sa: [0.14, 0.21], la: [0.14, 0.10], af: [0.17, 0.29], eu: [0.19, 0.25], ea: [0.25, 0.37], na: [0.36, 0.74] },
      },
      caption: 'Median CD in meters per region, with the interquartile range below; n is the number of scenes. Columns are ordered by π<sup>3</sup>.',
    },
    dcd: {
      order: ['sea', 'wca', 'la', 'eu', 'sa', 'af', 'na', 'ea'],
      rows: {
        'DA3':  { sea: [8.02, 10.22], wca: [11.03, 11.63], la: [7.57, 20.95], eu: [8.39, 10.89], sa: [9.53, 21.71], af: [10.61, 16.55], na: [17.61, 31.43], ea: [15.66, 18.60] },
        'VGGT': { sea: [2.85, 3.12], wca: [2.08, 2.94], la: [2.15, 2.94], eu: [3.03, 4.30], sa: [3.76, 4.66], af: [3.75, 5.16], na: [4.35, 8.52], ea: [6.63, 12.32] },
        'π<sup>3</sup>': { sea: [1.34, 1.20], wca: [1.43, 1.34], la: [1.48, 0.80], eu: [1.68, 1.42], sa: [1.97, 1.92], af: [2.24, 3.92], na: [2.27, 3.76], ea: [2.75, 4.76] },
      },
      caption: 'Median \\(\\widehat{\\mathrm{CD}}\\) (×10<sup>3</sup>) per region, with the interquartile range below; n is the number of scenes. Columns are ordered by π<sup>3</sup>; ▲/▼ mark regions ranked higher/lower than under CD.',
    },
  };

  // Matplotlib "coolwarm", sampled at 0, .25, .5, .75, 1 and interpolated per row (warmer = worse).
  const STOPS = [[59, 76, 192], [141, 176, 254], [221, 221, 221], [244, 154, 123], [180, 4, 38]];
  function heat(t) {
    const x = Math.min(Math.max(t, 0), 1) * (STOPS.length - 1);
    const i = Math.min(Math.floor(x), STOPS.length - 2), u = x - i;
    const a = STOPS[i], b = STOPS[i + 1];
    const c = a.map((v, k) => Math.round(v + (b[k] - v) * u));
    return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
  }

  function render(metric) {
    const table = document.getElementById('region-table');
    const caption = document.getElementById('region-caption');
    if (!table) return;
    const { order, rows, caption: text } = DATA[metric];
    const baseOrder = DATA.cd.order;

    let head = '<thead><tr><th>Model</th>';
    order.forEach((key, idx) => {
      const r = REGIONS[key];
      let mark = '';
      if (metric === 'dcd') {
        const before = baseOrder.indexOf(key);
        if (idx < before) mark = ' <span class="rank-up">▲</span>';
        else if (idx > before) mark = ' <span class="rank-down">▼</span>';
      }
      head += `<th>${r.name}${mark}<span class="n">n = ${r.n}</span></th>`;
    });
    head += '</tr></thead>';

    let body = '<tbody>';
    Object.entries(rows).forEach(([model, vals]) => {
      const meds = order.map((k) => vals[k][0]);
      const lo = Math.min(...meds), hi = Math.max(...meds);
      body += `<tr><td>${model}</td>`;
      order.forEach((k) => {
        const [med, iqr] = vals[k];
        const t = hi === lo ? 0 : (med - lo) / (hi - lo);
        const dark = t < 0.15 || t > 0.85;   // ends of coolwarm need light text
        body += `<td class="val${dark ? ' on-dark' : ''}" style="background:${heat(t)}">${med.toFixed(2)}<span class="iqr">(${iqr.toFixed(2)})</span></td>`;
      });
      body += '</tr>';
    });
    body += '</tbody>';

    table.innerHTML = head + body;
    caption.innerHTML = text;
    if (window.renderMathInElement) {
      window.renderMathInElement(caption, { delimiters: [{ left: '\\(', right: '\\)', display: false }] });
    }
  }

  document.addEventListener('DOMContentLoaded', () => {
    const toggle = document.getElementById('metric-toggle');
    if (!toggle) return;
    toggle.querySelectorAll('button').forEach((btn) => {
      btn.addEventListener('click', () => {
        toggle.querySelectorAll('button').forEach((b) => b.classList.toggle('is-active', b === btn));
        render(btn.dataset.metric);
      });
    });
    render('cd');
  });
})();
