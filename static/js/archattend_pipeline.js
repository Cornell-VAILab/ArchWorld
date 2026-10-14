// An interactive walk through Figure 5 of the paper. The panels, the palette
// and every image tile come from the figure itself; the page just takes them
// one at a time and lets the reader compare the before/after row in place.
(function () {
  const IMG = 'static/images/method/';
  // sampled straight out of the figure's vector source
  const EL = { eaves: '#a52a2a', pagoda: '#ffb90f', window: '#e9967a' };
  const TOKENS = ['#0047ab', '#a52a2a', '#ffb90f', '#e9967a', '#48d1cc', '#2e8b57'];
  const BEFORE = '#ff4444', AFTER = '#ff9800';
  // normalized to each reconstruction tile: x, y, w, h
  const CALLOUT = { before: [0.3433, 0.2975, 0.1274, 0.2203], after: [0.3338, 0.1881, 0.1817, 0.3005] };

  const chip = (c, round) =>
    `<span class="aw-pipe-chip${round ? ' is-round' : ''}" style="background:${c}"></span>`;

  // ---- panel 1: input images and semantic elements --------------------
  function panelInput() {
    const view = (n, label) => `
      <div class="aw-pipe-vrow">
        <span class="aw-pipe-vlab">I<sub>${n}</sub></span>
        <figure><img src="${IMG}i${n}.jpg" alt="Input view ${label}" loading="lazy"></figure>
        <figure><img src="${IMG}i${n}_mask.jpg" alt="SAM 3 segmentation of view ${label}" loading="lazy"></figure>
      </div>`;
    const leg = Object.entries(EL).map(([k, c]) =>
      `<li>${chip(c)}<span>&ldquo;${k}&rdquo;</span></li>`).join('');
    return `
      <div class="aw-pipe-grid aw-pipe-input">
        <div class="aw-pipe-views">
          <div class="aw-pipe-heads"><span></span><span>Images</span><span>Masks</span></div>
          ${view(1, 'I1')}
          ${view(2, 'I2')}
        </div>
        <div class="aw-pipe-legend">
          <div class="aw-pipe-colhead is-sym">&#8496;<sub>arch</sub></div>
          <ul>${leg}</ul>
          <div class="aw-pipe-colhead is-sym">&#8496;<sub>non</sub></div>
          <ul class="is-non">
            <li>${chip('#0047ab')}<span>&ldquo;sky&rdquo;</span></li>
            <li>${chip('#48d1cc')}<span>&ldquo;trees&rdquo;</span></li>
          </ul>
          <p>Both sets are segmented. The anchor is picked from &#8496;<sub>arch</sub>; &#8496;<sub>non</sub> is where attention leaks.</p>
        </div>
      </div>`;
  }

  // ---- panel 2: pass 1, the anchor element ----------------------------
  function panelPass1() {
    const strip = (n) => `<div class="aw-pipe-strip" aria-label="token representation of view ${n}">` +
      TOKENS.map((c) => `<i style="background:${c}"></i>`).join('') + '</div>';
    const row = (name) => `
      <div class="aw-pipe-eq">
        <span class="sum">&sum;</span><span class="fn">attn</span>
        <span class="paren">(</span>${chip(EL[name])}<span class="comma">,</span>${chip(EL[name], true)}<span class="paren">)</span>
        <span class="eq">=</span>
        <span class="sc">S<sup>23</sup></span><span class="paren">(</span><b style="color:${EL[name]}">${name}</b><span class="paren">)</span>
      </div>`;
    return `
      <div class="aw-pipe-grid aw-pipe-p1">
        <div class="aw-pipe-feed">
          <div class="aw-pipe-ins">
            <div><span class="aw-pipe-vlab">I<sub>1</sub></span><span class="arr">&rarr;</span></div>
            <div><span class="aw-pipe-vlab">I<sub>2</sub></span><span class="arr">&rarr;</span></div>
          </div>
          <div class="aw-pipe-backbone">3DFM Backbone</div>
        </div>
        <div class="aw-pipe-strips">
          <div class="aw-pipe-colhead is-sym">T<sup>23</sup></div>
          ${strip(1)}${strip(2)}
        </div>
        <div class="aw-pipe-eqs">
          <div class="aw-pipe-colhead">Semantic Cross-View Attention</div>
          ${row('eaves')}${row('pagoda')}${row('window')}
          <div class="aw-pipe-pick">
            <span class="aw-pipe-max">max</span>
            <span class="arr">&rarr;</span>
            <span class="aw-pipe-estar">e<sup>&lowast;</sup> = <b style="color:${EL.pagoda}">pagoda</b></span>
          </div>
        </div>
      </div>`;
  }

  // ---- panel 3: pass 2, the gradient-ascent update --------------------
  function panelPass2() {
    const col = (c, kind) => `<div class="aw-pipe-tcol is-${kind}">` +
      Array.from({ length: 8 }, () => `<i style="background:${c}"></i>`).join('') + '</div>';
    return `
      <div class="aw-pipe-grid aw-pipe-p2">
        <div class="aw-pipe-tgroup">
          <div class="aw-pipe-colhead is-sym">T<sup>l</sup></div>
          <div class="aw-pipe-tcols">${col('#a6a6a6', 'sq')}${col('#a6a6a6', 'ci')}</div>
        </div>
        <span class="arr aw-pipe-bigarr">&rarr;</span>
        <div class="aw-pipe-update">
          <div class="aw-pipe-colhead">Gradient-Ascent Update</div>
          <div class="aw-pipe-rule">
            \\(\\mathbf{T}^{l}  \\leftarrow  \\mathbf{T}^{l} + \\eta \\cdot \\nabla S^{l}(e^{\\ast})\\)
          </div>
          <p>One step in every global attention block. The model weights never move &mdash; only the token representation does.</p>
        </div>
        <span class="arr aw-pipe-bigarr">&rarr;</span>
        <div class="aw-pipe-tgroup">
          <div class="aw-pipe-colhead is-sym">T<sup>l+1</sup></div>
          <div class="aw-pipe-tcols">${col(EL.pagoda, 'sq')}${col(EL.pagoda, 'ci')}</div>
        </div>
      </div>`;
  }

  // ---- panel 4: output, before vs after -------------------------------
  function panelOutput() {
    const side = (k) => {
      const tone = k === 'before' ? BEFORE : AFTER;
      const [x, y, w, h] = CALLOUT[k];
      return `
        <div class="aw-pipe-side" data-side="${k}" style="--tone:${tone}">
          <figure class="aw-pipe-recon">
            <span class="aw-pipe-shot">
              <img src="${IMG}recon_${k}.jpg" alt="Reconstruction ${k} the intervention" loading="lazy">
              <span class="aw-pipe-callout" style="left:${x * 100}%;top:${y * 100}%;width:${w * 100}%;height:${h * 100}%"></span>
            </span>
            <figcaption>Reconstruction</figcaption>
          </figure>
          <figure class="aw-pipe-zoom">
            <img src="${IMG}zoom_${k}.jpg" alt="Close-up of the reconstruction ${k} the intervention" loading="lazy">
            <figcaption>Close-up</figcaption>
          </figure>
          <figure class="aw-pipe-attn">
            <img src="${IMG}attn_${k}_1.jpg" alt="Attention map for view 1 ${k} the intervention" loading="lazy">
            <figcaption>Attention, I<sub>1</sub></figcaption>
          </figure>
          <figure class="aw-pipe-attn">
            <img src="${IMG}attn_${k}_2.jpg" alt="Attention map for view 2 ${k} the intervention" loading="lazy">
            <figcaption>Attention, I<sub>2</sub></figcaption>
          </figure>
        </div>`;
    };
    return `
      <div class="aw-pipe-out">
        <div class="buttons has-addons is-centered aw-toggle aw-pipe-ba" role="group" aria-label="Before or after the intervention">
          <button class="button is-small is-active" type="button" data-show="before">Before</button>
          <button class="button is-small" type="button" data-show="after">After</button>
        </div>
        <div class="aw-pipe-stack">${side('before')}${side('after')}</div>
      </div>`;
  }

  function wireOutput(stage) {
    const bar = stage.querySelector('.aw-pipe-ba');
    if (!bar) return;
    const show = (k) => {
      stage.querySelectorAll('.aw-pipe-side').forEach((s) => s.classList.toggle('is-on', s.dataset.side === k));
      bar.querySelectorAll('button').forEach((b) => b.classList.toggle('is-active', b.dataset.show === k));
    };
    bar.addEventListener('click', (e) => {
      const b = e.target.closest('button[data-show]');
      if (b) show(b.dataset.show);
    });
    show('before');
  }

  const STEPS = [
    { tone: '#7a0000', title: 'Input images and semantic elements',
      lede: 'Every view of a scene is segmented with SAM 3 into architectural elements — here the eaves, the pagoda and the windows of Kinkaku-ji — and non-architectural ones.',
      html: panelInput },
    { tone: '#28448d', title: 'Pass 1 · Determine the anchor element',
      lede: 'For each element, sum the attention its tokens in one view pay to its tokens in another. The architectural element that scores highest becomes the anchor e∗. We ship the anchors, so a run of the released code starts at Pass 2.',
      html: panelPass1 },
    { tone: '#3b6c1e', title: 'Pass 2 · Architecture-centric intervention',
      lede: 'Run the model again and, in every global attention block, take a single gradient-ascent step on the token representation that raises attention to the anchor.',
      html: panelPass2, },
    { tone: '#5c4080', title: 'Output reconstruction and attention maps',
      lede: 'Attention stops spreading across the sky and the trees and gathers on the building, and the reconstruction fills in. Switch between before and after to compare them in place.',
      html: panelOutput, after: wireOutput },
  ];

  function init() {
    const root = document.getElementById('archattend-pipeline');
    if (!root) return;
    const stage = root.querySelector('.aw-pipe-stage');
    const title = root.querySelector('.aw-pipe-title');
    const lede = root.querySelector('.aw-pipe-lede');
    const dots = root.querySelector('.aw-pipe-dots');
    let i = 0;

    STEPS.forEach((s, k) => {
      const b = document.createElement('button');
      b.className = 'aw-pipe-dot';
      b.type = 'button';
      b.setAttribute('aria-label', s.title);
      b.addEventListener('click', () => go(k));
      dots.appendChild(b);
    });

    function go(k) {
      i = (k + STEPS.length) % STEPS.length;
      const s = STEPS[i];
      root.style.setProperty('--tone', s.tone);
      stage.innerHTML = s.html();
      if (s.after) s.after(stage);
      title.textContent = `${i + 1} / ${STEPS.length}  ·  ${s.title}`;
      lede.textContent = s.lede;
      [...dots.children].forEach((d, n) => d.classList.toggle('is-on', n === i));
      stage.classList.remove('is-in');
      void stage.offsetWidth;
      stage.classList.add('is-in');
      if (typeof window.renderMathInElement === 'function') {
        window.renderMathInElement(stage, {
          delimiters: [{ left: '\\(', right: '\\)', display: false }],
          throwOnError: false,
        });
      }
    }

    root.querySelector('.aw-pipe-prev').addEventListener('click', () => go(i - 1));
    root.querySelector('.aw-pipe-next').addEventListener('click', () => go(i + 1));
    root.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowLeft') { e.preventDefault(); go(i - 1); }
      if (e.key === 'ArrowRight') { e.preventDefault(); go(i + 1); }
    });
    go(0);
  }
  document.readyState === 'loading' ? document.addEventListener('DOMContentLoaded', init) : init();
})();
