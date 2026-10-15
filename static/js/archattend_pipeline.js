// Figure 5 of the paper, split into the two passes it actually describes and
// assembled one step at a time.
//
//   Figure 1  Semantic Cross-View Attention   (input -> masks -> elements ->
//             backbone -> per-element scores -> argmax picks the anchor)
//   Figure 2  Architecture-centric Intervention  (the gradient-ascent step on
//             the token representation, then the before/after output row)
//
// Both figures are drawn once in full; every element belongs to a stage, and
// stepping forward only ever reveals more, so the last stage IS the complete
// figure.
//
// ONE DRAWING, TWO FRAMES. The two figures share a viewBox width, a unit
// scale, a type scale, one chrome (a letter-spaced section header over a
// 2-unit rule), one arrow, one brace, one clearance and one way of centring
// text, so a number means the same thing in both and a symbol drawn in both
// comes out the same size.
//
// NOTHING IS PLACED BY EYE. Column widths are solved from the measured ink of
// the strings that sit in them, so the three columns of figure 1 span exactly
// EDGE..W-EDGE however wide "window" turns out to be; every arrow has the same
// length and the same GAPA of clearance at both ends; and a row of text shares
// ONE baseline offset computed from the row's combined ink, never one baseline
// per string (that is what put the legend off its grid).
//
// THE CANVAS IS CROPPED TO THE STEP IN BOTH AXES at a constant scale: the
// viewBox is cropped to what has been revealed and the figure box is narrowed
// by the same fraction, so the drawing grows into the card rather than
// zooming, and the leftover is split evenly on the two sides. Section rules
// are chrome: each one re-fits to its OWN revealed content on every render, so
// no rule ever runs past the thing it underlines.
(function () {
  'use strict';

  var NS = 'http://www.w3.org/2000/svg';
  var IMG = 'static/images/method/';

  /* ---- palette, sampled from the figure's own vector source ------------ */
  var INK = '#1f2937', MUTED = '#6b7280', BORDER = '#e5e7eb', FAINT = '#9ca3af';
  // Fills. The mask tiles are quantised to exactly these six, so every colour
  // in a mask has a legend entry and every legend entry occurs in a mask.
  var E = {
    eaves: '#a52a2a', pagoda: '#ffb90f', window: '#e9967a',
    sky: '#0047ab', trees: '#48d1cc', water: '#2e8b57'
  };
  // The same six as TEXT. Four of the fills are 2-3:1 on white, which is below
  // the floor for 15px type, so the words are set in darkened versions while
  // every swatch, token and mark keeps the pure fill.
  var E_INK = {
    eaves: '#a52a2a', pagoda: '#8a6400', window: '#b4502f',
    sky: '#0047ab', trees: '#157f7a', water: '#2e8b57'
  };
  var TONE_A = '#7a0000', TONE_B = '#28448d', TONE_C = '#3b6c1e', TONE_D = '#5c4080';
  var BOX_FILL = '#fff3c2', BOX_STROKE = '#e0ad3a', BOX_INK = '#6b4e00';
  var BEFORE = '#ff4444', AFTER = '#ff9800';
  var BEFORE_INK = '#c62828', AFTER_INK = '#b45309';
  var BEFORE_TINT = '#ffecec', AFTER_TINT = '#fff3e0';

  /* ---- one geometry for both figures ---------------------------------- */
  var W = 920;          // viewBox width of BOTH figures, so one unit is one unit
  var EDGE = 20;        // x of the left edge of the content, in BOTH figures
  var MARGIN = 20;      // white around the content, equal on all four sides

  /* ---- one arrow, and one clearance around it -------------------------- */
  var AR_LEN = 46, AR_W = 1.8, AR_HEAD = 10, AR_HALF = 4.6;
  var GAPA = 14;                      // clearance at BOTH ends of EVERY arrow
  var FLOW = GAPA + AR_LEN + GAPA;    // the gap an arrow needs between columns
  var BRACE_PAD = 4;                  // how far a brace clears what it brackets
  var BRACE_ARM = 12;                 // ink width of a brace, cusp to arms

  /* ---- one chrome ------------------------------------------------------ */
  // A section is a letter-spaced caps header over a 2-unit rule in the
  // section's tone. Measured DOWN FROM THE HEADER'S INK TOP, which is the
  // number the margin is quoted against.
  var SEC_RULE_Y = 16;   // top of the rule
  var SEC_RULE_H = 2;
  var SEC_BODY = 34;     // top of the section's content
  var SEC_SEP = 26;      // white between one section's last ink and the next header

  /* ---- one type scale -------------------------------------------------- */
  // Google Sans ships 400/500/700 only: 600 is synthesised and does not match
  // 700 anywhere else on the page, so nothing here asks for it.
  var T_TITLE = 12.5;   // section header: caps, 700, letter-spaced, in tone
  var T_HEAD = 12.5;    // column heading: sentence case, 700, ink -- ONE spec,
  //                       the same in both figures
  var T_LABEL = 11.5;   // the strings a heading heads: legend names, 500, ink
  var T_SMALL = 11;     // asides: the view labels, the repeat note
  var M_INLINE = 15;    // I_1, the score rows, e*
  var M_TAG = 18;       // T^23, T^l, T^{l+1}
  var M_FORM = 30;      // the update rule
  var ROW_INK = 1.519 * M_INLINE;   // ink height of a score row, measured
  // Cap height / em for Google Sans, measured by pixel scan. Used only as the
  // build-time placement, before measureText is available; every string is
  // re-placed on real ink as soon as the font is up.
  var CAP = 0.726;
  function capMid(size) { return CAP * size / 2; }

  /* ---------------------------------------------------- ink measurement */
  // Column widths are solved from real ink, so the figure needs the page's
  // own font metrics at build time. The figure is mounted lazily and the
  // build waits on document.fonts.ready, so this is the real face in practice;
  // the baked fallbacks below only matter if measureText is unavailable.
  var _ctx, _fam;
  function ctx2d() {
    if (_ctx === undefined) {
      try { _ctx = document.createElement('canvas').getContext('2d') || null; }
      catch (err) { _ctx = null; }
    }
    return _ctx;
  }
  function family() {
    if (_fam == null) {
      try { _fam = getComputedStyle(document.body).fontFamily || 'sans-serif'; }
      catch (err) { _fam = 'sans-serif'; }
    }
    return _fam;
  }
  // {w advance, l ink left of origin, r ink right of origin, a ascent, d descent}
  function ink(str, size, weight, ls) {
    var c = ctx2d();
    if (!c || !c.measureText) return null;
    c.font = (weight || 400) + ' ' + size + 'px ' + family();
    if ('letterSpacing' in c) c.letterSpacing = ls || '0px';
    var m = c.measureText(str);
    if (m.actualBoundingBoxAscent == null) return null;
    return {
      w: m.width, l: m.actualBoundingBoxLeft, r: m.actualBoundingBoxRight,
      a: m.actualBoundingBoxAscent, d: m.actualBoundingBoxDescent
    };
  }
  function inkW(str, size, weight, ls, fallback) {
    var m = ink(str, size, weight, ls);
    return m ? m.l + m.r : fallback;
  }
  function widest(list, size, weight, fallback) {
    var best = 0;
    for (var i = 0; i < list.length; i++) best = Math.max(best, inkW(list[i], size, weight, null, 0));
    return best || fallback;
  }

  // KaTeX cannot be measured with measureText, and it is not even loaded when
  // the figure is built, so the handful of typeset strings the layout depends
  // on carry their measured ink here. Measured in Chrome at katex 0.16.11 with
  // the page's own stylesheet; each is quoted as ink width, ink height, and
  // how far the ink's centre sits ABOVE the centre of the box C.tex places.
  var KW = {
    eSet: { w: 45.0 },                    // \mathcal{E}_{arch} at M_INLINE
    i: { w: 13.7 },                       // I_1 at M_INLINE
    iBox: { h: 14.2, up: 1.8 },           // ... its ink box
    row: { w: 254.0 },                    // widest score row at M_INLINE
    eStar: { w: 96.0 },                   // e^{*} = pagoda at M_INLINE
    t23: { h: 18.6, up: 2.7 },            // T^{23} at M_TAG
    tl: { h: 18.6, up: 2.7 },             // T^{l} at M_TAG
    tl1: { h: 18.6, up: 2.7 }             // T^{l+1} at M_TAG
  };

  /* ---------------------------------------------------------- primitives */
  function el(name, attrs, text) {
    var e = document.createElementNS(NS, name), k;
    if (attrs) for (k in attrs) if (attrs[k] != null) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    return e;
  }

  // Three z-layers, each holding one group per stage. This is load-bearing:
  // the anchor highlight belongs to a later stage than the formulas it sits
  // behind, so painting strictly by stage order would hide them.
  function canvas(nStages, label) {
    var svg = el('svg', {
      class: 'aw-pipe-svg', role: 'img', 'aria-label': label,
      viewBox: '0 0 ' + W + ' 1', preserveAspectRatio: 'xMidYMid meet'
    });
    var defs = el('defs');
    svg.appendChild(defs);
    var groups = { bg: [], mid: [], top: [] };
    ['bg', 'mid', 'top'].forEach(function (name) {
      var layer = el('g', { 'data-layer': name });
      svg.appendChild(layer);
      for (var s = 0; s < nStages; s++) {
        var g = el('g', { 'data-stage': String(s + 1), class: 'aw-pipe-op' });
        layer.appendChild(g);
        groups[name].push(g);
      }
    });
    var C = {
      svg: svg, defs: defs,
      add: function (layer, stage, name, attrs, text) {
        var e = el(name, attrs, text);
        groups[layer][stage - 1].appendChild(e);
        return e;
      },
      fit: function () { }
    };

    /* One text primitive. o.cy centres the string's ink on cy; o.inkTop puts
       its ink top on a line; o.inkLeft puts its ink left edge on a line.
       o.row names a BASELINE GROUP: every string in a group is placed with one
       offset computed from the group's combined ink, so a row of headings or a
       column of labels lands on one grid instead of on six measured ink boxes. */
    C.txt = function (st, x, y, s, o) {
      var size = o.size, t = C.add('top', st, 'text', {
        x: x, y: o.inkTop != null ? o.inkTop + CAP * size : y + capMid(size),
        'font-size': size, 'font-weight': o.weight,
        fill: o.fill, 'text-anchor': o.anchor || 'start', 'letter-spacing': o.ls || null
      }, s);
      if (o.inkTop != null) t.dataset.ty = o.inkTop; else t.dataset.cy = y;
      if (o.inkLeft != null) t.dataset.lx = o.inkLeft;
      else if (o.anchor === 'middle') t.dataset.cx = x;
      if (o.row) t.dataset.row = o.row;
      return t;
    };
    /* the same, turned a quarter turn, centred on (cx, cy) in both axes */
    C.rot = function (st, cx, cy, s, o) {
      var t = C.add('top', st, 'text', {
        x: cx, y: cy + capMid(o.size), 'font-size': o.size, 'font-weight': o.weight,
        fill: o.fill, 'text-anchor': 'middle', 'letter-spacing': o.ls || null,
        transform: 'rotate(-90 ' + cx + ' ' + cy + ')'
      }, s);
      t.dataset.cy = cy; t.dataset.cx = cx; t.dataset.rot = '1';
      return t;
    };
    /* KaTeX in a box that centres it; (cx, cy) is the centre of the box */
    C.tex = function (st, cx, cy, w, h, size, latex, plain, align) {
      var fo = C.add('top', st, 'foreignObject', {
        x: align === 'left' ? cx : cx - w / 2, y: cy - h / 2, width: w, height: h,
        style: 'overflow:visible;pointer-events:none'
      });
      var d = document.createElement('div');
      d.className = align === 'left' ? 'aw-pipe-fo aw-pipe-fo-l' : 'aw-pipe-fo';
      d.style.fontSize = size + 'px';
      var sp = document.createElement('span');
      sp.className = 'k';
      math(sp, latex, plain);
      d.appendChild(sp);
      fo.appendChild(d);
      return sp;
    };
    /* the one arrow: same length, weight, colour and head in both figures */
    C.arrow = function (st, x, y) {
      C.add('mid', st, 'line', {
        x1: x, y1: y, x2: x + AR_LEN - AR_HEAD + 1, y2: y,
        stroke: MUTED, 'stroke-width': AR_W, 'stroke-linecap': 'butt'
      });
      C.add('mid', st, 'polygon', {
        points: (x + AR_LEN - AR_HEAD) + ',' + (y - AR_HALF) + ' '
          + (x + AR_LEN) + ',' + y + ' '
          + (x + AR_LEN - AR_HEAD) + ',' + (y + AR_HALF),
        fill: MUTED
      });
    };
    /* the one brace: every brace in both figures clears the content it
       brackets by the same BRACE_PAD, top and bottom */
    C.brace = function (st, xTip, dir, y0, y1) {
      return C.add('mid', st, 'path', {
        d: bracePath(xTip, dir, y0 - BRACE_PAD, y1 + BRACE_PAD, 10),
        stroke: MUTED, 'stroke-width': AR_W,
        fill: 'none', 'stroke-linecap': 'round', 'stroke-linejoin': 'round'
      });
    };
    return C;
  }

  // dir +1 draws "{" with its cusp at xTip and its arms at xTip+BRACE_ARM;
  // dir -1 mirrors it. The cusp lands on (y0+y1)/2 by construction, which is
  // how a brace is made to point at exactly the row it selects.
  function bracePath(xTip, dir, y0, y1, r) {
    var xs = xTip + dir * 7, xa = xTip + dir * BRACE_ARM, ym = (y0 + y1) / 2;
    return 'M ' + xa + ' ' + y0 + ' Q ' + xs + ' ' + y0 + ' ' + xs + ' ' + (y0 + r)
      + ' L ' + xs + ' ' + (ym - r) + ' Q ' + xs + ' ' + ym + ' ' + xTip + ' ' + ym
      + ' Q ' + xs + ' ' + ym + ' ' + xs + ' ' + (ym + r)
      + ' L ' + xs + ' ' + (y1 - r) + ' Q ' + xs + ' ' + y1 + ' ' + xa + ' ' + y1;
  }

  // The square and the disc are the same object seen in two views, so they are
  // drawn at one size wherever they appear: in the token strip, in the token
  // blocks and inside attn(.,.). svg === true returns an HTML-embeddable copy
  // for the score rows, which are typeset rather than drawn.
  var MARK = 13;
  function markSVG(color, round) {
    var s = el('svg', {
      width: MARK, height: MARK, viewBox: '0 0 ' + MARK + ' ' + MARK,
      'aria-hidden': 'true', class: 'aw-pipe-mark'
    });
    s.appendChild(round
      ? el('circle', { cx: MARK / 2, cy: MARK / 2, r: MARK / 2 - 0.5, fill: color,
        stroke: 'rgba(0,0,0,.14)', 'stroke-width': 1 })
      : el('rect', { x: 0.5, y: 0.5, width: MARK - 1, height: MARK - 1, rx: 3,
        fill: color, stroke: 'rgba(0,0,0,.14)', 'stroke-width': 1 }));
    return s;
  }

  // Every placed string is measured once, when the font is ready, and moved
  // onto its real ink box -- no static rule can do better, because "eaves" is
  // an x-height band and "Close-up" runs from cap height to descender.
  //
  // BUT a string is only ever measured ALONE when it is alone. Strings that
  // belong to one row or one column carry the same data-row, and the whole
  // group is placed with ONE offset taken from the group's combined ink, so
  // six legend labels keep a constant offset from their six swatches and three
  // column headings share one baseline. Per-string centring is what knocked
  // them off the grid.
  function inkCentre(svg) {
    var c = ctx2d();
    if (!c || !c.measureText) return;
    var nodes = [].slice.call(svg.querySelectorAll('text[data-cy],text[data-ty]'));
    var rows = {};
    var ms = nodes.map(function (t) {
      var cs = getComputedStyle(t);
      c.font = cs.fontStyle + ' ' + cs.fontWeight + ' ' + cs.fontSize + ' ' + cs.fontFamily;
      if ('letterSpacing' in c) {
        c.letterSpacing = cs.letterSpacing === 'normal' ? '0px' : cs.letterSpacing;
      }
      var m = c.measureText(t.textContent);
      if (m.actualBoundingBoxAscent == null) return null;
      var key = t.dataset.row;
      if (key) {
        var r = rows[key] || (rows[key] = { a: -1e9, d: -1e9 });
        r.a = Math.max(r.a, m.actualBoundingBoxAscent);
        r.d = Math.max(r.d, m.actualBoundingBoxDescent);
      }
      return m;
    });
    nodes.forEach(function (t, i) {
      var m = ms[i];
      if (!m) return;
      var a = m.actualBoundingBoxAscent, d = m.actualBoundingBoxDescent;
      if (t.dataset.row) { a = rows[t.dataset.row].a; d = rows[t.dataset.row].d; }
      t.setAttribute('y', t.dataset.ty != null
        ? +t.dataset.ty + a                       // ink top on a line
        : +t.dataset.cy + (a - d) / 2);           // ink centred on a line
      // Horizontally, onto the centre of the ink rather than of the advance
      // width -- the half unit a right side bearing otherwise costs. For a
      // rotated string this axis is the screen's vertical one, which is why
      // the two corrections are not swapped for it: each stays on the axis of
      // the measurement it came from.
      if (m.actualBoundingBoxLeft == null) return;
      var iw = m.actualBoundingBoxRight - m.actualBoundingBoxLeft;
      if (t.dataset.lx != null) {
        t.setAttribute('x', +t.dataset.lx + m.actualBoundingBoxLeft);
      } else if (t.dataset.cx != null) {
        t.setAttribute('x', +t.dataset.cx + m.width / 2 - iw / 2);
      }
    });
  }

  /* ------------------------------------------------------------ KaTeX ---- */
  // KaTeX is fetched after DOMContentLoaded on purpose (a slow CDN response on
  // the critical path once stalled the whole page), so nothing here may assume
  // window.katex exists at build time. Every formula ships an ASCII fallback
  // inside its span and is re-rendered when KaTeX lands.
  //
  // The fallbacks are ASCII on purpose: Google Sans has no glyph for U+2130
  // script capital E, U+2217 asterisk operator, U+2211 summation, U+2207 nabla
  // or U+207A superscript plus, and document.fonts.check() does not report it
  // because it answers true as soon as *any* fallback face can draw the glyph.
  function math(span, latex, plain) {
    span.dataset.tex = latex;
    span.innerHTML = plain;
  }
  function typeset(root) {
    if (!window.katex) return false;
    var nodes = root.querySelectorAll('[data-tex]');
    for (var i = 0; i < nodes.length; i++) {
      var d = nodes[i];
      try {
        window.katex.render(d.dataset.tex, d, { throwOnError: false, displayMode: false });
      } catch (err) { /* the ASCII fallback stays put */ }
      d.removeAttribute('data-tex');
    }
    // The caption shipped in index.html writes its maths with the \( \)
    // delimiters rather than data-tex, and auto-render has already run by the
    // time the figure builds, so the caption needs rendering here too.
    if (window.renderMathInElement) {
      try {
        window.renderMathInElement(root, {
          delimiters: [{ left: '\\(', right: '\\)', display: false }],
          throwOnError: false
        });
      } catch (err) { /* leave the source text in place */ }
    }
    return true;
  }
  function whenKatex(fn) {
    if (fn()) return;
    var iv = setInterval(function () { if (fn()) clearInterval(iv); }, 150);
    setTimeout(function () { clearInterval(iv); }, 20000);
  }

  /* ---------------------------------------------------- the one chrome --- */
  // Each section owns its rule, and each rule re-fits to the right edge of its
  // OWN content at the current stage. A section whose content has not been
  // revealed yet carries no rule at all.
  function chrome(C) {
    var secs = [];
    return {
      // Returns the y of the section's content top.
      open: function (st, top, title, tone, rights) {
        C.txt(st, EDGE, 0, title, {
          size: T_TITLE, weight: 700, fill: tone, ls: '0.1em',
          inkLeft: EDGE, inkTop: top
        });
        secs.push({
          rect: C.add('bg', st, 'rect', {
            x: EDGE, y: top + SEC_RULE_Y, width: 1, height: SEC_RULE_H, fill: tone
          }),
          rights: rights
        });
        return top + SEC_BODY;
      },
      fit: function (i) {
        secs.forEach(function (s) {
          var r = s.rights[i];
          s.rect.setAttribute('width', Math.max(1, (r == null ? EDGE + 1 : r) - EDGE));
        });
      }
    };
  }

  /* =====================================================================
     FIGURE 1 — Semantic Cross-View Attention
     ===================================================================== */
  // Section A, three columns: the views, their masks, the elements they split
  // into. Each column is as wide as what it holds; between them sits one FLOW,
  // which is exactly one arrow with GAPA of air at each end.
  //
  // The three columns are SOLVED, not chosen: the element column is as wide as
  // its own measured ink, and the tile height is whatever makes
  //   2*(4/3 h + gap + 3/4 h) + 2*FLOW + elementColumn == W - 2*EDGE
  // true. That is what puts the figure's right edge on x = W - EDGE exactly,
  // on the same line as figure 2's.
  function f1Geom() {
    var A = {
      top: MARGIN, gap: 12,
      lg: 12, bg: 12, sg: 9,      // label->brace, brace->swatch, swatch->name
      sw: 18, pitch: 22, setGap: 18
    };
    A.body = A.top + SEC_BODY;                      // 54
    A.head = A.body;                                // the column headings' ink top
    A.imgY = A.head + CAP * T_HEAD + 14;            // 77.075
    A.nameW = widest(['eaves', 'pagoda', 'window', 'sky', 'trees', 'water'],
      T_LABEL, 500, 39);
    A.setW = KW.eSet.w;
    A.col3 = A.setW + A.lg + BRACE_ARM + A.bg + A.sw + A.sg + A.nameW;
    A.imgH = (W - 2 * EDGE - 2 * A.gap - 2 * FLOW - A.col3) / (4 / 3 + 3 / 4) / 2;
    A.w1 = A.imgH * 4 / 3;
    A.w2 = A.imgH * 3 / 4;
    A.colW = A.w1 + A.gap + A.w2;
    A.c1 = EDGE;
    A.c2 = A.c1 + A.colW + FLOW;
    A.c3 = A.c2 + A.colW + FLOW;                    // == W - EDGE - A.col3
    A.mid = A.imgY + A.imgH / 2;                    // the line panel A's arrows sit on
    A.imgB = A.imgY + A.imgH;
    // I_1 / I_2 under their tiles: 14 of air under the tile, measured on ink.
    A.cap = A.imgB + 14 + KW.iBox.h / 2 - KW.iBox.up;
    // the element column, left to right
    A.set = A.c3 + A.setW / 2;                      // centre of E_arch / E_non
    A.brace = A.c3 + A.setW + A.lg;                 // the cusp
    A.swX = A.brace + BRACE_ARM + A.bg;
    A.nameX = A.swX + A.sw + A.sg;
    // two sets of three swatches
    A.g1 = A.imgY;
    A.g1e = A.g1 + 3 * A.sw + 2 * (A.pitch - A.sw);
    A.g2 = A.g1e + A.setGap;
    A.g2e = A.g2 + 3 * A.sw + 2 * (A.pitch - A.sw);
    A.bottom = Math.max(A.cap + KW.iBox.h / 2 + KW.iBox.up, A.g2e);

    // Section B: one chain, left to right, one arrow in every gap.
    var B = { top: A.bottom + SEC_SEP };
    B.body = B.top + SEC_BODY;
    B.tag = B.body + KW.t23.h / 2 - KW.t23.up;      // T^23, ink top on the body line
    B.r1 = B.body + KW.t23.h + 13 + MARK / 2;       // the squares
    B.pitch = 52;
    B.r2 = B.r1 + B.pitch;
    B.r3 = B.r2 + B.pitch;
    B.box = { x: 0, w: 56, y: B.body, h: 2 * (B.r2 - B.body) };
    B.i = EDGE;                                      // I_1 / I_2, ink left on EDGE
    B.box.x = B.i + KW.i.w + GAPA + AR_LEN + GAPA;
    B.a2 = B.box.x + B.box.w + GAPA;
    B.strip = B.a2 + AR_LEN + GAPA;
    B.spitch = 17;
    B.stripW = 5 * B.spitch + MARK;
    // The chain to the right is solved backwards from x = W - EDGE, so the
    // anchor lands on the figure's right margin and every gap in it is GAPA.
    B.star = W - EDGE - KW.eStar.w;
    B.maxAt = B.star - GAPA - AR_LEN;
    B.braceAt = B.maxAt - GAPA;
    B.hiR = B.braceAt - BRACE_ARM - GAPA;            // right edge of the highlight
    B.row = B.hiR - 10 - KW.row.w;
    B.hiL = B.row - 10;
    B.join = B.row - GAPA - AR_LEN;
    B.bottom = B.box.y + B.box.h;
    return { A: A, B: B };
  }

  var F1 = (function () {
    var G = f1Geom(), A = G.A, B = G.B;
    var rA = [A.c1 + A.colW, A.c2 + A.colW, W - EDGE, W - EDGE, W - EDGE];
    var rB = [null, null, null, B.row + KW.row.w, W - EDGE];
    return {
      G: G,
      depth: [A.bottom + MARGIN, A.bottom + MARGIN, A.bottom + MARGIN,
        B.bottom + MARGIN, B.bottom + MARGIN],
      right: [rA[0], rA[1], rA[2], Math.max(rA[3], rB[3]), Math.max(rA[4], rB[4])],
      rA: rA, rB: rB,
      label: 'Two views of a pagoda, their SAM 3 masks, the resulting semantic '
        + 'elements, and the cross-view attention score of each architectural '
        + 'element, whose maximum selects the anchor',
      stages: [
        { label: 'Views', tone: TONE_A, dwell: 1700,
          caption: ['Two views of one scene, straight off the internet — no poses, no depth.'] },
        { label: 'Masks', tone: TONE_A, dwell: 1700,
          caption: ['SAM 3 segments every view, so each patch carries a semantic label.'] },
        { label: 'Elements', tone: TONE_A, dwell: 1900,
          caption: ['The labels split into architectural elements ',
            { tex: '\\mathcal{E}_{\\mathrm{arch}}', plain: 'E<sub>arch</sub>' },
            ' and non-architectural ones ',
            { tex: '\\mathcal{E}_{\\mathrm{non}}', plain: 'E<sub>non</sub>' }, '.'] },
        { label: 'Encode', tone: TONE_B, dwell: 2400,
          caption: ['Both views run through the frozen backbone; at layer 23 we sum, per element, '
            + 'the attention its tokens in one view pay to its tokens in the other.'] },
        { label: 'Anchor', tone: TONE_B, dwell: 2200,
          caption: ['The architectural element with the largest score becomes the anchor ',
            { tex: 'e^{*}', plain: 'e*' }, '. Here, the pagoda.'] }
      ]
    };
  })();

  F1.build = function () {
    var C = canvas(5, F1.label), ch = chrome(C);
    var A = F1.G.A, B = F1.G.B;

    function tile(st, x, y, w, h, file, alt) {
      C.add('mid', st, 'image', {
        x: x, y: y, width: w, height: h, href: IMG + file,
        preserveAspectRatio: 'xMidYMid slice', 'aria-label': alt
      });
      C.add('mid', st, 'rect', {
        x: x + 0.5, y: y + 0.5, width: w - 1, height: h - 1,
        rx: 4, fill: 'none', stroke: BORDER, 'stroke-width': 1
      });
    }
    // one column of section A: the 4:3 view, a gap, the 3:4 view, two labels
    function views(st, x, f1, f2, alt1, alt2) {
      tile(st, x, A.imgY, A.w1, A.imgH, f1, alt1);
      tile(st, x + A.w1 + A.gap, A.imgY, A.w2, A.imgH, f2, alt2);
      C.tex(st, x + A.w1 / 2, A.cap, 44, 26, M_INLINE, 'I_1', 'I<sub>1</sub>');
      C.tex(st, x + A.w1 + A.gap + A.w2 / 2, A.cap, 44, 26, M_INLINE, 'I_2', 'I<sub>2</sub>');
    }
    function heading(st, x0, x1, s) {
      C.txt(st, (x0 + x1) / 2, A.head + CAP * T_HEAD / 2, s,
        { size: T_HEAD, weight: 700, fill: INK, anchor: 'middle', row: 'f1head' });
    }

    /* ---- section A : input images and semantic elements ---------------- */
    ch.open(1, A.top, 'INPUT IMAGES AND SEMANTIC ELEMENTS', TONE_A, F1.rA);

    heading(1, A.c1, A.c1 + A.colW, 'Input views');
    views(1, A.c1, 'i1_4x3.jpg', 'i2_3x4.jpg', 'Input view 1, a pagoda across a pond',
      'Input view 2, the same pagoda from the side');

    // "SAM 3" names the column, not the arrow: nothing is written on an arrow.
    C.arrow(2, A.c1 + A.colW + GAPA, A.mid);
    heading(2, A.c2, A.c2 + A.colW, 'SAM 3 masks');
    views(2, A.c2, 'i1_mask_4x3.png', 'i2_mask_3x4.png', 'SAM 3 segmentation of view 1',
      'SAM 3 segmentation of view 2');

    C.arrow(3, A.c2 + A.colW + GAPA, A.mid);
    heading(3, A.c3, W - EDGE, 'Semantic elements');
    // Each brace spans exactly the swatches it brackets, top edge to bottom
    // edge, with no overshoot -- the same rule as the brace in section B.
    C.tex(3, A.set, (A.g1 + A.g1e) / 2, A.setW + 10, 26, M_INLINE,
      '\\mathcal{E}_{\\mathrm{arch}}', 'E<sub>arch</sub>');
    C.tex(3, A.set, (A.g2 + A.g2e) / 2, A.setW + 10, 26, M_INLINE,
      '\\mathcal{E}_{\\mathrm{non}}', 'E<sub>non</sub>');
    C.brace(3, A.brace, +1, A.g1, A.g1e);
    C.brace(3, A.brace, +1, A.g2, A.g2e);
    // Six elements, the same six the token strip in section B carries and the
    // same six the mask tiles are quantised to.
    ['eaves', 'pagoda', 'window', 'sky', 'trees', 'water'].forEach(function (name, i) {
      var y = (i < 3 ? A.g1 : A.g2) + (i % 3) * A.pitch;
      C.add('mid', 3, 'rect', {
        x: A.swX, y: y, width: A.sw, height: A.sw, rx: 4, fill: E[name],
        stroke: 'rgba(0,0,0,.14)', 'stroke-width': 1
      });
      // One baseline offset for all six, so the labels keep the swatches' grid.
      C.txt(3, A.nameX, y + A.sw / 2, name,
        { size: T_LABEL, weight: 500, fill: INK, row: 'f1legend' });
    });

    /* ---- section B : the scores, and the argmax over them -------------- */
    ch.open(4, B.top, 'DETERMINE THE ANCHOR ELEMENT', TONE_B, F1.rB);

    C.tex(4, B.strip + B.stripW / 2, B.tag, 70, 30, M_TAG,
      '\\mathbf{T}^{23}', 'T<sup>23</sup>');
    [B.r1, B.r3].forEach(function (y) {
      C.tex(4, B.i, y, 44, 26, M_INLINE, y === B.r1 ? 'I_1' : 'I_2',
        y === B.r1 ? 'I<sub>1</sub>' : 'I<sub>2</sub>', 'left');
      C.arrow(4, B.i + KW.i.w + GAPA, y);
      C.arrow(4, B.a2, y);
    });
    C.add('bg', 4, 'rect', {
      x: B.box.x, y: B.box.y, width: B.box.w, height: B.box.h, rx: 8,
      fill: BOX_FILL, stroke: BOX_STROKE, 'stroke-width': 1.4
    });
    C.rot(4, B.box.x + B.box.w / 2, B.box.y + B.box.h / 2, '3DFM Backbone',
      { size: T_HEAD, weight: 700, fill: BOX_INK });

    // Row 1 is squares (view 1) and row 2 circles (view 2); the same two marks,
    // at the same size, appear inside each score expression.
    ['sky', 'eaves', 'pagoda', 'window', 'trees', 'water'].forEach(function (name, i) {
      C.add('mid', 4, 'rect', {
        x: B.strip + i * B.spitch, y: B.r1 - MARK / 2, width: MARK, height: MARK,
        rx: 3, fill: E[name], stroke: 'rgba(0,0,0,.14)', 'stroke-width': 1
      });
      C.add('mid', 4, 'circle', {
        cx: B.strip + i * B.spitch + MARK / 2, cy: B.r3, r: MARK / 2,
        fill: E[name], stroke: 'rgba(0,0,0,.14)', 'stroke-width': 1
      });
    });
    // Connectors are not arrows: they carry no head and are exempt from the
    // one-length rule, because they join two rows rather than point at one.
    C.add('mid', 4, 'path', {
      d: 'M ' + (B.strip + B.stripW + 6) + ' ' + B.r1 + ' H ' + B.join
        + ' V ' + B.r3 + ' H ' + (B.strip + B.stripW + 6),
      fill: 'none', stroke: MUTED, 'stroke-width': 1.4
    });
    C.arrow(4, B.join, B.r2);

    // The three score rows carry no padding of any kind: identical prefixes,
    // left-aligned on one x, and each bracket closed tight on its own word.
    [['eaves', B.r1], ['pagoda', B.r2], ['window', B.r3]].forEach(function (r) {
      var fo = C.add('top', 4, 'foreignObject', {
        x: B.row, y: r[1] - 17, width: 320, height: 34,
        style: 'overflow:visible;pointer-events:none'
      });
      var wrap = document.createElement('div');
      wrap.className = 'aw-pipe-fo aw-pipe-fo-l';
      wrap.style.fontSize = M_INLINE + 'px';
      var line = document.createElement('div');
      line.className = 'aw-pipe-row';
      var a = document.createElement('span');
      a.className = 'k';
      math(a, '\\sum\\mathrm{attn}\\big(', 'sum attn(');
      var comma = document.createElement('span');
      comma.className = 'k';
      math(comma, ',\\,', ',');
      var b = document.createElement('span');
      b.className = 'k';
      math(b, '\\big)\\;=\\;S^{23}\\big(\\textcolor{' + E_INK[r[0]] + '}{\\textbf{'
        + r[0] + '}}\\big)', ') = S<sup>23</sup>(' + r[0] + ')');
      line.appendChild(a);
      line.appendChild(markSVG(E[r[0]], false));
      line.appendChild(comma);
      line.appendChild(markSVG(E[r[0]], true));
      line.appendChild(b);
      wrap.appendChild(line);
      fo.appendChild(wrap);
    });

    /* ---- stage 5 : the argmax ------------------------------------------ */
    // max, its arrow and e* share the middle row's line, so the argmax leaves
    // the brace on the line it entered. The brace clears the widest thing on
    // its left -- the highlight -- by GAPA, the same air every arrow gets.
    C.add('bg', 5, 'rect', {
      x: B.hiL, y: B.r2 - 19, width: B.hiR - B.hiL, height: 38, rx: 10,
      fill: BOX_FILL, stroke: BOX_STROKE, 'stroke-width': 1.4
    });
    C.brace(5, B.braceAt, -1, B.r1 - ROW_INK / 2, B.r3 + ROW_INK / 2);
    C.tex(5, B.maxAt + AR_LEN / 2, B.r2 - 15, 50, 22, T_HEAD, '\\max', 'max');
    C.arrow(5, B.maxAt, B.r2);
    C.tex(5, B.star, B.r2, KW.eStar.w + 8, 28, M_INLINE,
      'e^{*}=\\textcolor{' + E_INK.pagoda + '}{\\textbf{pagoda}}', 'e* = pagoda', 'left');

    C.fit = ch.fit;
    return C;
  };

  /* =====================================================================
     FIGURE 2 — Architecture-centric Intervention
     ===================================================================== */
  // Section G, the update. Same chrome, same arrow, same GAPA, same left
  // margin as figure 1, so the two figures stack in register.
  function f2Geom() {
    var G = {
      top: MARGIN, gh: 75, gap: 10,
      fw: 26, fgap: 12, cols: 3, mark: 11, mpitch: 13,
      boxH: 152, ellGap: 12.4
    };
    G.body = G.top + SEC_BODY;                      // 54
    G.block = G.cols * G.fw + (G.cols - 1) * G.fgap;         // 102
    G.ellW = 8.8;                                   // half-width of the 3 dots
    // "view 1" / "view 2" label the two groups directly, on the figure's own
    // left margin, instead of a key parked somewhere else in the band.
    G.viewW = widest(['view 1', 'view 2'], T_SMALL, 500, 30);
    G.e1 = EDGE + G.viewW + G.ellGap + G.ellW;      // centre of the left ellipsis
    G.b1 = G.e1 + G.ellW + G.ellGap;
    G.e2 = W - EDGE - G.ellW;                       // centre of the right ellipsis
    G.b2 = G.e2 - G.ellW - G.ellGap - G.block;
    G.boxX = G.b1 + G.block + GAPA + AR_LEN + GAPA;
    G.boxW = G.b2 - GAPA - AR_LEN - GAPA - G.boxX;
    G.tag = G.body + KW.tl.h / 2 - KW.tl.up;        // T^l, ink top on the body line
    G.y = G.body + KW.tl.h + 13;                    // the first group of tokens
    G.g2 = G.y + G.gh + G.gap;
    G.mid = G.y + G.gh + G.gap / 2;                 // the flow line
    G.bot = G.g2 + G.gh;
    G.bracket = G.bot + 12;
    G.note = G.bracket + 8 + 12 + CAP * T_SMALL / 2;
    G.bottom = G.note + CAP * T_SMALL / 2;

    // Section O, the output: four columns, before over after.
    var O = { top: G.bottom + SEC_SEP, h: 150, spine: 29, rowGap: 12 };
    O.body = O.top + SEC_BODY;
    O.name = O.body + CAP * T_HEAD / 2;
    O.y = O.body + CAP * T_HEAD + 12;
    O.ratios = [681 / 400, 1, 416 / 314, 1];        // native, measured off the files
    // Every tile is O.h tall and its own native width; the gap is whatever the
    // band has left over, so the row starts on the left margin and ends on the
    // right one without a tile being stretched.
    O.gap = (W - 2 * EDGE - O.spine
      - O.h * O.ratios.reduce(function (a, b) { return a + b; }, 0)) / 4;
    O.y2 = O.y + O.h + O.rowGap;
    O.x = [];
    var x = EDGE + O.spine + O.gap;
    O.ratios.forEach(function (r) { O.x.push(x); x += r * O.h + O.gap; });
    O.bottom = O.y2 + O.h;
    return { G: G, O: O };
  }

  var F2 = (function () {
    var g = f2Geom(), G = g.G, O = g.O;
    var rG = [G.b1 + G.block, G.boxX + G.boxW, W - EDGE, W - EDGE];
    var rO = [null, null, null, W - EDGE];
    return {
      g: g,
      depth: [G.bot + MARGIN, G.bot + MARGIN, G.bottom + MARGIN, O.bottom + MARGIN],
      right: [rG[0], rG[1], rG[2], W - EDGE],
      rG: rG, rO: rO,
      label: 'One gradient-ascent step on the token representation, and the '
        + 'reconstruction, close-up and attention maps before and after it',
      stages: [
        { label: 'Tokens', tone: TONE_C, dwell: 2200,
          caption: ['Each global attention block takes in the token representation ',
            { tex: '\\mathbf{T}^{l}', plain: 'T<sup>l</sup>' },
            ': squares are view 1’s patch tokens, circles view 2’s, each column is one '
            + 'token’s vector, and the ellipses stand for the remaining tokens.'] },
        { label: 'Gradient step', tone: TONE_C, dwell: 2400,
          caption: ['Every model weight stays frozen. Only ',
            { tex: '\\mathbf{T}^{l}', plain: 'T<sup>l</sup>' },
            ' is treated as a free variable, and it takes one gradient-ascent step on ',
            { tex: 'S^{l}(e^{*})', plain: 'S<sup>l</sup>(e*)' },
            ', the attention score of the anchor element.'] },
        { label: 'Updated tokens', tone: TONE_C, dwell: 2200,
          caption: ['The updated ',
            { tex: '\\mathbf{T}^{l+1}', plain: 'T<sup>l+1</sup>' },
            ' propagates downstream, and the same step repeats in all 24 global '
            + 'attention blocks — the weights never change.'] },
        { label: 'Before → After', tone: TONE_D, dwell: 2600,
          caption: ['The output on either side of the intervention. Afterwards attention '
            + 'gathers on the pagoda, and the reconstruction fills in structure that '
            + 'was missing.'] }
      ]
    };
  })();

  F2.build = function () {
    var C = canvas(4, F2.label), ch = chrome(C);
    var G = F2.g.G, O = F2.g.O;

    // Two framed groups of columns. The padding is arithmetic, not taste:
    // (fw - mark)/2 across, and (gh - 5*mark - 4*(mpitch-mark))/2 down.
    function tokenBlock(st, bx, fill) {
      var padY = (G.gh - (5 * G.mark + 4 * (G.mpitch - G.mark))) / 2;
      for (var c = 0; c < G.cols; c++) {
        var fx = bx + c * (G.fw + G.fgap);
        [G.y, G.g2].forEach(function (gy, gi) {
          C.add('mid', st, 'rect', {
            x: fx, y: gy, width: G.fw, height: G.gh, rx: 3,
            fill: '#fff', stroke: INK, 'stroke-width': 1.4
          });
          for (var k = 0; k < 5; k++) {
            var my = gy + padY + k * G.mpitch;
            if (gi === 0) {
              C.add('mid', st, 'rect', {
                x: fx + (G.fw - G.mark) / 2, y: my,
                width: G.mark, height: G.mark, rx: 1.5, fill: fill
              });
            } else {
              C.add('mid', st, 'circle', {
                cx: fx + G.fw / 2, cy: my + G.mark / 2, r: G.mark / 2, fill: fill
              });
            }
          }
        });
      }
    }
    // cx is the centre dot; the outer dots' edges land on the content band's
    // boundary, so the figure keeps one left edge and one right edge
    function ellipsis(st, cx) {
      [G.y + G.gh / 2, G.g2 + G.gh / 2].forEach(function (cy) {
        [-7, 0, 7].forEach(function (dx) {
          C.add('mid', st, 'circle', { cx: cx + dx, cy: cy, r: 1.8, fill: FAINT });
        });
      });
    }

    /* ---- stage 1 : the incoming token representation -------------------- */
    ch.open(1, G.top, 'GRADIENT-ASCENT UPDATE', TONE_C, F2.rG);
    // The two groups are named where they are, on the left margin, level with
    // the middle of the group each names -- no key to carry across the figure.
    [[G.y, 'view 1'], [G.g2, 'view 2']].forEach(function (p) {
      C.txt(1, EDGE, p[0] + G.gh / 2, p[1],
        { size: T_SMALL, weight: 500, fill: MUTED, inkLeft: EDGE, row: 'f2views' });
    });
    ellipsis(1, G.e1);
    tokenBlock(1, G.b1, FAINT);
    C.tex(1, G.b1 + G.block / 2, G.tag, 70, 30, M_TAG, '\\mathbf{T}^{l}', 'T<sup>l</sup>');

    /* ---- stage 2 : the update itself ------------------------------------ */
    C.arrow(2, G.b1 + G.block + GAPA, G.mid);
    C.add('bg', 2, 'rect', {
      x: G.boxX, y: G.mid - G.boxH / 2, width: G.boxW, height: G.boxH, rx: 14,
      fill: '#f3f6f2', stroke: '#a7bd9a', 'stroke-width': 1.5
    });
    // The box is sized to the formula, with the same padding on all four
    // sides, not a letterbox with the formula dropped into it.
    C.tex(2, G.boxX + G.boxW / 2, G.mid - 24, G.boxW, 60, M_FORM,
      '\\mathbf{T}^{l}\\;\\leftarrow\\;\\mathbf{T}^{l}+\\eta\\,'
      + '\\nabla_{\\mathbf{T}^{l}}S^{l}(e^{*})',
      'T<sup>l</sup> ← T<sup>l</sup> + η · grad S<sup>l</sup>(e*)');
    C.tex(2, G.boxX + G.boxW / 2, G.mid + 36, G.boxW, 34, M_INLINE,
      'e^{*}=\\textcolor{' + E_INK.pagoda + '}{\\textbf{pagoda}}', 'e* = pagoda');

    /* ---- stage 3 : the updated representation, and the repeat ----------- */
    C.arrow(3, G.boxX + G.boxW + GAPA, G.mid);
    tokenBlock(3, G.b2, E.pagoda);
    C.tex(3, G.b2 + G.block / 2, G.tag, 70, 30, M_TAG,
      '\\mathbf{T}^{l+1}', 'T<sup>l+1</sup>');
    ellipsis(3, G.e2);
    C.add('mid', 3, 'path', {
      d: 'M ' + G.b1 + ' ' + G.bracket + ' V ' + (G.bracket + 8)
        + ' H ' + (G.b2 + G.block) + ' V ' + G.bracket,
      fill: 'none', stroke: '#b9c0c9', 'stroke-width': 1.4, 'stroke-linejoin': 'round'
    });
    C.txt(3, W / 2, G.note, 'One step, applied in every one of the 24 global attention '
      + 'blocks — all model weights stay frozen.',
      { size: T_SMALL, weight: 400, fill: MUTED, anchor: 'middle' });

    /* ---- stage 4 : the output, before over after ------------------------ */
    // Stacked rather than laid out in one row of eight: side by side the tiles
    // would be 66 units tall, stacked they are 150.
    ch.open(4, O.top, 'OUTPUT · BEFORE AND AFTER', TONE_D, F2.rO);
    var COLS = [
      { name: 'Reconstruction', file: 'recon_' },
      { name: 'Close-up', file: 'zoom_' },
      { name: 'Attention, view 1', file: 'attn_', suffix: '_1' },
      { name: 'Attention, view 2', file: 'attn_', suffix: '_2' }
    ];
    COLS.forEach(function (c, i) {
      c.x = O.x[i];
      c.w = O.ratios[i] * O.h;
      // Same spec as figure 1's column headings, and one baseline for the row.
      C.txt(4, c.x + c.w / 2, O.name, c.name,
        { size: T_HEAD, weight: 700, fill: INK, anchor: 'middle', row: 'f2head' });
    });

    // The two reconstructions are different cameras -- the pagoda sits higher
    // and larger in the second -- so one fixed window would not frame the same
    // thing twice. Each window's CENTRE was recovered by matching its own
    // close-up tile against its own reconstruction (normalised cross-
    // correlation, 0.98 and 0.96). The two windows came out within 1.5% of the
    // same size, so both are DRAWN at one size: two callouts into two
    // identically sized close-ups have no business being different.
    var CALL = 48;
    var SIDES = [
      { key: 'before', label: 'BEFORE', y: O.y, color: BEFORE, ink: BEFORE_INK,
        tint: BEFORE_TINT, cx: 0.3480 + 64 / 681, cy: 0.2400 + 64 / 400 },
      { key: 'after', label: 'AFTER', y: O.y2, color: AFTER, ink: AFTER_INK,
        tint: AFTER_TINT, cx: 0.3289 + 65 / 681, cy: 0.1725 + 65 / 400 }
    ];
    SIDES.forEach(function (s) {
      var zoom = COLS[1], rec = COLS[0];
      var clip = el('clipPath', { id: 'aw-f2-clip-' + s.key });
      clip.appendChild(el('rect', { x: zoom.x, y: s.y, width: zoom.w, height: O.h, rx: 8 }));
      C.defs.appendChild(clip);
      // The spine's left edge is x = EDGE, the same line as the section rule
      // and the headings: the figure has one left boundary.
      C.add('bg', 4, 'rect', { x: EDGE, y: s.y, width: O.spine, height: O.h, rx: 6, fill: s.tint });
      C.rot(4, EDGE + O.spine / 2, s.y + O.h / 2, s.label,
        { size: T_HEAD, weight: 700, fill: s.ink, ls: '0.12em' });

      function img(c, alt) {
        C.add('mid', 4, 'image', {
          x: c.x, y: s.y, width: c.w, height: O.h,
          href: IMG + c.file + s.key + (c.suffix || '') + '.jpg',
          preserveAspectRatio: 'xMidYMid slice', 'aria-label': alt,
          'clip-path': c === COLS[1] ? 'url(#aw-f2-clip-' + s.key + ')' : null
        });
      }
      // inset by half the stroke, so a frame's outer edge is the tile's edge
      // and a heavier frame does not grow the figure's bounding box
      function frame(c, rx, stroke, sw) {
        C.add('mid', 4, 'rect', {
          x: c.x + sw / 2, y: s.y + sw / 2, width: c.w - sw, height: O.h - sw,
          rx: rx, fill: 'none', stroke: stroke, 'stroke-width': sw
        });
      }

      img(rec, 'Reconstructed point cloud ' + s.key + ' the intervention');
      frame(rec, 4, BORDER, 1);
      var cx = rec.x + s.cx * rec.w - CALL / 2, cy = s.y + s.cy * O.h - CALL / 2;
      // The wedge is cut at the reconstruction's edge and painted only in the
      // gutter: over the point cloud it would be a wash across half the image.
      // Its two edges are the leader lines, sampled at that edge.
      var gx = rec.x + rec.w, t = (gx - cx - CALL) / (zoom.x - cx - CALL);
      C.add('mid', 4, 'polygon', {
        points: gx + ',' + (cy + t * (s.y - cy)) + ' ' + zoom.x + ',' + s.y + ' '
          + zoom.x + ',' + (s.y + O.h) + ' '
          + gx + ',' + (cy + CALL + t * (s.y + O.h - cy - CALL)),
        fill: s.color, 'fill-opacity': 0.16
      });
      [[cy, s.y], [cy + CALL, s.y + O.h]].forEach(function (p) {
        C.add('mid', 4, 'line', {
          x1: cx + CALL, y1: p[0], x2: zoom.x, y2: p[1],
          stroke: s.color, 'stroke-opacity': 0.55, 'stroke-width': 1.2
        });
      });
      C.add('mid', 4, 'rect', {
        x: cx, y: cy, width: CALL, height: CALL, rx: 3,
        fill: 'none', stroke: s.color, 'stroke-width': 2.6
      });
      img(zoom, 'Close-up of the eaves ' + s.key + ' the intervention');
      frame(zoom, 8, s.color, 3);
      img(COLS[2], 'Attention map for view 1 ' + s.key + ' the intervention');
      frame(COLS[2], 4, BORDER, 1);
      img(COLS[3], 'Attention map for view 2 ' + s.key + ' the intervention');
      frame(COLS[3], 4, BORDER, 1);
    });

    C.fit = ch.fit;
    return C;
  };

  /* ===================================================================== */
  /* the player: control bar, stepper, caption, adaptive crop              */
  /* ===================================================================== */

  function setCaption(node, parts) {
    node.textContent = '';
    parts.forEach(function (p) {
      if (typeof p === 'string') {
        node.appendChild(document.createTextNode(p));
      } else {
        var sp = document.createElement('span');
        sp.className = 'aw-pipe-cm';
        math(sp, p.tex, p.plain);
        node.appendChild(sp);
      }
    });
    typeset(node);
  }

  function mount(root, spec) {
    var figure = root.querySelector('.aw-pipe-figure');
    var strip = root.querySelector('.aw-pipe-steps');
    var cap = root.querySelector('.aw-pipe-caption');
    var playBtn = root.querySelector('.aw-pipe-play');
    var resetBtn = root.querySelector('.aw-pipe-reset');
    var slow = window.matchMedia
      ? window.matchMedia('(prefers-reduced-motion: reduce)') : { matches: false };
    var n = spec.stages.length;

    var C = spec.build();
    figure.appendChild(C.svg);
    // Both the drawing and the caption carry maths, and the caption is on
    // screen before KaTeX has finished loading.
    whenKatex(function () { return typeset(C.svg) && typeset(cap); });
    inkCentre(C.svg);
    if (document.fonts && document.fonts.ready) {
      document.fonts.ready.then(function () { inkCentre(C.svg); });
    } else {
      setTimeout(function () { inkCentre(C.svg); }, 900);
    }
    var ops = [].slice.call(C.svg.querySelectorAll('.aw-pipe-op'));
    var summary = cap.innerHTML;

    var buttons = spec.stages.map(function (st, i) {
      var b = document.createElement('button');
      b.type = 'button';
      b.dataset.step = String(i);
      b.dataset.state = 'overview';
      b.title = st.label;
      b.setAttribute('aria-label', 'Stage ' + (i + 1) + ': ' + st.label);
      b.textContent = st.label;
      b.addEventListener('click', function () { overview = false; at = i; pause(); });
      strip.appendChild(b);
      return b;
    });

    var overview = true, at = n - 1, running = false, raf = 0, last = 0, t = 1;
    function done() { return at === n - 1 && t >= 1; }

    function render() {
      ops.forEach(function (g) {
        var s = +g.getAttribute('data-stage');
        g.classList.toggle('is-on', overview || s <= at + 1);
      });
      // The crop, the aspect ratio and the box width are three halves of one
      // statement and have to change in the same frame, or the figure is
      // briefly stretched. The leftover width is split evenly left and right,
      // so a short step is centred rather than pushed against the left edge,
      // and the box is narrowed by the same fraction so the drawing holds its
      // scale instead of zooming.
      var i = overview ? n - 1 : at;
      var right = spec.right[i], depth = spec.depth[i];
      C.fit(i);
      var x0 = EDGE - MARGIN, vw = right - EDGE + 2 * MARGIN;
      C.svg.setAttribute('viewBox', x0 + ' 0 ' + vw + ' ' + depth);
      figure.style.setProperty('--figure-ratio', vw + '/' + depth);
      figure.style.setProperty('--figure-width', (100 * vw / W).toFixed(3) + '%');
      buttons.forEach(function (b, k) {
        b.dataset.state = overview ? 'overview'
          : k < at ? 'complete' : k === at ? 'current' : 'pending';
        if (!overview && k === at) b.setAttribute('aria-current', 'step');
        else b.removeAttribute('aria-current');
      });
      if (!overview) {
        var b = buttons[at];
        strip.scrollTo({
          left: Math.max(0, b.offsetLeft - (strip.clientWidth - b.offsetWidth) / 2),
          behavior: slow.matches ? 'instant' : 'smooth'
        });
      } else {
        strip.scrollTo({ left: 0, behavior: 'instant' });
      }
      var mode = running ? 'pause' : (!overview && done()) ? 'replay' : 'play';
      playBtn.querySelector('use').setAttribute('href', '#aw-i-' + mode);
      var word = mode === 'pause' ? 'Pause' : mode === 'replay' ? 'Replay' : 'Play';
      playBtn.querySelector('span').textContent = word;
      playBtn.setAttribute('aria-label', word + ' this figure');
      playBtn.title = word + ' this figure';
      if (overview) { cap.innerHTML = summary; typeset(cap); }
      else setCaption(cap, [(at + 1) + ' / ' + n + ' · '].concat(spec.stages[at].caption));
      root.style.setProperty('--tone', overview ? 'var(--aw-accent)' : spec.stages[at].tone);
    }

    function pause() { running = false; cancelAnimationFrame(raf); last = 0; render(); }
    function tick(now) {
      if (!running) return;
      if (last) t += Math.min(now - last, 100) / spec.stages[at].dwell;
      last = now;
      if (t >= 1) {
        t = 1;
        if (at === n - 1) { pause(); return; }
        at += 1; t = 0; render();
      }
      raf = requestAnimationFrame(tick);
    }

    playBtn.addEventListener('click', function () {
      if (running) { pause(); return; }
      if (overview || done()) { at = 0; t = 0; } else if (t >= 1) { t = 0; }
      overview = false;
      if (slow.matches) { at = n - 1; t = 1; render(); return; }
      running = true; last = 0; render();
      raf = requestAnimationFrame(tick);
    });
    // Reset returns to the complete figure, not to step one.
    resetBtn.addEventListener('click', function () {
      overview = true; at = n - 1; t = 1; pause();
    });

    strip.addEventListener('keydown', function (e) {
      var target = e.target.closest ? e.target.closest('[data-step]') : null;
      var i = buttons.indexOf(target);
      if (i < 0 || ['ArrowLeft', 'ArrowRight', 'Home', 'End'].indexOf(e.key) < 0) return;
      e.preventDefault();
      var k = e.key === 'Home' ? 0 : e.key === 'End' ? n - 1
        : (i + (e.key === 'ArrowRight' ? 1 : -1) + n) % n;
      buttons[k].focus({ preventScroll: true });
      buttons[k].click();
    });

    if ('IntersectionObserver' in window) {
      new IntersectionObserver(function (es) {
        if (!es[0].isIntersecting && running) pause();
      }, { threshold: 0.08 }).observe(root);
    }
    document.addEventListener('visibilitychange', function () {
      if (document.hidden && running) pause();
    });
    render();
    root.dataset.pipeReady = '1';
  }

  /* SVG <image> has no loading="lazy", so building eagerly would pull every
     tile on first paint for a figure well below the fold. The build is also
     held until the page font is up, because the column widths are solved from
     its metrics; 1.5 s is the ceiling, after which the baked fallbacks stand. */
  function whenFont(fn) {
    if (!document.fonts || !document.fonts.ready || document.fonts.status === 'loaded') {
      fn(); return;
    }
    var done = false;
    var go = function () { if (!done) { done = true; fn(); } };
    document.fonts.ready.then(go);
    setTimeout(go, 1500);
  }

  function boot() {
    [['archattend-fig1', function () { return F1; }],
      ['archattend-fig2', function () { return F2; }]].forEach(function (pair) {
      var root = document.getElementById(pair[0]);
      if (!root) return;
      var go = function () { whenFont(function () { mount(root, pair[1]()); }); };
      if (!('IntersectionObserver' in window)) { go(); return; }
      var io = new IntersectionObserver(function (entries) {
        if (!entries.some(function (e) { return e.isIntersecting; })) return;
        io.disconnect();
        go();
      }, { rootMargin: '400px' });
      io.observe(root);
    });
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
  // measurement hook for the layout harness; harmless in production
  window.__awPipe = { F1: F1, F2: F2, W: W, EDGE: EDGE, MARGIN: MARGIN, GAPA: GAPA,
    AR_LEN: AR_LEN, ink: ink };
})();
