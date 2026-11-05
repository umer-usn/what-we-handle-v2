/* ==========================================================================
   Curved card carousel — endless sideways scroller whose strip bows outward
   toward each edge (flat in the middle, tall and squeezed at the sides),
   with RGB colour-splitting that grows with speed.

   How it works
   - The three cards are painted once into an off-screen 2D canvas (a seamless,
     tileable strip), read from the real DOM so type, colours and spacing stay
     in sync with the CSS.
   - A WebGL fragment shader walks every output pixel backwards through the bend:
       · vertical:   srcY = y / s(x),  s = 1 + (S-1)·t²      → top/bottom curve outward
       · horizontal: srcX = x + dir·K·t^p                    → content squeezes toward the edge
     where t eases 0 → 1 from the end of the flat centre to the stage edge.
     R, G and B are sampled with slightly different horizontal bends → fringes.
   - The <ul> stays in the DOM for assistive tech. If WebGL isn't available the
     page falls back to the same carousel, flat, using plain CSS transforms.
   - Motion: slow drift (pauses on hover), drag / trackpad / arrow keys with momentum.
   ========================================================================== */
(() => {
  'use strict';

  const CFG = {
    drift: 38,        // px/s idle speed
    flat: 0.5,        // share of the half-width that stays perfectly flat
    height: 1.6,      // card height multiplier at the very edge
    squeeze: 95,      // horizontal squeeze at the very edge (px)
    squeezePow: 2,    // bend profile exponent
    splitBase: 0.025, // RGB split at rest (fraction of the squeeze)
    splitSpeed: 0.11, // extra split at full speed
    friction: 2.4,    // how quickly momentum settles back to the drift
    maxSpeed: 3000,   // px/s clamp
  };

  const stage = document.querySelector('[data-gc="stage"]');
  if (!stage) return;
  const canvas = stage.querySelector('[data-gc="canvas"]');
  const track = stage.querySelector('[data-gc="track"]');
  const originals = Array.from(track.children);
  const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* ---- State --------------------------------------------------------------- */
  let W = 0, H = 0, Hc = 0, setW = 0, gap = 0;
  let pos = 0, vel = reduceMotion ? 0 : CFG.drift;
  let dragging = false, hovering = false, lastX = 0, lastT = 0;
  let gl = null, prog = null, tex = null, U = null;
  let fallbackCards = null;
  const TEX_PAD = 2;                        // transparent rows above/below the cards → smooth curved edges

  const rest = () => (reduceMotion || hovering ? 0 : CFG.drift);

  /* ---- WebGL setup --------------------------------------------------------- */
  const VERT = 'attribute vec2 a;void main(){gl_Position=vec4(a,0.,1.);}';
  const FRAG = `
    precision highp float;
    uniform sampler2D uTex;
    uniform vec2  uRes;     // drawing-buffer size (device px)
    uniform vec2  uSize;    // canvas size (css px)
    uniform float uSetW;    // strip width (css px)
    uniform float uH;       // card height (css px)
    uniform float uPad;
    uniform float uPos;     // scroll offset (css px)
    uniform float uFlat, uS, uK, uPow;
    uniform vec3  uSplit;   // per-channel strength of the horizontal bend

    vec4 tap(float sx, float srcY) {
      float u = fract((uPos + sx) / uSetW);
      float v = (uPad + uH * 0.5 - srcY) / (uH + 2.0 * uPad);
      if (v < 0.0 || v > 1.0) return vec4(0.0);
      return texture2D(uTex, vec2(u, v));
    }

    void main() {
      vec2  p  = gl_FragCoord.xy / uRes * uSize;       // css px, origin bottom-left
      float yc = p.y - uSize.y * 0.5;                  // up = +
      float u  = p.x / uSize.x * 2.0 - 1.0;            // -1 … 1
      float t  = clamp((abs(u) - uFlat) / (1.0 - uFlat), 0.0, 1.0);
      float s  = 1.0 + (uS - 1.0) * t * t;             // vertical scale at this x
      float shift = sign(u) * uK * pow(t, uPow);       // horizontal squeeze
      float srcY = yc / s;

      vec4 r = tap(p.x + shift * uSplit.r, srcY);
      vec4 g = tap(p.x + shift * uSplit.g, srcY);
      vec4 b = tap(p.x + shift * uSplit.b, srcY);
      float a = (r.a + g.a + b.a) / 3.0;
      gl_FragColor = vec4(min(vec3(r.r, g.g, b.b), vec3(a)), a);   // premultiplied
    }`;

  function initGL() {
    gl = canvas.getContext('webgl', { alpha: true, premultipliedAlpha: true, antialias: false });
    if (!gl) return false;
    const sh = (type, src) => {
      const s = gl.createShader(type);
      gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      return s;
    };
    try {
      prog = gl.createProgram();
      gl.attachShader(prog, sh(gl.VERTEX_SHADER, VERT));
      gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FRAG));
      gl.linkProgram(prog);
      if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
    } catch (e) { console.warn('[glass-cards] shader failed, using flat fallback:', e); return false; }
    gl.useProgram(prog);

    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(prog, 'a');
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

    U = {};
    ['uTex', 'uRes', 'uSize', 'uSetW', 'uH', 'uPad', 'uPos', 'uFlat', 'uS', 'uK', 'uPow', 'uSplit']
      .forEach((n) => { U[n] = gl.getUniformLocation(prog, n); });

    tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
    gl.disable(gl.DEPTH_TEST);
    gl.clearColor(0, 0, 0, 0);
    return true;
  }

  /* ---- Paint the strip texture from the DOM -------------------------------- */
  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  function wrapLines(ctx, text, maxW) {
    const words = text.trim().split(/\s+/);
    const lines = [];
    let cur = '';
    for (const w of words) {
      const next = cur ? cur + ' ' + w : w;
      if (cur && ctx.measureText(next).width > maxW) { lines.push(cur); cur = w; }
      else cur = next;
    }
    if (cur) lines.push(cur);
    return lines;
  }

  function buildTexture() {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const ts = Math.min(3, Math.max(2, dpr * 1.5));     // texture px per css px (curve magnifies up to ~1.6×)
    const first = originals[0];
    const cs = getComputedStyle(first);
    const cw = first.offsetWidth;
    gap = parseFloat(getComputedStyle(track).columnGap) || 0;
    setW = (cw + gap) * originals.length;

    const c = document.createElement('canvas');
    c.width = Math.ceil(setW * ts);
    c.height = Math.ceil((H + 2 * TEX_PAD) * ts);
    const ctx = c.getContext('2d');
    ctx.scale(ts, ts);

    const radius = parseFloat(cs.borderTopLeftRadius) || 12;
    const bw = parseFloat(cs.borderTopWidth) || 1;
    const padL = parseFloat(cs.paddingLeft) || 24;
    const padT = parseFloat(cs.paddingTop) || 24;
    const inner = cw - 2 * bw - 2 * padL;

    originals.forEach((card, i) => {
      const x = gap / 2 + i * (cw + gap);               // half a gap of air at each end → seam sits in a gap
      const y = TEX_PAD;
      ctx.fillStyle = cs.borderTopColor;
      roundRect(ctx, x, y, cw, H, radius); ctx.fill();
      ctx.fillStyle = cs.backgroundColor;
      roundRect(ctx, x + bw, y + bw, cw - 2 * bw, H - 2 * bw, radius - bw); ctx.fill();

      const img = card.querySelector('img');
      const p = card.querySelector('p');
      const ix = x + bw + padL, iy = y + bw + padT;
      if (img && img.complete && img.naturalWidth) ctx.drawImage(img, ix, iy, 32, 32);

      const ps = getComputedStyle(p);
      const fs = parseFloat(ps.fontSize);
      const lh = parseFloat(ps.lineHeight) || fs * 1.2;
      ctx.font = `${ps.fontWeight} ${fs}px ${ps.fontFamily}`;
      if ('letterSpacing' in ctx) ctx.letterSpacing = ps.letterSpacing;
      ctx.fillStyle = ps.color;
      ctx.textBaseline = 'alphabetic';
      const m = ctx.measureText('Hg');
      const asc = m.fontBoundingBoxAscent ?? fs * 0.97;
      const desc = m.fontBoundingBoxDescent ?? fs * 0.24;
      const baseOffset = (lh - (asc + desc)) / 2 + asc;
      const top = iy + 32 + (parseFloat(getComputedStyle(card).rowGap) || 24);
      wrapLines(ctx, p.textContent, inner).forEach((line, li) => {
        ctx.fillText(line, ix, top + li * lh + baseOffset);
      });
    });

    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, c);
  }

  /* ---- Sizing -------------------------------------------------------------- */
  function resize() {
    W = stage.clientWidth;
    H = stage.clientHeight;
    if (!gl) { measureFallback(); return; }
    Hc = Math.ceil(H * CFG.height + 8);
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    canvas.style.width = W + 'px';
    canvas.style.height = Hc + 'px';
    canvas.style.top = -(Hc - H) / 2 + 'px';
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(Hc * dpr);
    gl.viewport(0, 0, canvas.width, canvas.height);
    buildTexture();
  }

  /* ---- Input --------------------------------------------------------------- */
  const clampV = (v) => Math.max(-CFG.maxSpeed, Math.min(CFG.maxSpeed, v));

  stage.addEventListener('pointerdown', (e) => {
    if (e.pointerType === 'mouse' && e.button !== 0) return;
    dragging = true;
    lastX = e.clientX; lastT = performance.now();
    vel = 0;
    stage.classList.add('is-dragging');
    stage.setPointerCapture(e.pointerId);
  });
  stage.addEventListener('pointermove', (e) => {
    if (!dragging) return;
    const now = performance.now();
    const dx = e.clientX - lastX;
    const dt = Math.max(1, now - lastT) / 1000;
    pos -= dx;
    vel = vel * 0.6 + (-dx / dt) * 0.4;              // smoothed release velocity
    lastX = e.clientX; lastT = now;
  });
  const endDrag = () => {
    if (!dragging) return;
    dragging = false;
    stage.classList.remove('is-dragging');
    if (performance.now() - lastT > 80) vel = rest(); // paused before release → no fling
    vel = clampV(vel);
  };
  stage.addEventListener('pointerup', endDrag);
  stage.addEventListener('pointercancel', endDrag);
  stage.addEventListener('pointerenter', (e) => { if (e.pointerType === 'mouse') hovering = true; });
  stage.addEventListener('pointerleave', () => { hovering = false; });

  // Trackpad swipe / shift+wheel. Plain vertical wheel keeps scrolling the page.
  stage.addEventListener('wheel', (e) => {
    const dx = Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : (e.shiftKey ? e.deltaY : 0);
    if (!dx) return;
    e.preventDefault();
    pos += dx;
    vel = clampV(dx * 40);
  }, { passive: false });

  stage.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowRight') { vel = 900; e.preventDefault(); }
    if (e.key === 'ArrowLeft')  { vel = -900; e.preventDefault(); }
  });

  /* ---- Flat fallback (no WebGL) -------------------------------------------- */
  function startFallback() {
    for (let i = 1; i < 4; i++) {
      originals.forEach((c) => {
        const clone = c.cloneNode(true);
        clone.setAttribute('aria-hidden', 'true');
        track.appendChild(clone);
      });
    }
  }
  function measureFallback() {
    gap = parseFloat(getComputedStyle(track).columnGap) || 0;
    setW = (originals[0].offsetWidth + gap) * originals.length;
  }

  /* ---- Frame loop ---------------------------------------------------------- */
  let prev = 0;
  let visible = true;
  new IntersectionObserver((es) => { visible = es[0].isIntersecting; }).observe(stage);

  function frame(now) {
    const dt = Math.min(0.05, (now - prev) / 1000 || 0);   // clamp so a hidden tab doesn't jump
    prev = now;
    if (visible) {
      if (!dragging) {
        vel += (rest() - vel) * (1 - Math.exp(-dt * CFG.friction));
        pos += vel * dt;
      }
      const wrapped = ((pos % setW) + setW) % setW;

      if (gl) {
        const speed = Math.min(1, Math.abs(vel) / 1500);
        const d = CFG.splitBase + CFG.splitSpeed * speed;
        gl.uniform1i(U.uTex, 0);
        gl.uniform2f(U.uRes, canvas.width, canvas.height);
        gl.uniform2f(U.uSize, W, Hc);
        gl.uniform1f(U.uSetW, setW);
        gl.uniform1f(U.uH, H);
        gl.uniform1f(U.uPad, TEX_PAD);
        gl.uniform1f(U.uPos, wrapped + gap / 2);          // card 1 starts flush with the left edge at pos 0
        gl.uniform1f(U.uFlat, CFG.flat);
        gl.uniform1f(U.uS, CFG.height);
        gl.uniform1f(U.uK, CFG.squeeze);
        gl.uniform1f(U.uPow, CFG.squeezePow);
        gl.uniform3f(U.uSplit, 1 + d, 1, 1 - d);          // red bends most, blue least
        gl.clear(gl.COLOR_BUFFER_BIT);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      } else {
        track.style.transform = `translate3d(${(-setW - wrapped).toFixed(2)}px,0,0)`;
      }
    }
    requestAnimationFrame(frame);
  }

  async function init() {
    try { await document.fonts.load('400 16px Inter'); } catch (_) {}
    await Promise.all(originals.map((c) => {
      const img = c.querySelector('img');
      return img && !img.complete ? img.decode().catch(() => {}) : null;
    }));

    if (initGL()) stage.classList.add('is-gl');
    else { gl = null; startFallback(); }
    resize();
    requestAnimationFrame(frame);

    let rz;
    new ResizeObserver(() => { clearTimeout(rz); rz = setTimeout(resize, 80); }).observe(stage);
  }
  init();
})();
