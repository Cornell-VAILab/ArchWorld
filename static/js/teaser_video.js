// Teaser video behaves like a GIF: autoplays muted and loops, with no controls. It pauses
// while scrolled off screen (saves CPU), and stays on its first frame (the figure) for
// visitors who prefer reduced motion.
//
// The file is kept at full resolution, so it is the heaviest thing on the page.
// Rather than shrink it, the source is attached only after the window has
// loaded and the browser is idle: the poster is the teaser figure, so the
// section looks finished from the first paint, and the video streams in behind
// everything else instead of ahead of it.
document.addEventListener('DOMContentLoaded', () => {
  const video = document.getElementById('teaser-video');
  if (!video) return;

  // Reduced motion: never fetch it at all, just leave the poster up.
  if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;

  const idle = (fn) => ('requestIdleCallback' in window
    ? requestIdleCallback(fn, { timeout: 3000 })
    : setTimeout(fn, 500));

  const start = () => idle(() => {
    const src = video.dataset.src;
    if (!src || video.currentSrc) return;
    video.preload = 'auto';
    video.src = src;
    video.load();
    const play = () => video.play().catch(() => {});
    if (!('IntersectionObserver' in window)) { play(); return; }
    new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) play(); else video.pause();
    }, { threshold: 0.25 }).observe(video);
  });

  if (document.readyState === 'complete') start();
  else window.addEventListener('load', start, { once: true });
});
