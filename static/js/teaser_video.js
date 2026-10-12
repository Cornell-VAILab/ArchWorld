// Teaser video behaves like a GIF: autoplays muted and loops, with no controls. It pauses
// while scrolled off screen (saves CPU), and stays on its first frame (the figure) for
// visitors who prefer reduced motion.
document.addEventListener('DOMContentLoaded', () => {
  const video = document.getElementById('teaser-video');
  if (!video) return;
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    video.removeAttribute('autoplay');
    video.pause();
    return;
  }
  if (!('IntersectionObserver' in window)) return;
  new IntersectionObserver(([entry]) => {
    if (entry.isIntersecting) video.play().catch(() => {});
    else video.pause();
  }, { threshold: 0.25 }).observe(video);
});
