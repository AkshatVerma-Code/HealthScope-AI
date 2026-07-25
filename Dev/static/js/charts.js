/* ============================================================
   charts.js — Gauge & Confidence Ring Charts (Canvas API)
   ============================================================ */

/**
 * Draw a semicircular gauge (180°) for risk scores.
 * @param {HTMLCanvasElement} canvas
 * @param {number} value  - 0 to 100
 * @param {string} status - 'High Risk' | 'Moderate Risk' | 'Low Risk'
 */
function drawRiskGauge(canvas, value, status) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width;
  const H = canvas.height;
  const cx = W / 2;
  const cy = H * 0.72;
  const R = W * 0.4;
  const startAngle = Math.PI;
  const endAngle = 2 * Math.PI;

  ctx.clearRect(0, 0, W, H);

  // Track background
  ctx.beginPath();
  ctx.arc(cx, cy, R, startAngle, endAngle);
  ctx.strokeStyle = 'rgba(255,255,255,0.08)';
  ctx.lineWidth = 12;
  ctx.lineCap = 'round';
  ctx.stroke();

  // Determine color
  let color;
  if (status === 'High Risk')     color = '#ef4444';
  else if (status === 'Moderate Risk') color = '#f59e0b';
  else                            color = '#10b981';

  // Filled arc
  const fillEnd = startAngle + (value / 100) * Math.PI;
  ctx.beginPath();
  ctx.arc(cx, cy, R, startAngle, fillEnd);
  ctx.strokeStyle = color;
  ctx.lineWidth = 12;
  ctx.lineCap = 'round';
  ctx.shadowColor = color;
  ctx.shadowBlur = 12;
  ctx.stroke();
  ctx.shadowBlur = 0;

  // Tick marks
  for (let i = 0; i <= 10; i++) {
    const angle = startAngle + (i / 10) * Math.PI;
    const x1 = cx + (R - 16) * Math.cos(angle);
    const y1 = cy + (R - 16) * Math.sin(angle);
    const x2 = cx + (R - 20) * Math.cos(angle);
    const y2 = cy + (R - 20) * Math.sin(angle);
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.strokeStyle = 'rgba(255,255,255,0.2)';
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }
}

/**
 * Animate gauge from 0 to target value.
 */
function animateGauge(canvas, targetValue, status) {
  let current = 0;
  const duration = 1200;
  const start = performance.now();

  function step(now) {
    const elapsed = now - start;
    const progress = Math.min(elapsed / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    current = targetValue * eased;
    drawRiskGauge(canvas, current, status);
    if (progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

/**
 * Draw a full confidence ring (360°) for DL predictions.
 * @param {HTMLCanvasElement} canvas
 * @param {number} value - 0 to 100
 */
function drawConfidenceRing(canvas, value) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width;
  const H = canvas.height;
  const cx = W / 2;
  const cy = H / 2;
  const R = Math.min(W, H) / 2 - 14;

  ctx.clearRect(0, 0, W, H);

  // Background ring
  ctx.beginPath();
  ctx.arc(cx, cy, R, 0, 2 * Math.PI);
  ctx.strokeStyle = 'rgba(255,255,255,0.08)';
  ctx.lineWidth = 14;
  ctx.stroke();

  // Colored arc
  const endAngle = -Math.PI / 2 + (value / 100) * 2 * Math.PI;
  const grad = ctx.createLinearGradient(0, 0, W, H);
  grad.addColorStop(0, '#00d4aa');
  grad.addColorStop(1, '#7b6cf6');
  ctx.beginPath();
  ctx.arc(cx, cy, R, -Math.PI / 2, endAngle);
  ctx.strokeStyle = grad;
  ctx.lineWidth = 14;
  ctx.lineCap = 'round';
  ctx.shadowColor = '#00d4aa';
  ctx.shadowBlur = 16;
  ctx.stroke();
  ctx.shadowBlur = 0;
}

/**
 * Animate the confidence ring.
 */
function animateConfidenceRing(canvas, targetValue) {
  let current = 0;
  const duration = 1400;
  const start = performance.now();

  function step(now) {
    const elapsed = now - start;
    const progress = Math.min(elapsed / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    current = targetValue * eased;
    drawConfidenceRing(canvas, current);
    if (progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

/**
 * Animate class score bars from left to right.
 */
function animateScoreBars() {
  const bars = document.querySelectorAll('.class-score-bar');
  bars.forEach((bar, i) => {
    const target = parseFloat(bar.dataset.value || 0);
    setTimeout(() => {
      bar.style.width = target + '%';
    }, i * 120);
  });
}

/* ── Initialize on DOM ready ─── */
document.addEventListener('DOMContentLoaded', function () {

  // Risk gauges
  document.querySelectorAll('.risk-gauge-canvas').forEach(canvas => {
    const value  = parseFloat(canvas.dataset.value || 0);
    const status = canvas.dataset.status || 'Low Risk';
    canvas.width  = 130;
    canvas.height = 100;
    animateGauge(canvas, value, status);
  });

  // Confidence rings
  document.querySelectorAll('.confidence-ring-canvas').forEach(canvas => {
    const value = parseFloat(canvas.dataset.value || 0);
    canvas.width  = 180;
    canvas.height = 180;
    animateConfidenceRing(canvas, value);
  });

  // Score bars
  animateScoreBars();
});
