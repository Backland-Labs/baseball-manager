/* charts.js -- Canvas-based WPA chart */

function drawWPAChart(canvasId, decisions) {
  'use strict';

  var canvas = document.getElementById(canvasId);
  if (!canvas) return;

  var ctx = canvas.getContext('2d');
  var W = canvas.width;
  var H = canvas.height;
  var pad = { top: 20, right: 20, bottom: 30, left: 50 };
  var plotW = W - pad.left - pad.right;
  var plotH = H - pad.top - pad.bottom;

  // Clear
  ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--bg-card').trim() || '#181b24';
  ctx.fillRect(0, 0, W, H);

  // No data check
  if (!decisions || decisions.length === 0) {
    ctx.fillStyle = '#7c829a';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('No decision data', W / 2, H / 2);
    return;
  }

  // Extract WP values from decisions
  // Decisions might have different shapes depending on the log source
  var wpValues = [];
  decisions.forEach(function (d, i) {
    var wp = 0.5;
    if (d.wp_before !== undefined) {
      wp = d.wp_before;
    } else if (d.game_state) {
      // Use inning/half to approximate position -- WP not in raw log
      wp = 0.5; // default fallback
    }
    wpValues.push({ index: i, wp: wp, isActive: d.is_active_decision || false });
  });

  // Axis helpers
  function xPos(i) { return pad.left + (i / Math.max(wpValues.length - 1, 1)) * plotW; }
  function yPos(wp) { return pad.top + (1 - wp) * plotH; }

  // Grid lines
  ctx.strokeStyle = '#2a2e3d';
  ctx.lineWidth = 1;
  [0, 0.25, 0.5, 0.75, 1.0].forEach(function (v) {
    var y = yPos(v);
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(W - pad.right, y);
    ctx.stroke();
  });

  // 0.50 baseline (stronger)
  ctx.strokeStyle = '#4f8ff7';
  ctx.lineWidth = 1.5;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(pad.left, yPos(0.5));
  ctx.lineTo(W - pad.right, yPos(0.5));
  ctx.stroke();
  ctx.setLineDash([]);

  // Y-axis labels
  ctx.fillStyle = '#7c829a';
  ctx.font = '10px sans-serif';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  [0, 0.25, 0.5, 0.75, 1.0].forEach(function (v) {
    ctx.fillText((v * 100).toFixed(0) + '%', pad.left - 6, yPos(v));
  });

  // X-axis label
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  ctx.fillText('Decision Turn', W / 2, H - 8);

  // Draw WP line
  if (wpValues.length > 1) {
    ctx.strokeStyle = '#e1e4eb';
    ctx.lineWidth = 2;
    ctx.beginPath();
    wpValues.forEach(function (pt, i) {
      var x = xPos(i);
      var y = yPos(pt.wp);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Color segments above/below 0.5
    for (var i = 1; i < wpValues.length; i++) {
      var prev = wpValues[i - 1];
      var curr = wpValues[i];
      ctx.strokeStyle = curr.wp >= 0.5 ? '#34d399' : '#f87171';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(xPos(i - 1), yPos(prev.wp));
      ctx.lineTo(xPos(i), yPos(curr.wp));
      ctx.stroke();
    }
  }

  // Draw dots for active decisions
  wpValues.forEach(function (pt) {
    if (!pt.isActive) return;
    var x = xPos(pt.index);
    var y = yPos(pt.wp);
    ctx.fillStyle = pt.wp >= 0.5 ? '#34d399' : '#f87171';
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#e1e4eb';
    ctx.lineWidth = 1;
    ctx.stroke();
  });
}
