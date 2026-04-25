import asyncio
import base64
import binascii
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Ostalb-Hack Image Receiver")


class ImagePayload(BaseModel):
    image: str  # base64-encoded JPEG


class PointsPayload(BaseModel):
    # Liste von [x, y] mit Werten zwischen 0 und 1.
    points: list[tuple[float, float]]


# Aktuell zu projizierende Punkte (normalisiert, 0..1).
_current_points: list[tuple[float, float]] = []

# WebSocket-Clients, die Punkt-Updates abonnieren.
_ws_clients: set[WebSocket] = set()
_ws_lock = threading.Lock()
_main_loop: asyncio.AbstractEventLoop | None = None


@app.on_event("startup")
async def _capture_loop() -> None:
    global _main_loop
    _main_loop = asyncio.get_running_loop()


async def _broadcast_points(points: list[tuple[float, float]]) -> None:
    msg = {"points": [list(p) for p in points]}
    with _ws_lock:
        clients = list(_ws_clients)
    dead: list[WebSocket] = []
    for ws in clients:
        try:
            await ws.send_json(msg)
        except Exception:
            dead.append(ws)
    if dead:
        with _ws_lock:
            for ws in dead:
                _ws_clients.discard(ws)


def _schedule_broadcast(points: list[tuple[float, float]]) -> None:
    loop = _main_loop
    if loop is None:
        return
    asyncio.run_coroutine_threadsafe(_broadcast_points(points), loop)


INDEX_HTML = """<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8" />
  <title>Ostalb-Hack Projector</title>
  <style>
    html, body {
      margin: 0;
      padding: 0;
      height: 100%;
      width: 100%;
      background: #000;
      overflow: hidden;
      font-family: system-ui, sans-serif;
      color: #fff;
    }
    #wrap {
      position: fixed;
      inset: 0;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    canvas {
      display: block;
      width: 100vw;
      height: 100vh;
    }
    #help {
      position: fixed;
      top: 8px;
      left: 8px;
      padding: 6px 10px;
      background: rgba(0,0,0,0.55);
      border: 1px solid #444;
      border-radius: 4px;
      font-size: 12px;
      line-height: 1.4;
      pointer-events: none;
      z-index: 10;
    }
    #help b { color: #ffd34d; }
  </style>
</head>
<body>
  <div id="wrap"><canvas id="stage"></canvas></div>
  <div id="help">
    <b>C</b>: Kalibrierung an/aus &nbsp; <b>R</b>: Reset &nbsp;
    <b>1-4</b>: Ecke wählen &nbsp; <b>Pfeile</b>: feinjustieren (Shift = grob)
    <br/>im Kalibriermodus: Ecken mit Maus ziehen
  </div>
<script>
  // Logisches Seitenverhältnis (Breite:Höhe). Die Eingabepunkte liegen in
  // einem normalisierten 2:3-Rechteck; nach Kalibrierung werden sie per
  // Homographie auf die vier physischen Eckpunkte gemappt.
  const ASPECT_W = 2;
  const ASPECT_H = 3;

  // Reale Größe des erkannten Bereichs (zwischen den ArUco-Markern) in mm.
  // Muss zum REGION_ASPECT in vision.py passen (aktuell 87/47).
  const REGION_W_MM = 870;
  const REGION_H_MM = 470;

  const canvas = document.getElementById('stage');
  const ctx = canvas.getContext('2d');
  let points = [];
  let cssW = 0, cssH = 0;

  // Kalibrierung: vier Ecken in normalisierten Canvas-Koordinaten (0..1).
  // Reihenfolge: 0=TL, 1=TR, 2=BR, 3=BL.
  const DEFAULT_CORNERS = [[0,0],[1,0],[1,1],[0,1]];
  let corners = loadCorners();
  // Kalibriermodus kann per URL-Parameter ?calibrate=1 (oder ?cal=1) erzwungen
  // werden – nützlich, wenn die Seite im Kiosk-Modus geöffnet wird.
  const _qs = new URLSearchParams(window.location.search);
  let calibrating = ['1','true','yes','on'].includes(
    (_qs.get('calibrate') || _qs.get('cal') || '').toLowerCase()
  );
  let dragIdx = -1;
  let selectedIdx = 0;
  let H = null; // 3x3-Homographie als Flatten-Array (a,b,c,d,e,f,g,h,1)

  function loadCorners() {
    try {
      const raw = localStorage.getItem('projector.corners');
      if (raw) {
        const arr = JSON.parse(raw);
        if (Array.isArray(arr) && arr.length === 4) return arr.map(p => [+p[0], +p[1]]);
      }
    } catch (e) {}
    return DEFAULT_CORNERS.map(p => p.slice());
  }
  function saveCorners() {
    localStorage.setItem('projector.corners', JSON.stringify(corners));
  }

  // Homographie vom Einheitsquadrat (0,0)(1,0)(1,1)(0,1) auf die vier Zielpunkte.
  // Ausgabe in CSS-Pixeln. Standardformel nach Heckbert.
  function buildHomography() {
    const [p0, p1, p2, p3] = corners.map(p => [p[0]*cssW, p[1]*cssH]);
    const dx1 = p1[0]-p2[0], dx2 = p3[0]-p2[0], sx = p0[0]-p1[0]+p2[0]-p3[0];
    const dy1 = p1[1]-p2[1], dy2 = p3[1]-p2[1], sy = p0[1]-p1[1]+p2[1]-p3[1];
    let g, h;
    const det = dx1*dy2 - dx2*dy1;
    if (Math.abs(sx) < 1e-9 && Math.abs(sy) < 1e-9) {
      g = 0; h = 0;
    } else if (Math.abs(det) < 1e-9) {
      g = 0; h = 0;
    } else {
      g = (sx*dy2 - sy*dx2) / det;
      h = (dx1*sy - dy1*sx) / det;
    }
    const a = p1[0]-p0[0] + g*p1[0];
    const b = p3[0]-p0[0] + h*p3[0];
    const c = p0[0];
    const d = p1[1]-p0[1] + g*p1[1];
    const e = p3[1]-p0[1] + h*p3[1];
    const f = p0[1];
    H = [a,b,c,d,e,f,g,h,1];
  }
  function applyH(u, v) {
    const w = H[6]*u + H[7]*v + 1;
    return [(H[0]*u + H[1]*v + H[2]) / w,
            (H[3]*u + H[4]*v + H[5]) / w];
  }

  function resize() {
    // Canvas füllt den kompletten Bildschirm.
    cssW = Math.floor(window.innerWidth);
    cssH = Math.floor(window.innerHeight);

    const dpr = window.devicePixelRatio || 1;
    canvas.style.width = cssW + 'px';
    canvas.style.height = cssH + 'px';
    canvas.width = Math.floor(cssW * dpr);
    canvas.height = Math.floor(cssH * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    buildHomography();
    draw();
  }

  function draw() {
    // Hintergrund
    ctx.fillStyle = calibrating ? '#003a8c' : '#000';
    ctx.fillRect(0, 0, cssW, cssH);

    // Datenpunkte (durch Homographie gemappt).
    const r = Math.max(6, Math.min(cssW, cssH) * 0.015);
    const mapped = points.map(p => applyH(p[0], p[1]));

    // Linien zwischen den Punkten: pro 4er-Gruppe ein geschlossenes Viereck.
    if (mapped.length >= 2) {
      ctx.strokeStyle = '#ff2a2a';
      ctx.lineWidth = Math.max(2, r * 0.35);
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      for (let i = 0; i < mapped.length; i += 4) {
        const quad = mapped.slice(i, i + 4);
        if (quad.length < 2) break;
        ctx.beginPath();
        ctx.moveTo(quad[0][0], quad[0][1]);
        for (let j = 1; j < quad.length; j++) {
          ctx.lineTo(quad[j][0], quad[j][1]);
        }
        if (quad.length === 4) ctx.closePath();
        ctx.stroke();
      }
    }

    ctx.fillStyle = '#ff2a2a';
    for (const [x, y] of mapped) {
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
    }

    // Maße in mm an die Kanten jedes 4er-Vierecks schreiben.
    if (points.length >= 4) {
      const fontPx = Math.max(12, Math.round(r * 1.4));
      ctx.font = `bold ${fontPx}px system-ui, sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      for (let i = 0; i + 4 <= points.length; i += 4) {
        const quadN = points.slice(i, i + 4);   // normalisiert (0..1)
        const quadP = mapped.slice(i, i + 4);    // CSS-Pixel
        // Schwerpunkt des Vierecks (für Beschriftungs-Offset nach innen).
        let cx = 0, cy = 0;
        for (const [x, y] of quadP) { cx += x; cy += y; }
        cx /= 4; cy /= 4;
        for (let j = 0; j < 4; j++) {
          const k = (j + 1) % 4;
          const dxN = (quadN[k][0] - quadN[j][0]) * REGION_W_MM;
          const dyN = (quadN[k][1] - quadN[j][1]) * REGION_H_MM;
          const mm = Math.hypot(dxN, dyN);
          const label = `${mm.toFixed(0)} mm`;

          const mxp = (quadP[j][0] + quadP[k][0]) / 2;
          const myp = (quadP[j][1] + quadP[k][1]) / 2;

          // Richtung der Kante in Pixeln (zum Drehen der Schrift).
          const ex = quadP[k][0] - quadP[j][0];
          const ey = quadP[k][1] - quadP[j][1];
          let angle = Math.atan2(ey, ex);
          // Schrift nicht "auf dem Kopf" zeigen.
          if (angle > Math.PI / 2) angle -= Math.PI;
          if (angle < -Math.PI / 2) angle += Math.PI;

          // Senkrecht zur Kante nach innen (Richtung Schwerpunkt) versetzen,
          // damit die Linie nicht überschrieben wird.
          const elen = Math.hypot(ex, ey) || 1;
          let nx = -ey / elen, ny = ex / elen;       // Normale zur Kante
          if ((cx - mxp) * nx + (cy - myp) * ny < 0) { nx = -nx; ny = -ny; }
          const off = fontPx * 0.85;
          const tx = mxp + nx * off;
          const ty = myp + ny * off;

          ctx.save();
          ctx.translate(tx, ty);
          ctx.rotate(angle);
          const tw = ctx.measureText(label).width;
          ctx.fillStyle = 'rgba(0,0,0,0.6)';
          ctx.fillRect(-tw / 2 - 4, -fontPx / 2 - 2, tw + 8, fontPx + 4);
          ctx.fillStyle = '#ffd34d';
          ctx.fillText(label, 0, 0);
          ctx.restore();
        }
      }
    }

    if (calibrating) drawCalibrationOverlay();
  }

  function drawCalibrationOverlay() {
    // Rahmen entlang der Ecken.
    ctx.strokeStyle = '#ffd34d';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < 4; i++) {
      const x = corners[i][0]*cssW, y = corners[i][1]*cssH;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.stroke();

    // Diagonalen + Mittelkreuz als visuelle Hilfe.
    ctx.strokeStyle = 'rgba(255,211,77,0.4)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(corners[0][0]*cssW, corners[0][1]*cssH);
    ctx.lineTo(corners[2][0]*cssW, corners[2][1]*cssH);
    ctx.moveTo(corners[1][0]*cssW, corners[1][1]*cssH);
    ctx.lineTo(corners[3][0]*cssW, corners[3][1]*cssH);
    ctx.stroke();

    // Eckpunkte mit Nummer.
    const labels = ['1 TL', '2 TR', '3 BR', '4 BL'];
    for (let i = 0; i < 4; i++) {
      const x = corners[i][0]*cssW, y = corners[i][1]*cssH;
      const sel = (i === selectedIdx);
      ctx.fillStyle = sel ? '#ffd34d' : '#ffffff';
      ctx.beginPath();
      ctx.arc(x, y, sel ? 14 : 10, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = '#000';
      ctx.font = 'bold 12px system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(String(i+1), x, y);
      // Beschriftung neben dem Punkt
      ctx.fillStyle = '#ffd34d';
      ctx.font = '12px system-ui, sans-serif';
      ctx.textAlign = (i === 1 || i === 2) ? 'right' : 'left';
      const dx = (i === 1 || i === 2) ? -18 : 18;
      ctx.fillText(labels[i], x + dx, y);
    }
  }

  // ---- Maus-Interaktion ----
  function eventCanvasPos(ev) {
    const rect = canvas.getBoundingClientRect();
    return [ev.clientX - rect.left, ev.clientY - rect.top];
  }
  canvas.addEventListener('mousedown', (ev) => {
    if (!calibrating) return;
    const [mx, my] = eventCanvasPos(ev);
    let best = -1, bestD = 30; // Pixel-Toleranz
    for (let i = 0; i < 4; i++) {
      const dx = corners[i][0]*cssW - mx;
      const dy = corners[i][1]*cssH - my;
      const d = Math.hypot(dx, dy);
      if (d < bestD) { bestD = d; best = i; }
    }
    if (best >= 0) {
      dragIdx = best;
      selectedIdx = best;
      draw();
    }
  });
  canvas.addEventListener('mousemove', (ev) => {
    if (dragIdx < 0) return;
    const [mx, my] = eventCanvasPos(ev);
    corners[dragIdx][0] = Math.max(0, Math.min(1, mx / cssW));
    corners[dragIdx][1] = Math.max(0, Math.min(1, my / cssH));
    buildHomography();
    draw();
  });
  function endDrag() {
    if (dragIdx >= 0) { saveCorners(); dragIdx = -1; }
  }
  window.addEventListener('mouseup', endDrag);
  canvas.addEventListener('mouseleave', endDrag);

  // ---- Tastatur ----
  window.addEventListener('keydown', (ev) => {
    const k = ev.key.toLowerCase();
    if (k === 'c') {
      calibrating = !calibrating;
      draw();
      return;
    }
    if (k === 'r') {
      corners = DEFAULT_CORNERS.map(p => p.slice());
      saveCorners();
      buildHomography();
      draw();
      return;
    }
    if (!calibrating) return;
    if (k >= '1' && k <= '4') {
      selectedIdx = parseInt(k, 10) - 1;
      draw();
      return;
    }
    const step = (ev.shiftKey ? 10 : 1);
    let dx = 0, dy = 0;
    if (ev.key === 'ArrowLeft')  dx = -step;
    if (ev.key === 'ArrowRight') dx =  step;
    if (ev.key === 'ArrowUp')    dy = -step;
    if (ev.key === 'ArrowDown')  dy =  step;
    if (dx || dy) {
      ev.preventDefault();
      corners[selectedIdx][0] = Math.max(0, Math.min(1, corners[selectedIdx][0] + dx / cssW));
      corners[selectedIdx][1] = Math.max(0, Math.min(1, corners[selectedIdx][1] + dy / cssH));
      buildHomography();
      saveCorners();
      draw();
    }
  });

  // ---- WebSocket: laufende Punkt-Updates ----
  // Stabilisierung: neue Vierecke werden per Schwerpunkt-Distanz auf die
  // bisherigen abgebildet; jeder Eckpunkt wird über einen Tiefpass geglättet
  // und nur übernommen, wenn die Bewegung eine Totzone überschreitet.
  // Dadurch zittern die Brett-Konturen nicht mehr im Takt der Erkennung.
  const SMOOTH_ALPHA = 0.25;        // Anteil des neuen Werts (0..1, klein = ruhiger)
  const DEAD_ZONE = 0.004;          // < 0.4 % vom Bild -> ignorieren (Rauschen)
  const SNAP_DIST = 0.06;           // > 6 %  -> direkt übernehmen (echte Bewegung)
  const HOLD_MS = 1500;             // wie lange ein Viereck gehalten wird, das gerade nicht erkannt wurde
  function chunkQuads(arr) {
    const out = [];
    for (let i = 0; i + 4 <= arr.length; i += 4) out.push(arr.slice(i, i + 4));
    return out;
  }
  function centroid(quad) {
    let cx = 0, cy = 0;
    for (const [x, y] of quad) { cx += x; cy += y; }
    return [cx / 4, cy / 4];
  }
  // Beste Eck-Rotation (0..3) finden, sodass die Punkte bestmöglich passen.
  function bestRotation(prev, next) {
    let bestRot = 0, bestSum = Infinity;
    for (let r = 0; r < 4; r++) {
      let sum = 0;
      for (let i = 0; i < 4; i++) {
        const a = prev[i], b = next[(i + r) % 4];
        sum += Math.hypot(a[0] - b[0], a[1] - b[1]);
      }
      if (sum < bestSum) { bestSum = sum; bestRot = r; }
    }
    return bestRot;
  }
  function smoothPoint(prev, next) {
    const dx = next[0] - prev[0], dy = next[1] - prev[1];
    const d = Math.hypot(dx, dy);
    if (d < DEAD_ZONE) return prev;            // zu klein: ignorieren
    if (d > SNAP_DIST) return next;            // große Bewegung: direkt
    return [prev[0] + dx * SMOOTH_ALPHA, prev[1] + dy * SMOOTH_ALPHA];
  }

  // Tracker: jedes Viereck behält über Frames hinweg seinen Slot, auch
  // wenn es einzelne Frames lang nicht erkannt wird. Das verhindert das
  // kurzzeitige "Ausblenden" der Projektion, sobald die Erkennung mal
  // einen Frame aussetzt – solange überhaupt etwas erkannt wird (oder
  // die HOLD_MS-Frist nicht abgelaufen ist), bleibt das letzte bekannte
  // Viereck stehen.
  let tracked = [];   // Array von { pts: [[x,y]*4], lastSeenAt: ms }

  // Animation: zwischen WebSocket-Updates wird kontinuierlich (per
  // requestAnimationFrame) zwischen den aktuell gezeichneten Punkten
  // (`points`) und dem zuletzt empfangenen Ziel (`targetPoints`)
  // interpoliert. Dadurch entstehen weiche Bewegungen statt Ruckler im
  // 5-fps-Takt der Kamera.
  let targetPoints = [];
  const ANIM_ALPHA = 0.18;          // Annäherung pro Frame (0..1)
  const ANIM_SNAP_DIST = 0.08;      // große Sprünge nicht weichzeichnen
  const ANIM_MIN_STEP = 0.0008;     // unter diesem Rest-Delta direkt einrasten
  function stabilizePoints(incoming) {
    const newQuads = chunkQuads(incoming);
    const now = (typeof performance !== 'undefined' ? performance.now() : Date.now());

    // 1) Bestehende Tracks per Centroid-Distanz greedy zu neuen Vierecken
    //    zuordnen. Tracks behalten ihren Slot/Index – so bleibt die
    //    Reihenfolge der Ausgabe stabil (wichtig für die Animations-
    //    interpolation, die per Index arbeitet).
    const trackCent = tracked.map(t => centroid(t.pts));
    const usedTrack = new Array(tracked.length).fill(false);
    const newToTrack = new Array(newQuads.length).fill(-1);
    const order = newQuads.map((q, i) => ({ i, c: centroid(q) }));
    for (const { i, c } of order) {
      let best = -1, bestD = Infinity;
      for (let t = 0; t < tracked.length; t++) {
        if (usedTrack[t]) continue;
        const d = Math.hypot(c[0] - trackCent[t][0], c[1] - trackCent[t][1]);
        if (d < bestD) { bestD = d; best = t; }
      }
      if (best >= 0 && bestD < 0.25) { newToTrack[i] = best; usedTrack[best] = true; }
    }

    // 2) Zugeordnete Tracks aktualisieren (mit Eckrotation + Glättung).
    for (let i = 0; i < newQuads.length; i++) {
      const tIdx = newToTrack[i];
      if (tIdx < 0) continue;
      const prev = tracked[tIdx].pts;
      const rot = bestRotation(prev, newQuads[i]);
      const aligned = Array.from({ length: 4 }, (_, k) => newQuads[i][(k + rot) % 4]);
      tracked[tIdx].pts = aligned.map((p, k) => smoothPoint(prev[k], p));
      tracked[tIdx].lastSeenAt = now;
    }

    // 3) Neue Vierecke (ohne Match) als frische Tracks anhängen.
    for (let i = 0; i < newQuads.length; i++) {
      if (newToTrack[i] >= 0) continue;
      tracked.push({
        pts: newQuads[i].map(p => p.slice()),
        lastSeenAt: now,
      });
    }

    // 4) Tracks, die zu lange nicht mehr gesehen wurden, entfernen.
    //    Frische/aktuell sichtbare Tracks bleiben unangetastet, auch
    //    wenn dieser Frame sie nicht enthält.
    tracked = tracked.filter(t => (now - t.lastSeenAt) < HOLD_MS);

    // 5) Flach ausgeben (Reihenfolge = Reihenfolge in `tracked`).
    const flat = [];
    for (const t of tracked) for (const p of t.pts) flat.push(p);
    return flat;
  }

  let ws = null;
  let wsRetry = 0;
  function connectWS() {
    const proto = (location.protocol === 'https:') ? 'wss:' : 'ws:';
    const url = proto + '//' + location.host + '/ws/points';
    try {
      ws = new WebSocket(url);
    } catch (e) {
      scheduleReconnect();
      return;
    }
    ws.onopen = () => { wsRetry = 0; };
    ws.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data);
        const incoming = Array.isArray(data.points) ? data.points : [];
        targetPoints = stabilizePoints(incoming);
        // Wenn die Anzahl wächst (neues Viereck): die zusätzlichen
        // Slots direkt am Ziel einrasten, damit sie nicht aus dem
        // (0,0)-Default heran-animieren. Bestehende Slots laufen
        // weiter über die Animationsschleife.
        if (points.length < targetPoints.length) {
          for (let i = points.length; i < targetPoints.length; i++) {
            points.push(targetPoints[i].slice());
          }
          draw();
        } else if (points.length > targetPoints.length) {
          // Anzahl schrumpft: hinten überzählige verwerfen.
          points.length = targetPoints.length;
          draw();
        }
      } catch (e) { /* ignore */ }
    };
    ws.onclose = () => { ws = null; scheduleReconnect(); };
    ws.onerror = () => { try { ws.close(); } catch (e) {} };
  }
  function scheduleReconnect() {
    wsRetry = Math.min(wsRetry + 1, 6);
    const delay = Math.min(5000, 250 * Math.pow(2, wsRetry));
    setTimeout(connectWS, delay);
  }

  window.addEventListener('resize', resize);
  resize();
  connectWS();

  // Render-/Animationsschleife: läuft konstant mit der Bildwiederholrate
  // des Browsers und schiebt die aktuell gezeichneten Punkte stetig in
  // Richtung der zuletzt empfangenen Zielpunkte. So bleibt die Projektion
  // auch zwischen den (langsameren) WebSocket-Updates flüssig sichtbar.
  function animate() {
    if (targetPoints.length === points.length && points.length > 0) {
      let changed = false;
      for (let i = 0; i < points.length; i++) {
        const p = points[i], t = targetPoints[i];
        const dx = t[0] - p[0], dy = t[1] - p[1];
        const d = Math.hypot(dx, dy);
        if (d <= ANIM_MIN_STEP) {
          if (p[0] !== t[0] || p[1] !== t[1]) {
            p[0] = t[0]; p[1] = t[1]; changed = true;
          }
          continue;
        }
        if (d > ANIM_SNAP_DIST) {
          p[0] = t[0]; p[1] = t[1];
        } else {
          p[0] += dx * ANIM_ALPHA;
          p[1] += dy * ANIM_ALPHA;
        }
        changed = true;
      }
      if (changed) draw();
    }
    requestAnimationFrame(animate);
  }
  requestAnimationFrame(animate);

  // Im Kiosk-Modus liegt der Tastatur-Fokus manchmal nicht auf dem Canvas,
  // wodurch C/R/Pfeiltasten ignoriert würden. Window/Canvas explizit fokussieren.
  canvas.tabIndex = 0;
  window.focus();
  canvas.focus();
  window.addEventListener('click', () => { window.focus(); canvas.focus(); });
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def hello() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML)


@app.get("/points")
def get_points() -> dict:
    return {"points": _current_points}


@app.post("/points")
async def set_points(payload: PointsPayload) -> dict:
    cleaned: list[tuple[float, float]] = []
    for p in payload.points:
        if len(p) != 2:
            raise HTTPException(status_code=400, detail="each point needs [x, y]")
        x, y = float(p[0]), float(p[1])
        cleaned.append((max(0.0, min(1.0, x)), max(0.0, min(1.0, y))))
    global _current_points
    _current_points = cleaned
    await _broadcast_points(cleaned)
    return {"count": len(cleaned)}


@app.websocket("/ws/points")
async def ws_points(ws: WebSocket) -> None:
    await ws.accept()
    with _ws_lock:
        _ws_clients.add(ws)
    try:
        # Aktuellen Stand sofort schicken.
        await ws.send_json({"points": [list(p) for p in _current_points]})
        while True:
            # Wir erwarten keine Nachrichten vom Client; receive blockiert,
            # bis die Verbindung geschlossen wird.
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        with _ws_lock:
            _ws_clients.discard(ws)


@app.post("/upload")
def upload(payload: ImagePayload):
    try:
        raw = base64.b64decode(payload.image, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid base64: {exc}") from exc

    if len(raw) < 4 or raw[:3] != b"\xff\xd8\xff":
        raise HTTPException(status_code=400, detail="payload is not a JPEG")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
    filename = f"{ts}.jpg"
    (UPLOAD_DIR / filename).write_bytes(raw)
    (UPLOAD_DIR / "latest.jpg").write_bytes(raw)

    return {"saved": filename, "bytes": len(raw)}


@app.get("/latest")
def latest():
    path = UPLOAD_DIR / "latest.jpg"
    if not path.exists():
        raise HTTPException(status_code=404, detail="no image yet")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/list")
def list_images():
    files = sorted(p.name for p in UPLOAD_DIR.glob("*.jpg") if p.name != "latest.jpg")
    return {"count": len(files), "files": files}


# --- Browser im Kiosk-Modus automatisch öffnen ---------------------------
# Mehrfache Aufrufe werden durch einen Marker (Env-Var) verhindert; das ist
# wichtig bei `uvicorn --reload`, wo das Modul mehrfach importiert wird.
def _launch_kiosk_once(url: str) -> None:
    if os.environ.get("OSTALB_KIOSK_LAUNCHED") == "1":
        return
    os.environ["OSTALB_KIOSK_LAUNCHED"] = "1"

    def _runner() -> None:
        try:
            from projector import _open_browser, _wait_for_server
        except Exception as exc:  # pragma: no cover
            print(f"[server] projector import failed: {exc}")
            return
        if _wait_for_server(url, timeout=15.0):
            _open_browser(url)
        else:
            print(f"[server] Kiosk: Server unter {url} nicht erreichbar.")

    threading.Thread(target=_runner, name="kiosk-launcher", daemon=True).start()


def _launch_webcam_once(camera_cfg: dict, server_url: str) -> None:
    if os.environ.get("OSTALB_CAMERA_LAUNCHED") == "1":
        return
    os.environ["OSTALB_CAMERA_LAUNCHED"] = "1"

    def _runner() -> None:
        try:
            from vision import run_webcam
        except Exception as exc:  # pragma: no cover
            print(f"[server] vision import failed: {exc}")
            return
        try:
            run_webcam(
                camera=int(camera_cfg.get("index", 0)),
                server_url=server_url,
                fps=float(camera_cfg.get("fps", 5.0)),
                show=bool(camera_cfg.get("show_preview", False)),
                width=camera_cfg.get("width"),
                height=camera_cfg.get("height"),
                rotate=int(camera_cfg.get("rotate", 0)),
                contour_channel=str(camera_cfg.get("contour_channel", "blue")),
            )
        except Exception as exc:
            print(f"[server] Webcam-Pipeline beendet: {exc}")

    threading.Thread(target=_runner, name="webcam-pipeline", daemon=True).start()


@app.on_event("startup")
def _on_startup() -> None:
    from config import load_config
    cfg = load_config()
    url = cfg["server"]["url"]
    print(f"[server] config: kamera={cfg['camera']}, server={cfg['server']}")
    if cfg["server"].get("kiosk", True):
        _launch_kiosk_once(url)
    if cfg["camera"].get("enabled", True):
        _launch_webcam_once(cfg["camera"], url)
