import base64
import binascii
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
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
      width: 1001vw;
      height: 100vh;
    }
    #help {
      position: fixed;
      top: 8px;1
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
    ctx.fillStyle = '#ff2a2a';
    const r = Math.max(6, Math.min(cssW, cssH) * 0.015);
    for (const p of points) {
      const [x, y] = applyH(p[0], p[1]);
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
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

  async function poll() {
    try {
      const res = await fetch('/points', { cache: 'no-store' });
      if (res.ok) {
        const data = await res.json();
        points = Array.isArray(data.points) ? data.points : [];
        draw();
      }
    } catch (e) { /* ignore */ }
  }

  window.addEventListener('resize', resize);
  resize();
  poll();
  setInterval(poll, 3000);

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
def set_points(payload: PointsPayload) -> dict:
    cleaned: list[tuple[float, float]] = []
    for p in payload.points:
        if len(p) != 2:
            raise HTTPException(status_code=400, detail="each point needs [x, y]")
        x, y = float(p[0]), float(p[1])
        cleaned.append((max(0.0, min(1.0, x)), max(0.0, min(1.0, y))))
    global _current_points
    _current_points = cleaned
    return {"count": len(cleaned)}


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
