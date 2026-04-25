import base64
import time
import cv2
import numpy as np
from pathlib import Path

_hand_detector = None
_hand_detector_video = False


def _get_hand_detector():
    global _hand_detector, _hand_detector_video
    if _hand_detector is not None:
        return _hand_detector
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as _mp_python
        from mediapipe.tasks.python import vision as _mp_vision
        model = Path(__file__).parent / 'hand_landmarker.task'
        if not model.exists():
            return None
        opts = _mp_vision.HandLandmarkerOptions(
            base_options=_mp_python.BaseOptions(model_asset_path=str(model)),
            num_hands=2,
            running_mode=_mp_vision.RunningMode.VIDEO,
        )
        _hand_detector = _mp_vision.HandLandmarker.create_from_options(opts)
        _hand_detector_video = True
        return _hand_detector
    except Exception as exc:
        print(f"[vision] HandLandmarker nicht verfügbar: {exc}", file=__import__('sys').stderr)
        return None


def _compute_hovers(
    region_bgr: np.ndarray,
    contours_norm: list,
) -> list[bool]:
    """Gibt pro Brett an, ob gerade eine Hand darueber erkannt wird.

    Arbeitet direkt auf dem bereits gewarpten Region-Bild (deutlich kleiner
    als das Vollbild) und nutzt MediaPipe im VIDEO-Modus mit internem
    Tracking.
    """
    hovers = [False] * len(contours_norm)
    detector = _get_hand_detector()
    if detector is None or not contours_norm or region_bgr is None or region_bgr.size == 0:
        return hovers

    import mediapipe as mp

    rh, rw = region_bgr.shape[:2]
    rgb = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    if _hand_detector_video:
        ts_ms = int(time.monotonic() * 1000)
        result = detector.detect_for_video(mp_image, ts_ms)
    else:
        result = detector.detect(mp_image)
    if not result.hand_landmarks:
        return hovers

    # Brett-Quads in Region-Pixelkoordinaten
    nx = max(1, rw - 1)
    ny = max(1, rh - 1)
    quads_px = [
        np.array([[int(x * nx), int(y * ny)] for x, y in quad], dtype=np.int32)
        for quad in contours_norm
    ]

    # Palmenmittelpunkt jeder erkannten Hand pruefen
    # Landmarks 0 (Handwurzel) + 5,9,13,17 (MCP-Gelenke) = Handflaechen-Zentrum
    palm_indices = (0, 5, 9, 13, 17)
    for hand_lms in result.hand_landmarks:
        xs = [hand_lms[i].x for i in palm_indices]
        ys = [hand_lms[i].y for i in palm_indices]
        rx = (sum(xs) / len(xs)) * rw
        ry = (sum(ys) / len(ys)) * rh

        for qi, quad_px in enumerate(quads_px):
            if not hovers[qi] and cv2.pointPolygonTest(quad_px, (float(rx), float(ry)), False) >= 0:
                hovers[qi] = True

    return hovers

PANEL_H = 540          # unified panel height; width scales with aspect ratio
REGION_ASPECT = 87 / 47  # width / height of the known region between markers
REGION_THUMB_H = 320
MAX_DIM = 1280         # longest edge cap before processing
MAX_DISPLAY_W = 7680   # max breite des kombinierten Vorschau-Fensters in px
MAX_DISPLAY_H = 4320   # max höhe des kombinierten Vorschau-Fensters in px
# Region is 75 cm × 50 cm → perimeter = 250 cm
# 15 cm / 250 cm = 0.06,  200 cm / 250 cm = 0.80
CONTOUR_MIN_LEN = 0.06  # min arc length as fraction of region perimeter (2 * (w + h))
CONTOUR_MAX_LEN = 0.80  # max arc length as fraction of region perimeter

# Blauer Fehler-Punkt auf dem Brett: bewusst weiter HSV-Bereich, damit kein
# Alarm verpasst wird. Hue in OpenCV: 0..179. Blau-Klebepunkte liegen meist
# bei H≈100..115; wir akzeptieren H 95..135 (deckt auch leichten Cyan-/
# Lila-Drift ab). Sättigung muss merklich vorhanden sein, sonst werden
# graue Bretter / Schatten getriggert.
DOT_HSV_LOWER = (95, 80, 50)
DOT_HSV_UPPER = (135, 255, 255)
DOT_MIN_AREA_PX = 30          # kleinere Blobs werden als Rauschen verworfen
DOT_MAX_AREA_FRAC = 0.05      # max. 5 % der Region (sonst kein „Punkt“)

# Cache der zuletzt gesehenen ArUco-Marker pro ID. Wird genutzt, wenn im
# aktuellen Frame nicht alle 4 Marker erkannt werden – dann fallen wir auf
# die letzte bekannte Position zurück, statt die Region komplett zu verlieren.
# Wird invalidiert, sobald sich die Frame-Auflösung ändert.
_LAST_MARKERS: dict[int, np.ndarray] = {}
_LAST_MARKERS_SHAPE: tuple[int, int] | None = None

# ArUco-Detector ist teuer zu konstruieren -> einmalig cachen.
_ARUCO_DETECTOR = None


def _get_aruco_detector():
    global _ARUCO_DETECTOR
    if _ARUCO_DETECTOR is None:
        aruco = cv2.aruco
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        _ARUCO_DETECTOR = aruco.ArucoDetector(aruco_dict, aruco.DetectorParameters())
    return _ARUCO_DETECTOR


def to_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


# Channels werden in BGR-Reihenfolge ausgewertet (OpenCV-Default).
_CHANNEL_INDEX = {"blue": 0, "green": 1, "red": 2}


def to_intensity(img: np.ndarray, mode: str = "blue") -> np.ndarray:
    """Liefert ein Single-Channel-Bild für die Konturen-Pipeline.

    ``mode='blue'`` blendet rote Projektionslinien weitgehend aus, weil rotes
    Licht im Blau-Kanal kaum Energie trägt. ``mode='gray'`` entspricht dem
    klassischen ``cv2.cvtColor(BGR2GRAY)`` (Regressions-Schalter).
    """
    if img.ndim == 2:
        return img
    m = (mode or "blue").lower()
    if m == "gray":
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    idx = _CHANNEL_INDEX.get(m)
    if idx is None:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img[:, :, idx]


def resize_panel(img: np.ndarray) -> np.ndarray:
    img = to_bgr(img)
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((PANEL_H, 1, 3), dtype=np.uint8)

    new_w = max(1, int(round(w * PANEL_H / h)))
    return cv2.resize(img, (new_w, PANEL_H))


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out



def detect_blue_dots(region_bgr: np.ndarray) -> tuple[list[tuple[float, float, float]], np.ndarray]:
    """Suche blaue Klebepunkte in der gewarpten Region.

    Liefert eine Liste ``[(x_norm, y_norm, area_px), ...]`` mit den
    Schwerpunkten in normalisierten Region-Koordinaten (0..1) sowie die
    binäre HSV-Maske (für Visualisierung).
    """
    if region_bgr is None or region_bgr.size == 0:
        return [], np.zeros((1, 1), dtype=np.uint8)
    h, w = region_bgr.shape[:2]
    hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(DOT_HSV_LOWER, dtype=np.uint8),
                       np.array(DOT_HSV_UPPER, dtype=np.uint8))
    # Rauschen wegputzen, dann Lücken schließen.
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel5)

    max_area = DOT_MAX_AREA_FRAC * float(w * h)
    nx = max(1, w - 1)
    ny = max(1, h - 1)
    n_labels, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    dots: list[tuple[float, float, float]] = []
    for i in range(1, n_labels):  # 0 = Hintergrund
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < DOT_MIN_AREA_PX or area > max_area:
            continue
        cx, cy = centroids[i]
        dots.append((round(float(cx) / nx, 4), round(float(cy) / ny, 4), area))
    return dots, mask


def order_points(pts: np.ndarray) -> np.ndarray:
    """Return points in [TL, TR, BR, BL] order."""
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # TL: smallest x+y
    rect[2] = pts[np.argmax(s)]   # BR: largest x+y
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]   # TR: smallest x-y
    rect[3] = pts[np.argmax(d)]   # BL: largest x-y
    return rect


def extract_between_tags(frame: np.ndarray, corners) -> np.ndarray | None:
    """Perspective-warp the region enclosed by 4 ArUco markers."""
    centers = np.array([c[0].mean(axis=0) for c in corners], dtype=np.float32)
    src = order_points(centers)

    w = int(max(
        np.linalg.norm(src[1] - src[0]),
        np.linalg.norm(src[2] - src[3]),
    ))
    h = int(round(w / REGION_ASPECT))
    if w < 4 or h < 4:
        return None

    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, M, (w, h))


def draw_tag_connections(img: np.ndarray, corners) -> None:
    if len(corners) != 4:
        return

    centers = np.array([c[0].mean(axis=0) for c in corners], dtype=np.float32)
    src = order_points(centers).astype(np.int32)
    cv2.polylines(img, [src], isClosed=True, color=(0, 255, 255), thickness=3, lineType=cv2.LINE_AA)
    for p in src:
        cv2.circle(img, tuple(p), 6, (0, 0, 255), -1, lineType=cv2.LINE_AA)


def build_display(frame, gray, thresh_img, contour_img, corners, ids, region: np.ndarray | None, channel: str = "blue") -> np.ndarray:
    original_panel = frame.copy()
    gray_panel = to_bgr(gray)
    chan_label = (channel or "blue").lower()
    chan_suffix = f" [{chan_label}]"
    thresh_title = ("Threshold (between tags)" if region is not None else "Threshold") + chan_suffix
    contour_title = ("Contours (between tags)" if region is not None else "Contours") + chan_suffix
    gray_title_base = "Grayscale (between tags)" if region is not None else "Grayscale"
    gray_title = gray_title_base + chan_suffix

    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(original_panel, corners, ids)
        draw_tag_connections(original_panel, corners)

        if len(ids) == 4 and region is None:
            draw_tag_connections(gray_panel, corners)

    tl = label(resize_panel(original_panel), "Original (ArUco + lines)")
    tr = label(resize_panel(gray_panel), gray_title)
    bl = label(resize_panel(to_bgr(thresh_img)), thresh_title)
    br = label(resize_panel(to_bgr(contour_img)), contour_title)

    # Alle vier Panels auf dieselbe Breite bringen, dann 2×2 zusammensetzen.
    w = max(tl.shape[1], tr.shape[1], bl.shape[1], br.shape[1])

    def pad_w(img):
        if img.shape[1] < w:
            img = cv2.copyMakeBorder(img, 0, 0, 0, w - img.shape[1], cv2.BORDER_CONSTANT)
        return img

    top = np.hstack([pad_w(tl), pad_w(tr)])
    bot = np.hstack([pad_w(bl), pad_w(br)])
    return np.vstack([top, bot])


def process_frame(
    frame: np.ndarray,
    return_stages: bool = False,
    contour_channel: str = "blue",
):
    """Erkenne Bretter-Eckpunkte in einem BGR-Frame.

    ``contour_channel`` steuert, welches Single-Channel-Bild für die
    Threshold-/Konturen-Pipeline verwendet wird. Default ``"blue"`` blendet
    rotes Projektor-Eigenlicht weitgehend aus. ArUco bleibt auf klassischen
    Graustufen, weil die Marker schwarz/weiß sind.
    """
    h, w = frame.shape[:2]
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    detector = _get_aruco_detector()

    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray_full)

    # Marker-Cache aktualisieren / bei Auflösungswechsel verwerfen.
    global _LAST_MARKERS, _LAST_MARKERS_SHAPE
    cur_shape = (frame.shape[0], frame.shape[1])
    if _LAST_MARKERS_SHAPE != cur_shape:
        _LAST_MARKERS.clear()
        _LAST_MARKERS_SHAPE = cur_shape
    if ids is not None and len(ids) > 0:
        for c, i in zip(corners, ids.flatten()):
            _LAST_MARKERS[int(i)] = c.copy()

    # Falls weniger als 4 Marker im aktuellen Frame: mit zwischengespeicherten
    # Positionen auffüllen, damit die Region-Extraktion stabil bleibt.
    cur_ids = set(int(i) for i in (ids.flatten() if ids is not None else []))
    if len(cur_ids) < 4 and len(_LAST_MARKERS) >= 4:
        merged_corners = list(corners) if corners is not None else []
        merged_ids = list(cur_ids)
        for mid, mcorner in _LAST_MARKERS.items():
            if mid in cur_ids:
                continue
            merged_corners.append(mcorner)
            merged_ids.append(mid)
            if len(merged_ids) == 4:
                break
        if len(merged_ids) == 4:
            corners = tuple(merged_corners)
            ids = np.array(merged_ids, dtype=np.int32).reshape(-1, 1)

    region = None
    if ids is not None and len(ids) == 4:
        region = extract_between_tags(frame, corners)

    if region is not None:
        gray = to_intensity(region, contour_channel)
    else:
        gray = to_intensity(frame, contour_channel)

    # Invertiert: Bretter (dunkel) -> weiss; noetig fuer RETR_EXTERNAL,
    # damit findContours die Bretter und nicht den Hintergrund verfolgt.
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Nur ÄUSSERE Konturen, sonst werden Innen- und Außenrand der Bretter
    # doppelt gefunden und die Eck-Erkennung springt zwischen beiden.
    raw_contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ref = 2 * (gray.shape[1] + gray.shape[0])
    min_px = CONTOUR_MIN_LEN * ref
    max_px = CONTOUR_MAX_LEN * ref
    filtered_contours = [
        c for c in raw_contours
        if min_px <= cv2.arcLength(c, True) <= max_px
    ]
    h, w = gray.shape[:2]
    nx = max(1, w - 1)
    ny = max(1, h - 1)
    result = []
    quads_px: list[np.ndarray] = []
    for c in filtered_contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        quad = approx.reshape(4, 2).astype(np.float32)
        quads_px.append(quad.astype(np.int32))
        normalized = [[round(float(x) / nx, 4), round(float(y) / ny, 4)] for x, y in quad]
        result.append(normalized)

    # Blaue Fehler-Punkte nur in der gewarpten Region suchen (außerhalb
    # gibt es keine sinnvolle Bezugsfläche). Pro Brett-Quad max. einen
    # Treffer (größte Fläche gewinnt).
    errors: list[list[float] | None] = [None] * len(result)
    dot_mask = np.zeros((h, w), dtype=np.uint8)
    dots: list[tuple[float, float, float]] = []
    if region is not None and quads_px:
        dots, dot_mask = detect_blue_dots(region)
        # Sortiere nach Fläche absteigend, damit der größte Punkt pro
        # Brett gewinnt.
        for dx, dy, area in sorted(dots, key=lambda d: d[2], reverse=True):
            px = dx * nx
            py = dy * ny
            for qi, quad_px in enumerate(quads_px):
                if errors[qi] is not None:
                    continue
                if cv2.pointPolygonTest(quad_px, (float(px), float(py)), False) >= 0:
                    errors[qi] = [round(dx, 4), round(dy, 4)]
                    break

    if not return_stages:
        return result, errors

    # Konturen-Visualisierung (auf der Region bzw. dem Vollbild zeichnen).
    base = region if region is not None else frame
    contour_img = base.copy()
    cv2.drawContours(contour_img, filtered_contours, -1, (0, 255, 0), 2)
    for quad in quads_px:
        cv2.polylines(contour_img, [quad], True, (0, 0, 255), 2)
        for p in quad:
            cv2.circle(contour_img, tuple(int(v) for v in p), 5, (0, 0, 255), -1)
    # Blaue Punkte einzeichnen (in der Region-Visualisierung).
    for dx, dy, _area in dots:
        cx = int(round(dx * nx))
        cy = int(round(dy * ny))
        cv2.circle(contour_img, (cx, cy), 10, (0, 165, 255), 2, lineType=cv2.LINE_AA)
        cv2.circle(contour_img, (cx, cy), 3, (0, 165, 255), -1, lineType=cv2.LINE_AA)

    stages = {
        "frame": frame,
        "gray": gray,
        "thresh": thresh,
        "contour_img": contour_img,
        "corners": corners,
        "ids": ids,
        "region": region,
        "contour_channel": contour_channel,
        "errors": errors,
        "dot_mask": dot_mask,
    }
    return result, errors, stages


def process_image(image_b64: str) -> list[list[list[float]]]:
    data = base64.b64decode(image_b64)
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode base64 image")
    contours, _errors = process_frame(frame)
    return contours


def _open_webcam(index: int, width: int | None = None, height: int | None = None) -> cv2.VideoCapture:
    # CAP_DSHOW vermeidet auf Windows lange Initialisierungszeiten / MSMF-Probleme.
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Konnte Webcam {index} nicht öffnen")
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


# Wiederverwendete HTTP-Verbindung pro Ziel-Host, um TCP-Handshake-
# Overhead in der Capture-Schleife zu vermeiden.
_HTTP_CONN: dict[tuple[str, str, int], object] = {}


def _post_points(points: list[list[float]], server_url: str,
                 errors: list[list[float] | None] | None = None,
                 hovers: list[bool] | None = None) -> None:
    import json
    from http.client import HTTPConnection, HTTPSConnection
    from urllib.parse import urlsplit

    payload: dict = {"points": points}
    if errors is not None:
        payload["errors"] = errors
    if hovers is not None:
        payload["hovers"] = hovers
    body = json.dumps(payload).encode("utf-8")

    parts = urlsplit(server_url)
    scheme = parts.scheme or "http"
    host = parts.hostname or "127.0.0.1"
    port = parts.port or (443 if scheme == "https" else 80)
    base_path = parts.path.rstrip("/")
    path = base_path + "/points"

    key = (scheme, host, port)
    conn = _HTTP_CONN.get(key)

    def _new_conn():
        if scheme == "https":
            return HTTPSConnection(host, port, timeout=2.0)
        return HTTPConnection(host, port, timeout=2.0)

    headers = {"Content-Type": "application/json", "Content-Length": str(len(body))}
    for attempt in range(2):
        if conn is None:
            conn = _new_conn()
            _HTTP_CONN[key] = conn
        try:
            conn.request("POST", path, body=body, headers=headers)
            resp = conn.getresponse()
            resp.read()
            return
        except Exception:
            try:
                conn.close()
            except Exception:
                pass
            _HTTP_CONN.pop(key, None)
            conn = None
            if attempt == 1:
                raise


def run_capture_loop(
    on_result,
    *,
    camera: int = 0,
    fps: float = 5.0,
    width: int | None = None,
    height: int | None = None,
    rotate: int = 0,
    contour_channel: str = "blue",
    stop_event=None,
) -> None:
    """Webcam-Loop ohne HTTP/Preview - ruft ``on_result(points, errors, hovers)``
    in-process auf. Geeignet, um direkt aus dem Server-Prozess zu laufen.

    ``stop_event`` (threading.Event) bricht die Schleife sauber ab.
    """
    cap = _open_webcam(camera, width, height)
    period = 1.0 / max(0.1, fps)
    rot_code = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }.get(int(rotate) % 360)
    print(f"[vision] Webcam {camera} geoeffnet (in-process, rotate={rotate}deg)")
    last_send = 0.0
    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.1)
                continue
            if rot_code is not None:
                frame = cv2.rotate(frame, rot_code)
            now = time.time()
            if now - last_send < period:
                time.sleep(max(0.0, period - (now - last_send)))
                continue
            last_send = now
            try:
                contours, errors, stages = process_frame(
                    frame, return_stages=True, contour_channel=contour_channel,
                )
                flat = [[float(x), float(y)] for c in contours for x, y in c]
                hovers: list[bool] = [False] * len(contours)
                if stages.get("region") is not None and stages.get("corners") is not None:
                    try:
                        hovers = _compute_hovers(stages["region"], contours)
                    except Exception as hover_exc:
                        print(f"[vision] Hand-Hover: {hover_exc}", file=__import__('sys').stderr)
                try:
                    on_result(flat, errors, hovers)
                except Exception as cb_exc:
                    print(f"[vision] Callback-Fehler: {cb_exc}", file=__import__('sys').stderr)
            except Exception as exc:
                print(f"[vision] Fehler: {exc}", file=__import__('sys').stderr)
    finally:
        cap.release()


def run_webcam(
    camera: int = 0,
    server_url: str = "http://127.0.0.1:8000/",
    fps: float = 5.0,
    show: bool = True,
    width: int | None = None,
    height: int | None = None,
    rotate: int = 0,
    contour_channel: str = "blue",
) -> None:
    """Liest kontinuierlich von der Webcam, erkennt Bretter und sendet sie an den Server.

    ``rotate`` dreht jeden Frame vor der Verarbeitung um 0/90/180/270 Grad.
    """
    cap = _open_webcam(camera, width, height)
    period = 1.0 / max(0.1, fps)
    rot_code = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }.get(int(rotate) % 360)
    print(f"[vision] Webcam {camera} geöffnet – sende an {server_url} (rotate={rotate}°, Strg+C zum Beenden)")

    win_name = "vision (stages: original | region | gray | threshold | contours)"
    raw_win_name = "camera (raw)"
    if show:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.namedWindow(raw_win_name, cv2.WINDOW_NORMAL)

    import time
    last_send = 0.0
    last_panel = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[vision] Frame konnte nicht gelesen werden", file=__import__('sys').stderr)
                time.sleep(0.1)
                continue
            if rot_code is not None:
                frame = cv2.rotate(frame, rot_code)

            now = time.time()
            if now - last_send >= period:
                last_send = now
                try:
                    contours, errors, stages = process_frame(frame, return_stages=True, contour_channel=contour_channel)  # type: ignore[misc]
                    if show:
                        last_panel = build_display(
                            stages["frame"], stages["gray"], stages["thresh"],
                            stages["contour_img"], stages["corners"], stages["ids"],
                            stages["region"], stages.get("contour_channel", contour_channel),
                        )
                    flat = [[float(x), float(y)] for c in contours for x, y in c]
                    hovers: list[bool] = [False] * len(contours)
                    if stages.get("region") is not None and stages.get("corners") is not None:
                        try:
                            hovers = _compute_hovers(stages["region"], contours)
                        except Exception as hover_exc:
                            print(f"[vision] Hand-Hover: {hover_exc}", file=__import__('sys').stderr)
                    _post_points(flat, server_url, errors=errors, hovers=hovers)
                    if show is False:
                        n_err = sum(1 for e in errors if e is not None)
                        print(f"[vision] {len(contours)} Vierecke -> {len(flat)} Punkte, {n_err} Fehler-Markierung(en)")
                except Exception as exc:
                    print(f"[vision] Fehler: {exc}", file=__import__('sys').stderr)

            if show:
                img = last_panel if last_panel is not None else frame
                ph, pw = img.shape[:2]
                scale = min(MAX_DISPLAY_W / pw, MAX_DISPLAY_H / ph, 1.0)
                if scale < 1.0:
                    img = cv2.resize(img, (int(pw * scale), int(ph * scale)), interpolation=cv2.INTER_AREA)
                cv2.imshow(win_name, img)
                cv2.imshow(raw_win_name, frame)
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break
            else:
                # Kleine Pause, damit die Schleife die CPU nicht voll auslastet.
                time.sleep(max(0.0, period - (time.time() - now)))
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if show:
            cv2.destroyAllWindows()


def main() -> None:
    import argparse
    from config import load_config
    cfg = load_config()
    cam = cfg["camera"]
    srv = cfg["server"]

    parser = argparse.ArgumentParser(description="Erkenne Bretter und sende Punkte an den Projektor-Server.")
    parser.add_argument("--image", default="test.jpeg", help="Bilddatei verarbeiten und Ergebnis anzeigen (Default: test.jpeg).")
    parser.add_argument("--camera", type=int, default=cam["index"], help=f"Webcam-Index (Default {cam['index']}).")
    parser.add_argument("--server", default=srv["url"], help="URL des Projektor-Servers.")
    parser.add_argument("--fps", type=float, default=cam["fps"], help=f"Sende-/Verarbeitungsrate (Default {cam['fps']}).")
    parser.add_argument("--no-show", action="store_true", help="Kein Vorschau-Fenster anzeigen.")
    parser.add_argument("--show", action="store_true", help="Vorschau-Fenster anzeigen (überschreibt Config).")
    parser.add_argument("--width", type=int, default=cam.get("width"), help="Webcam-Auflösung Breite.")
    parser.add_argument("--height", type=int, default=cam.get("height"), help="Webcam-Auflösung Höhe.")
    parser.add_argument("--rotate", type=int, default=int(cam.get("rotate", 0)),
                        choices=[0, 90, 180, 270],
                        help="Frame um 0/90/180/270 Grad drehen (Default aus Config).")
    parser.add_argument("--contour-channel", default=cam.get("contour_channel", "blue"),
                        choices=["blue", "green", "red", "gray"],
                        help="Single-Channel für Threshold/Konturen. 'blue' blendet rote Projektion aus (Default aus Config).")
    args = parser.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        raise SystemExit(f"Konnte Bild nicht laden: {args.image}")
    contours, errors, stages = process_frame(frame, return_stages=True, contour_channel=args.contour_channel)
    print("contours:", contours)
    print("errors:  ", errors)
    display = build_display(
        stages["frame"], stages["gray"], stages["thresh"],
        stages["contour_img"], stages["corners"], stages["ids"],
        stages["region"], stages.get("contour_channel", args.contour_channel),
    )
    ph, pw = display.shape[:2]
    scale = min(MAX_DISPLAY_W / pw, MAX_DISPLAY_H / ph, 1.0)
    if scale < 1.0:
        display = cv2.resize(display, (int(pw * scale), int(ph * scale)), interpolation=cv2.INTER_AREA)
    cv2.imshow("vision – " + args.image, display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
