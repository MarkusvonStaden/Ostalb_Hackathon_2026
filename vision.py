import base64
import cv2
import numpy as np

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


def to_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


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


def build_display(frame, gray, thresh_img, contour_img, corners, ids, region: np.ndarray | None) -> np.ndarray:
    original_panel = frame.copy()
    gray_panel = to_bgr(gray)
    thresh_title = "Threshold (between tags)" if region is not None else "Threshold"
    contour_title = "Contours (between tags)" if region is not None else "Contours"

    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(original_panel, corners, ids)
        draw_tag_connections(original_panel, corners)

        if len(ids) == 4 and region is None:
            draw_tag_connections(gray_panel, corners)

    tl = label(resize_panel(original_panel), "Original (ArUco + lines)")
    tr = label(resize_panel(gray_panel), "Grayscale (between tags)" if region is not None else "Grayscale")
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
):
    """Erkenne Bretter-Eckpunkte in einem BGR-Frame."""
    h, w = frame.shape[:2]
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    aruco = cv2.aruco
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    detector = aruco.ArucoDetector(aruco_dict, aruco.DetectorParameters())

    gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray_full)

    region = None
    if ids is not None and len(ids) == 4:
        region = extract_between_tags(frame, corners)

    if region is not None:
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    else:
        gray = gray_full

    blurred = cv2.GaussianBlur(gray, (1, 1), 0)
    # Invertiert: Bretter (dunkel) -> weiß; nötig für RETR_EXTERNAL,
    # damit findContours die Bretter und nicht den Hintergrund verfolgt.
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
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

    if not return_stages:
        return result

    # Konturen-Visualisierung (auf der Region bzw. dem Vollbild zeichnen).
    base = region if region is not None else frame
    contour_img = base.copy()
    cv2.drawContours(contour_img, filtered_contours, -1, (0, 255, 0), 2)
    for quad in quads_px:
        cv2.polylines(contour_img, [quad], True, (0, 0, 255), 2)
        for p in quad:
            cv2.circle(contour_img, tuple(int(v) for v in p), 5, (0, 0, 255), -1)

    stages = {
        "frame": frame,
        "gray": gray,
        "thresh": thresh,
        "contour_img": contour_img,
        "corners": corners,
        "ids": ids,
        "region": region,
    }
    return result, stages


def process_image(image_b64: str) -> list[list[list[float]]]:
    data = base64.b64decode(image_b64)
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode base64 image")
    return process_frame(frame)


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


def _post_points(points: list[list[float]], server_url: str) -> None:
    import json
    import urllib.request
    body = json.dumps({"points": points}).encode("utf-8")
    req = urllib.request.Request(
        server_url.rstrip("/") + "/points",
        data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=2.0) as resp:
        resp.read()


def run_webcam(
    camera: int = 0,
    server_url: str = "http://127.0.0.1:8000/",
    fps: float = 5.0,
    show: bool = True,
    width: int | None = None,
    height: int | None = None,
    rotate: int = 0,
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
    if show:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

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
                    if show:
                        contours, stages = process_frame(frame, return_stages=True)
                        last_panel = build_display(
                            stages["frame"], stages["gray"], stages["thresh"],
                            stages["contour_img"], stages["corners"], stages["ids"],
                            stages["region"],
                        )
                    else:
                        contours = process_frame(frame)
                    flat = [[float(x), float(y)] for c in contours for x, y in c]
                    _post_points(flat, server_url)
                    if show is False:
                        print(f"[vision] {len(contours)} Vierecke -> {len(flat)} Punkte")
                except Exception as exc:
                    print(f"[vision] Fehler: {exc}", file=__import__('sys').stderr)

            if show:
                img = last_panel if last_panel is not None else frame
                ph, pw = img.shape[:2]
                scale = min(MAX_DISPLAY_W / pw, MAX_DISPLAY_H / ph, 1.0)
                if scale < 1.0:
                    img = cv2.resize(img, (int(pw * scale), int(ph * scale)), interpolation=cv2.INTER_AREA)
                cv2.imshow(win_name, img)
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
    args = parser.parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        raise SystemExit(f"Konnte Bild nicht laden: {args.image}")
    contours, stages = process_frame(frame, return_stages=True)
    print(contours)
    display = build_display(
        stages["frame"], stages["gray"], stages["thresh"],
        stages["contour_img"], stages["corners"], stages["ids"],
        stages["region"],
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
