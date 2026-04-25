import base64
import cv2
import numpy as np

PANEL_H = 540          # unified panel height; width scales with aspect ratio
REGION_ASPECT = 75 / 50  # width / height of the known region between markers
REGION_THUMB_H = 320
MAX_DIM = 1280         # longest edge cap before processing
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
    aruco_panel = frame.copy()
    gray_panel = to_bgr(gray)
    thresh_panel = to_bgr(thresh_img)
    gray_title = "Grayscale"
    thresh_title = "Threshold"
    contour_title = "Contours"

    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(original_panel, corners, ids)
        cv2.aruco.drawDetectedMarkers(aruco_panel, corners, ids)
        draw_tag_connections(original_panel, corners)

        # Only draw the tag-area outline when grayscale/contours are still full-frame.
        if len(ids) == 4 and region is None:
            gray_panel = gray_panel.copy()
            draw_tag_connections(gray_panel, corners)

    if region is not None:
        gray_title = "Grayscale (between tags)"
        thresh_title = "Threshold (between tags)"
        contour_title = "Contours (between tags)"

    panels = [label(resize_panel(original_panel), "Original (ArUco + lines)")]

    if region is not None:
        panels.append(label(resize_panel(region), "Between tags"))

    panels.append(label(resize_panel(gray_panel), gray_title))
    panels.append(label(resize_panel(thresh_panel), thresh_title))
    panels.append(label(resize_panel(to_bgr(contour_img)), contour_title))

    return np.hstack(panels)


def process_image(image_b64: str) -> list[list[list[float]]]:
    data = base64.b64decode(image_b64)
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode base64 image")

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

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    raw_contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    ref = 2 * (gray.shape[1] + gray.shape[0])
    min_px = CONTOUR_MIN_LEN * ref
    max_px = CONTOUR_MAX_LEN * ref
    filtered_contours = [
        c for c in raw_contours
        if min_px <= cv2.arcLength(c, True) <= max_px
    ]
    h, w = gray.shape[:2]
    result = []
    for c in filtered_contours:
        box = cv2.boxPoints(cv2.minAreaRect(c))
        normalized = [[round(x / w, 4), round(y / h, 4)] for x, y in box]
        result.append(normalized)

    return result


def main() -> None:
    with open("test.jpg", "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    contours = process_image(b64)
    print(contours)


if __name__ == "__main__":
    main()
