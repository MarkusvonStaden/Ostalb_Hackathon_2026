import cv2
import numpy as np

PANEL_W, PANEL_H = 320, 240
TAG_THUMB = 150


def to_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def resize_panel(img: np.ndarray) -> np.ndarray:
    return cv2.resize(img, (PANEL_W, PANEL_H))


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out


def crop_tag(frame: np.ndarray, corner: np.ndarray) -> np.ndarray:
    pts = corner[0].astype(int)
    x, y, w, h = cv2.boundingRect(pts)
    pad = 12
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((TAG_THUMB, TAG_THUMB, 3), dtype=np.uint8)
    return cv2.resize(crop, (TAG_THUMB, TAG_THUMB))


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
    h = int(max(
        np.linalg.norm(src[3] - src[0]),
        np.linalg.norm(src[2] - src[1]),
    ))
    if w < 4 or h < 4:
        return None

    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, M, (w, h))


def build_display(frame, gray, edges, corners, ids) -> np.ndarray:
    # --- top row: original | grayscale | edges ---
    aruco_panel = frame.copy()
    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(aruco_panel, corners, ids)

    top = np.hstack([
        label(resize_panel(frame), "Original"),
        label(resize_panel(to_bgr(gray)), "Grayscale"),
        label(resize_panel(to_bgr(edges)), "Edges"),
        label(resize_panel(aruco_panel), "ArUco overlay"),
    ])

    # --- bottom row: individual tag crops + extracted region when 4 tags present ---
    if ids is not None and len(ids) > 0:
        tag_imgs = []
        for i, corner in enumerate(corners):
            thumb = crop_tag(frame, corner)
            thumb = label(thumb, f"ID {ids[i][0]}")
            tag_imgs.append(thumb)

        if len(ids) == 4:
            region = extract_between_tags(frame, corners)
            if region is not None:
                region_thumb = cv2.resize(region, (TAG_THUMB * 2, TAG_THUMB))
                tag_imgs.append(label(region_thumb, "Between tags"))
                cv2.imshow("Region between tags", region)

        bottom_strip = np.hstack(tag_imgs)
        total_w = top.shape[1]
        if bottom_strip.shape[1] < total_w:
            pad = np.zeros((TAG_THUMB, total_w - bottom_strip.shape[1], 3), dtype=np.uint8)
            bottom_strip = np.hstack([bottom_strip, pad])
        else:
            bottom_strip = bottom_strip[:, :total_w]

        return np.vstack([top, bottom_strip])

    return top


def main() -> None:
    capture = cv2.VideoCapture(0)

    aruco = cv2.aruco
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    detector = aruco.ArucoDetector(aruco_dict, aruco.DetectorParameters())

    try:
        while True:
            _, frame = capture.read()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)

            corners, ids, _ = detector.detectMarkers(gray)

            display = build_display(frame, gray, edges, corners, ids)
            cv2.imshow("Part detection", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        capture.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


if __name__ == "__main__":
    main()
