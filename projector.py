"""Projector: öffnet den bestehenden Server im Browser-Kiosk-Modus."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Iterable, Sequence

DEFAULT_URL = "http://127.0.0.1:8000/"


Point = Sequence[float]  # (x, y) mit x, y in [0, 1]


def project_points(points: Iterable[Point], base_url: str = DEFAULT_URL) -> None:
    """Projiziere die gegebenen normalisierten Punkte (0..1) auf die Website.

    Erwartet eine Liste mit beliebig vielen (typisch 4) Punkten der Form
    ``(x, y)`` mit Werten zwischen 0 und 1. Sendet sie an ``POST /points``
    des bestehenden Servers; die geöffnete Seite zeichnet sie auf das Canvas.
    """
    pts: list[list[float]] = []
    for p in points:
        if len(p) != 2:
            raise ValueError("each point needs exactly 2 components (x, y)")
        x, y = float(p[0]), float(p[1])
        pts.append([max(0.0, min(1.0, x)), max(0.0, min(1.0, y))])

    url = base_url.rstrip("/") + "/points"
    body = json.dumps({"points": pts}).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=2.0) as resp:
        resp.read()


def _wait_for_server(url: str, timeout: float = 10.0) -> bool:
    import urllib.error
    import urllib.request

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=0.5):
                return True
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            time.sleep(0.2)
    return False


def _find_browser() -> tuple[str, list[str]] | None:
    """Suche einen installierten Browser und liefere (pfad, kiosk-args)."""
    candidates: list[tuple[list[str], list[str]]] = [
        # Google Chrome
        (
            [
                r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
                "google-chrome",
                "chrome",
            ],
            ["--kiosk", "--no-first-run", "--noerrdialogs", "--disable-translate"],
        ),
        # Microsoft Edge
        (
            [
                r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
                "msedge",
            ],
            ["--kiosk", "--edge-kiosk-type=fullscreen", "--no-first-run"],
        ),
        # Firefox (kein echter Kiosk, aber Vollbild via --kiosk seit FF 71)
        (
            [
                r"C:\Program Files\Mozilla Firefox\firefox.exe",
                r"C:\Program Files (x86)\Mozilla Firefox\firefox.exe",
                "firefox",
            ],
            ["--kiosk"],
        ),
    ]

    for paths, args in candidates:
        for p in paths:
            resolved = p if Path(p).is_file() else shutil.which(p)
            if resolved:
                return resolved, args
    return None


def _open_browser(url: str) -> subprocess.Popen | None:
    found = _find_browser()
    if not found:
        print("[projector] Kein Browser gefunden – bitte manuell öffnen:", url, file=sys.stderr)
        return None
    exe, args = found

    # Eigenes Profil-Verzeichnis erzwingt eine separate Browser-Instanz.
    # Sonst leitet ein bereits laufender Chrome/Edge die URL nur weiter und
    # der gestartete Prozess beendet sich sofort.
    extra: list[str] = []
    name = Path(exe).name.lower()
    if name in {"chrome.exe", "msedge.exe", "google-chrome", "chrome", "msedge"}:
        profile_dir = Path(tempfile.gettempdir()) / "ostalb-projector-profile"
        profile_dir.mkdir(exist_ok=True)
        extra = [f"--user-data-dir={profile_dir}"]

    print(f"[projector] starte Browser im Kiosk-Modus: {exe}")
    return subprocess.Popen([exe, *args, *extra, url])


def main() -> None:
    parser = argparse.ArgumentParser(description="Öffnet den bestehenden Server im Kiosk-Browser.")
    parser.add_argument(
        "url", nargs="?", default=DEFAULT_URL,
        help=f"URL des bestehenden Servers (Default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--no-wait", action="store_true",
        help="Nicht auf Erreichbarkeit des Servers warten.",
    )
    parser.add_argument(
        "--calibrate", action="store_true",
        help="\u00d6ffnet den Browser direkt im Kalibriermodus (?calibrate=1).",
    )
    parser.add_argument(
        "--demo-points", action="store_true",
        help="Sendet vier Beispielpunkte an den Server (kein Browser).",
    )
    parser.add_argument(
        "--corner-points", action="store_true",
        help="Sendet die vier Eckpunkte (0,0),(1,0),(1,1),(0,1) zur Kalibrierprüfung.",
    )
    parser.add_argument(
        "--clear", action="store_true",
        help="Leert die aktuelle Punktliste auf dem Server.",
    )
    args = parser.parse_args()

    if args.clear:
        print(f"[projector] leere Punktliste auf {args.url}")
        project_points([], base_url=args.url)
        return

    if args.corner_points:
        corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        print(f"[projector] sende Eck-Punkte an {args.url}: {corners}")
        project_points(corners, base_url=args.url)
        return

    if args.demo_points:
        demo = [
            (0.10, 0.15),
            (0.85, 0.18),
            (0.88, 0.82),
            (0.12, 0.78),
        ]
        print(f"[projector] sende Demo-Punkte an {args.url}: {demo}")
        project_points(demo, base_url=args.url)
        return

    if not args.no_wait:
        print(f"[projector] warte auf Server unter {args.url} ...")
        if not _wait_for_server(args.url):
            print(
                f"[projector] Server unter {args.url} nicht erreichbar – starte Browser trotzdem.",
                file=sys.stderr,
            )

    target_url = args.url
    if args.calibrate:
        sep = '&' if '?' in target_url else '?'
        target_url = f"{target_url}{sep}calibrate=1"
        print(f"[projector] Kalibriermodus aktiv: {target_url}")

    browser = _open_browser(target_url)
    if browser is None:
        return

    print("[projector] Browser gestartet (Strg+C zum Beenden)")
    try:
        browser.wait()
    except KeyboardInterrupt:
        pass
    finally:
        if browser.poll() is None:
            browser.terminate()


if __name__ == "__main__":
    main()
