"""Konfiguration für Server, Webcam und Projektor.

Liest ``config.json`` neben dieser Datei. Werte können per Umgebungsvariablen
überschrieben werden, z. B. ``OSTALB_CAMERA_INDEX=1``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

CONFIG_PATH = Path(__file__).parent / "config.json"

DEFAULTS: dict[str, Any] = {
    "camera": {
        "index": 0,
        "width": 1280,
        "height": 720,
        "fps": 5,
        "enabled": True,
        "show_preview": False,
        "contour_channel": "blue",
    },
    "server": {
        "url": "http://127.0.0.1:8000/",
        "kiosk": True,
    },
    "audio": {
        # ``None`` = System-Default. Sonst entweder ein int (PyAudio-Device-Index)
        # oder ein string (Teil-Match auf den Device-Namen, case-insensitive).
        "input_device": None,
        "output_device": None,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _env_overrides() -> dict[str, Any]:
    env = os.environ
    out: dict[str, Any] = {"camera": {}, "server": {}, "audio": {}}
    if "OSTALB_CAMERA_INDEX" in env:
        out["camera"]["index"] = int(env["OSTALB_CAMERA_INDEX"])
    if "OSTALB_CAMERA_WIDTH" in env:
        out["camera"]["width"] = int(env["OSTALB_CAMERA_WIDTH"])
    if "OSTALB_CAMERA_HEIGHT" in env:
        out["camera"]["height"] = int(env["OSTALB_CAMERA_HEIGHT"])
    if "OSTALB_CAMERA_FPS" in env:
        out["camera"]["fps"] = float(env["OSTALB_CAMERA_FPS"])
    if "OSTALB_CAMERA_ENABLED" in env:
        out["camera"]["enabled"] = env["OSTALB_CAMERA_ENABLED"].lower() in {"1", "true", "yes", "on"}
    if "OSTALB_NO_CAMERA" in env and env["OSTALB_NO_CAMERA"] == "1":
        out["camera"]["enabled"] = False
    if "OSTALB_CONTOUR_CHANNEL" in env:
        out["camera"]["contour_channel"] = env["OSTALB_CONTOUR_CHANNEL"]
    if "OSTALB_KIOSK_URL" in env:
        out["server"]["url"] = env["OSTALB_KIOSK_URL"]
    if "OSTALB_NO_KIOSK" in env and env["OSTALB_NO_KIOSK"] == "1":
        out["server"]["kiosk"] = False
    if "OSTALB_AUDIO_INPUT" in env:
        v = env["OSTALB_AUDIO_INPUT"]
        out["audio"]["input_device"] = int(v) if v.lstrip("-").isdigit() else v
    if "OSTALB_AUDIO_OUTPUT" in env:
        v = env["OSTALB_AUDIO_OUTPUT"]
        out["audio"]["output_device"] = int(v) if v.lstrip("-").isdigit() else v
    return out


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    p = Path(path) if path else CONFIG_PATH
    cfg = DEFAULTS
    if p.exists():
        try:
            cfg = _deep_merge(cfg, json.loads(p.read_text(encoding="utf-8")))
        except Exception as exc:  # pragma: no cover
            print(f"[config] {p} konnte nicht gelesen werden: {exc}")
    cfg = _deep_merge(cfg, _env_overrides())
    return cfg
