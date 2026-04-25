import base64
import binascii
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Ostalb-Hack Image Receiver")


class ImagePayload(BaseModel):
    image: str  # base64-encoded JPEG


@app.get("/")
def hello():
    return {"message": "Hello, World!"}


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
