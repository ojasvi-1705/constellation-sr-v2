"""CONSTELLATION SR - API: upload -> preprocess -> super-resolve -> PNG (logged to a database)"""
import io
import ipaddress
import os
import socket
import time
import urllib.request
from urllib.parse import urlparse
from pathlib import Path
from typing import List

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from database.models import InferenceRequest, Metric, SessionLocal
from satsrs_core.models.edsr import create_edsr
from satsrs_core.models.rcan import create_rcan
from satsrs_core.preprocessing.atmospheric import create_corrector
from satsrs_core.preprocessing.radiometric import create_calibrator

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "weights"
OUT_DIR = Path(os.getenv("OUTPUT_DIR", ROOT / "outputs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)
MAX_SIDE = int(os.getenv("MAX_SIDE", 512))  # longest input side (larger images are shrunk)
MAX_BYTES = 10 * 2**20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="CONSTELLATION SR", version="1.0.0")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
FACTORIES = {"edsr": create_edsr, "rcan": create_rcan}
_models = {}


def get_model(name):
    if name not in _models:
        m = FACTORIES[name]().eval().to(DEVICE)
        w = WEIGHTS / f"{name}.pth"
        if w.exists():
            m.load_state_dict(torch.load(w, map_location=DEVICE))
        _models[name] = m
    return _models[name]


def active_models():
    """Models that take part in the average: those with trained weights (all of them if none are trained)."""
    trained = [n for n in FACTORIES if (WEIGHTS / f"{n}.pth").exists()]
    return trained or list(FACTORIES)


def quality(a, b):
    """PSNR, global SSIM and SAM between arrays in 0..1 (H,W,3)."""
    psnr = 10 * np.log10(1 / (float(np.mean((a - b) ** 2)) or 1e-10))
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    cov = ((a - a.mean()) * (b - b.mean())).mean()
    ssim = (2 * a.mean() * b.mean() + c1) * (2 * cov + c2) / ((a.mean() ** 2 + b.mean() ** 2 + c1) * (a.var() + b.var() + c2))
    cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-8)
    return float(psnr), float(ssim), float(np.degrees(np.arccos(np.clip(cos, -1, 1))).mean())


def sr_tiled(model, x, tile=128, pad=32):
    """Run the model tile by tile (with overlap) so large images fit in memory; output is 2x."""
    _, _, H, W = x.shape
    out = torch.zeros(1, 3, 2 * H, 2 * W)
    for y in range(0, H, tile):
        for c in range(0, W, tile):
            y0, x0, y1, x1 = max(y - pad, 0), max(c - pad, 0), min(y + tile + pad, H), min(c + tile + pad, W)
            t = model(x[:, :, y0:y1, x0:x1].to(DEVICE)).cpu()
            h, w = 2 * (min(y + tile, H) - y), 2 * (min(c + tile, W) - c)
            out[:, :, 2 * y:2 * y + h, 2 * c:2 * c + w] = t[:, :, 2 * (y - y0):2 * (y - y0) + h, 2 * (c - x0):2 * (c - x0) + w]
    return out


def fetch_url(url):
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        raise HTTPException(400, "Enter a valid http(s) image URL.")
    try:
        for info in socket.getaddrinfo(p.hostname, None):
            ip = ipaddress.ip_address(info[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                raise HTTPException(400, "That address is not allowed.")
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "constellation-sr"}), timeout=10) as r:
            data = r.read(MAX_BYTES + 1)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(400, "Could not download that URL.")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image is larger than 10 MB.")
    return data


def process(data):
    img = Image.open(io.BytesIO(data)).convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    start = time.time()
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = create_corrector().flaash_correction(create_calibrator().toa_to_boa(arr, 0), 0)
    x = torch.from_numpy(arr.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    names = active_models()
    with torch.inference_mode():
        y = sum(sr_tiled(get_model(n), x) for n in names) / len(names)  # ensemble: average of the models
    out = y.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
    secs = time.time() - start
    ref = np.asarray(img.resize((img.width * 2, img.height * 2), Image.BICUBIC), dtype=np.float32) / 255.0
    return out, img.size, secs, quality(out, ref)


def run(data, filename, db):
    rec = InferenceRequest(model="+".join(active_models()), filename=filename)
    db.add(rec)
    db.commit()
    try:
        out, size, secs, (p, s, m) = process(data)
        img = Image.fromarray((out * 255).astype(np.uint8))
        path = OUT_DIR / f"{rec.id}.png"
        img.save(path)
        rec.status, rec.output_path = "completed", str(path)
        rec.input_size, rec.output_size = f"{size[0]}x{size[1]}", f"{img.width}x{img.height}"
        rec.processing_time = round(secs, 2)
        db.add(Metric(request_id=rec.id, psnr=p, ssim=s, sam=m))
    except UnidentifiedImageError:
        rec.status, rec.error = "failed", "Could not read this file as an image. Use a PNG or JPG."
    except Exception as e:  # noqa: BLE001
        rec.status, rec.error = "failed", str(e)
    db.commit()
    return rec


@app.get("/")
async def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "constellation-sr"}


@app.get("/info")
async def info():
    return {
        "project": "CONSTELLATION SR",
        "models": list(FACTORIES),
        "ensemble": active_models(),
        "scale": 2,
        "max_input_side": MAX_SIDE,
        "device": DEVICE,
        "trained_weights": {n: (WEIGHTS / f"{n}.pth").exists() for n in FACTORIES},
    }


@app.post("/api/v1/infer")
async def infer(file: UploadFile = File(...)):
    return respond(await file.read(), file.filename)


@app.post("/api/v1/infer-url")
async def infer_url(url: str = Form(...)):
    return respond(fetch_url(url), url[:200])


def respond(data, name):
    with SessionLocal() as db:
        rec = run(data, name, db)
    if rec.status == "failed":
        raise HTTPException(400, rec.error)
    return Response(
        Path(rec.output_path).read_bytes(),
        media_type="image/png",
        headers={
            "X-Request-Id": rec.id,
            "X-Input-Size": rec.input_size,
            "X-Output-Size": rec.output_size,
            "X-Processing-Time": str(rec.processing_time),
            "Access-Control-Expose-Headers": "X-Request-Id, X-Input-Size, X-Output-Size, X-Processing-Time",
        },
    )


@app.post("/api/v1/batch")
async def batch(files: List[UploadFile] = File(...)):
    results = []
    with SessionLocal() as db:
        for f in files:
            rec = run(await f.read(), f.filename, db)
            results.append({"request_id": rec.id, "filename": rec.filename, "status": rec.status, "error": rec.error})
    return {"count": len(results), "results": results}


@app.get("/api/v1/status/{request_id}")
async def status(request_id: str):
    with SessionLocal() as db:
        rec = db.get(InferenceRequest, request_id)
        if not rec:
            raise HTTPException(404, "Unknown request id.")
        m = db.query(Metric).filter_by(request_id=rec.id).first()
        return {
            "request_id": rec.id, "status": rec.status, "model": rec.model, "error": rec.error,
            "input_size": rec.input_size, "output_size": rec.output_size, "processing_time": rec.processing_time,
            "metrics_vs_bicubic": {"psnr": m.psnr, "ssim": m.ssim, "sam": m.sam} if m else None,
        }


@app.get("/api/v1/download/{request_id}")
async def download(request_id: str):
    with SessionLocal() as db:
        rec = db.get(InferenceRequest, request_id)
    if not rec or rec.status != "completed":
        raise HTTPException(404, "No completed result for this id.")
    return FileResponse(rec.output_path, media_type="image/png", filename=f"{rec.id}.png")
