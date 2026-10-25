import io
import os
import tempfile

os.environ["DATABASE_URL"] = f"sqlite:///{tempfile.mkdtemp()}/t.db"
os.environ["OUTPUT_DIR"] = tempfile.mkdtemp()

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from api.app import app

client = TestClient(app)


def png(w=32, h=24):
    buf = io.BytesIO()
    Image.fromarray((np.random.rand(h, w, 3) * 255).astype("uint8")).save(buf, "PNG")
    return buf.getvalue()


def infer(model="edsr"):
    return client.post("/api/v1/infer", files={"file": ("a.png", png(), "image/png")}, data={"model": model})


def test_pages():
    assert client.get("/").status_code == 200
    assert client.get("/health").json()["status"] == "healthy"
    assert client.get("/info").json()["scale"] == 2


def test_infer_doubles_size():
    r = infer()
    assert r.status_code == 200 and r.headers["x-output-size"] == "64x48"
    assert Image.open(io.BytesIO(r.content)).size == (64, 48)


def test_status_and_download():
    rid = infer().headers["x-request-id"]
    s = client.get(f"/api/v1/status/{rid}").json()
    assert s["status"] == "completed" and s["metrics_vs_bicubic"]["psnr"] > 0
    assert client.get(f"/api/v1/download/{rid}").headers["content-type"] == "image/png"
    assert client.get("/api/v1/status/nope").status_code == 404


def test_bad_input():
    r = client.post("/api/v1/infer", files={"file": ("x.txt", b"nope", "text/plain")}, data={"model": "edsr"})
    assert r.status_code == 400
    assert client.post("/api/v1/infer", files={"file": ("a.png", png(), "image/png")}, data={"model": "zzz"}).status_code == 400


def test_batch():
    files = [("files", ("a.png", png(), "image/png")), ("files", ("b.png", png(), "image/png"))]
    r = client.post("/api/v1/batch", files=files, data={"model": "edsr"}).json()
    assert r["count"] == 2 and all(x["status"] == "completed" for x in r["results"])


def test_url_blocks_local_and_bad_urls():
    for u in ("http://127.0.0.1:8000/health", "ftp://x/y.png", "not a url"):
        assert client.post("/api/v1/infer-url", data={"url": u, "model": "edsr"}).status_code == 400


def test_url_endpoint_rejects_bad_and_private_urls():
    for u in ("not a url", "ftp://example.com/a.png", "http://127.0.0.1/a.png", "http://localhost:8000/health"):
        assert client.post("/api/v1/infer-url", data={"url": u, "model": "edsr"}).status_code == 400


def test_info_reports_device():
    assert client.get("/info").json()["device"] in ("cpu", "cuda")


def test_tiled_matches_whole_image():
    import torch
    from api.app import get_model, sr_tiled
    x = torch.rand(1, 3, 150, 200)
    with torch.inference_mode():
        assert (get_model("edsr")(x) - sr_tiled(get_model("edsr"), x)).abs().max() < 1e-3
