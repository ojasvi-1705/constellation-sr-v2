"""Train EDSR/RCAN on SEN2VENuS (Sentinel-2 10 m -> VENuS 5 m, x2).
Usage: python train.py --data data --model edsr --iters 20000
"""
import argparse, glob, random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from satsrs_core.models.edsr import create_edsr
from satsrs_core.models.rcan import create_rcan
from satsrs_core.preprocessing.atmospheric import create_corrector
from satsrs_core.preprocessing.radiometric import create_calibrator

STRETCH = 0.3  # reflectance 0..0.3 -> 0..1 so images look like normal photos
_cal, _cor = create_calibrator(), create_corrector()


def prep(x):
    """Same preprocessing as inference: radiometric -> atmospheric."""
    a = _cor.flaash_correction(_cal.toa_to_boa(x.numpy(), 0), 0)
    return torch.from_numpy(a.astype(np.float32))


def load(files, per_pair, cap):
    lr, hr = [], []
    for f in files:
        h = f.replace("_10m_", "_05m_")
        if not Path(h).exists():
            continue
        a, b = torch.load(f), torch.load(h)  # int16 [n,4,128,128] / [n,4,256,256]
        i = torch.randperm(a.shape[0])[:per_pair]
        conv = lambda t: t[i][:, [2, 1, 0]].float().div(1e4 * STRETCH).clamp(0, 1).half()  # B4,B3,B2 = RGB
        lr.append(conv(a)); hr.append(conv(b))
        if sum(len(t) for t in lr) >= cap:
            break
    return torch.cat(lr), torch.cat(hr)


def load_folder(files, cap):
    """Plain image folder (UC Merced, Copernicus snapshots...): cut 256x256 tiles as HR; LR = bicubic x2 down."""
    hr = []
    for f in files:
        im = np.asarray(Image.open(f).convert("RGB"))
        for y in range(0, im.shape[0] - 255, 256):
            for x in range(0, im.shape[1] - 255, 256):
                hr.append(torch.from_numpy(im[y:y + 256, x:x + 256].copy()).permute(2, 0, 1))
        if len(hr) >= cap:
            break
    hr = torch.stack(hr[:cap]).float().div(255)
    lr = F.interpolate(hr, scale_factor=0.5, mode="bicubic", antialias=True).clamp(0, 1)
    return lr.half(), hr.half()


def batch(lr, hr, bs, ps=48):
    xs, ys = [], []
    for i in np.random.randint(0, len(lr), bs):
        y0, x0 = np.random.randint(0, 128 - ps + 1, 2)
        l = lr[i][:, y0:y0 + ps, x0:x0 + ps].float()
        h = hr[i][:, 2 * y0:2 * (y0 + ps), 2 * x0:2 * (x0 + ps)].float()
        k = random.randint(0, 3)
        l, h = torch.rot90(l, k, (1, 2)), torch.rot90(h, k, (1, 2))
        if random.random() < 0.5:
            l, h = l.flip(2), h.flip(2)
        xs.append(l); ys.append(h)
    return torch.stack(xs), torch.stack(ys)


def psnr(a, b):
    return (10 * torch.log10(1 / F.mse_loss(a, b))).item()


def validate(model, lr, hr, dev, n=48):
    model.eval(); sr, bi = [], []
    with torch.no_grad():
        for s in range(0, min(n, len(lr)), 8):
            l = lr[s:s + 8].float(); h = hr[s:s + 8].float().to(dev)
            sr.append(psnr(model(prep(l).to(dev)).clamp(0, 1), h))
            bi.append(psnr(F.interpolate(l, scale_factor=2, mode="bicubic").clamp(0, 1).to(dev), h))
    model.train()
    return float(np.mean(sr)), float(np.mean(bi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--model", default="edsr", choices=["edsr", "rcan"])
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--per-pair", type=int, default=40)
    ap.add_argument("--cap", type=int, default=4000)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    files = sorted(glob.glob(f"{a.data}/**/*_10m_b2b3b4b8.pt", recursive=True))
    loader = lambda fs, cap: load(fs, a.per_pair, cap)
    if not files:  # plain image folder mode
        files = sorted(p for e in ("tif", "tiff", "png", "jpg") for p in glob.glob(f"{a.data}/**/*.{e}", recursive=True))
        loader = load_folder
    random.Random(0).shuffle(files)
    nval = max(1, len(files) // 10)  # hold out files for validation
    tr_lr, tr_hr = loader(files[nval:], a.cap)
    va_lr, va_hr = loader(files[:nval], 200)
    print(f"device={dev} train_patches={len(tr_lr)} val_patches={len(va_lr)}")

    model = {"edsr": create_edsr, "rcan": create_rcan}[a.model]().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [a.iters // 2], 0.5)
    Path("weights").mkdir(exist_ok=True)
    best = -1

    for it in range(1, a.iters + 1):
        l, h = batch(tr_lr, tr_hr, a.bs)
        loss = F.l1_loss(model(prep(l).to(dev)), h.to(dev))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % 1000 == 0 or it == a.iters:
            p, b = validate(model, va_lr, va_hr, dev)
            print(f"iter {it}  loss {loss.item():.4f}  val PSNR {p:.2f} dB  (bicubic {b:.2f} dB)")
            if p > best:
                best = p
                torch.save(model.state_dict(), f"weights/{a.model}.pth")

    Path("samples").mkdir(exist_ok=True)
    for k in range(min(3, len(va_lr))):
        img = (va_lr[k].float().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        Image.fromarray(img).save(f"samples/sample_{k + 1}.png")
    print(f"Saved weights/{a.model}.pth and samples/*.png")


if __name__ == "__main__":
    main()
