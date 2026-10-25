"""Download the UC Merced trial dataset (GitHub copy: 16 classes, 1,600 aerial images, 256x256, ~250 MB)
into data/ucmerced. Uses only the Python standard library - no 7-Zip needed."""
import urllib.request
import zipfile
from pathlib import Path

URL = "https://codeload.github.com/rkushell/UCMerced_LandUse/zip/refs/heads/main"
out, tmp = Path("data/ucmerced"), Path("data/ucmerced.zip")
out.mkdir(parents=True, exist_ok=True)
if not tmp.exists():
    with urllib.request.urlopen(URL, timeout=60) as r, open(tmp, "wb") as f:
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            print(f"\rdownloaded {done / 2**20:.0f} MB", end="", flush=True)
    print()
with zipfile.ZipFile(tmp) as z:
    names = [n for n in z.namelist() if n.lower().endswith((".tif", ".png", ".jpg"))]
    for n in names:
        (out / Path(n).parent.name).mkdir(exist_ok=True)
        (out / Path(n).parent.name / Path(n).name).write_bytes(z.read(n))
tmp.unlink()
print(f"Extracted {len(names)} images to {out}")
