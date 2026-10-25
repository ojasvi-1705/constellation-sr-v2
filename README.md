# Constellation SR

Prototype for the SIH problem: turn a low-resolution satellite image into a 2x higher-resolution one.

Pipeline (unchanged): radiometric calibration -> atmospheric correction -> EDSR / RCAN (tiled inference) -> PNG.
Inputs: upload, image URL, or bundled samples. Runs on the GPU if PyTorch can see one, else CPU.
Each request is logged to SQLite with PSNR / SSIM / SAM against a bicubic upscale of the input (a sanity check only).

Endpoints: `/` (UI), `/health`, `/info`, `POST /api/v1/infer`, `POST /api/v1/infer-url`, `POST /api/v1/batch`,
`POST /api/v1/infer-url`, `GET /api/v1/status/{id}`, `GET /api/v1/download/{id}`.

## Train on the trial dataset
```
python check_gpu.py                                  # diagnose GPU problems
python download_data.py                              # UC Merced copy from GitHub (~250 MB)
python train.py --data data/ucmerced --model edsr --iters 10000 --bs 8
```
Low-resolution inputs are made by bicubic 2x downsampling of the 256x256 images, the standard SR benchmark setup.
Weights are saved to `weights/edsr.pth`; restart the server to load them. `/info` shows whether they are loaded.

Dataset: UC Merced Land Use (Yang & Newsam, 2010), USGS imagery at 0.3 m; this GitHub copy has 16 of the 21 classes.
The optional real-pair dataset (Sentinel-2 to VENuS, SEN2VENuS) also works with `train.py`.

Tests: `python -m pytest tests -v`. Docker (untested): `docker compose -f docker/docker-compose.yml up`.

Large inputs are processed tile by tile (128 px tiles, 32 px overlap), so a 4 GB GPU is enough. Longest input side is capped by `MAX_SIDE` (default 512).
See QUICK_START.md for setup, GPU check and training.
