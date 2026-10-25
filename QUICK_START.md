# Quick start (Windows, VS Code terminal)
```
& "C:\Users\Ojasvi\AppData\Local\Programs\Python\Python311\python.exe" -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip uninstall -y torch
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
python check_gpu.py
```
`check_gpu.py` tells you exactly what is wrong if the GPU is not detected.

Train (trial dataset, ~250 MB, standard library only):
```
python download_data.py
python train.py --data data/ucmerced --model edsr --iters 6000 --bs 8
```
Or use your own Sentinel-2 images from https://browser.dataspace.copernicus.eu/ (save true-colour PNG/JPG
snapshots, at least 256x256, into `data/copernicus/`) and train with `--data data/copernicus`.

Run: `python -m uvicorn api.app:app --reload`, then open http://127.0.0.1:8000
