"""Diagnose GPU problems: python check_gpu.py"""
import sys
import torch

print("python", sys.version.split()[0], "| torch", torch.__version__, "| built for CUDA:", torch.version.cuda)
if not torch.cuda.is_available():
    print("CUDA is NOT available.")
    if torch.version.cuda is None:
        print("-> This is the CPU-only PyTorch. Fix:\n   python -m pip uninstall -y torch\n"
              "   python -m pip install torch --index-url https://download.pytorch.org/whl/cu126")
    else:
        print("-> PyTorch has CUDA but cannot see the GPU. Update the NVIDIA driver (check nvidia-smi), then restart.")
    sys.exit(1)
p = torch.cuda.get_device_properties(0)
print(f"GPU: {p.name}, {p.total_memory / 2**30:.1f} GB")
torch.nn.Conv2d(64, 64, 3, padding=1).cuda()(torch.randn(1, 64, 128, 128, device="cuda"))
torch.cuda.synchronize()
print("GPU test OK - use: python train.py --data data/ucmerced --bs 8")
