import argparse
import time

import torch

from src.model.leadmm import LeADMM
from src.model.modular import ModularPost, ModularPre, ModularPrePost


def build_models():
    return {
        "ADMM-100": LeADMM(iters=100, trainable=False),
        "Unrolled-20": LeADMM(iters=20, trainable=True),
        "Modular-Pre": ModularPre(iters=5),
        "Modular-Post": ModularPost(iters=5),
        "Modular-PrePost": ModularPrePost(iters=5),
    }


@torch.no_grad()
def measure(model, lensless, psf, device, warmup, runs):
    model.eval()
    for _ in range(warmup):
        model(lensless=lensless, psf=psf)
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(runs):
        model(lensless=lensless, psf=psf)
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--height", type=int, default=380)
    parser.add_argument("--width", type=int, default=507)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lensless = torch.rand(args.batch_size, args.height, args.width, 3, device=device)
    psf = torch.rand(args.batch_size, args.height, args.width, 3, device=device) * 0.01

    print(f"device: {device}, batch_size: {args.batch_size}, runs: {args.runs}")
    for name, model in build_models().items():
        model = model.to(device)
        t = measure(model, lensless, psf, device, args.warmup, args.runs)
        per_image = t / args.batch_size
        print(f"{name:18s} {per_image * 1000:9.1f} ms/image   {1 / per_image:8.1f} img/s")


if __name__ == "__main__":
    main()
