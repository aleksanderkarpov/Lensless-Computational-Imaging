import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torchmetrics import (
    MeanSquaredError,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity

from lensless_helpers.preprocessor import ALIGNMENT

TOP, LEFT = ALIGNMENT["top_left"]
HEIGHT, WIDTH = ALIGNMENT["height"], ALIGNMENT["width"]


def load_rgb(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_recon(path):
    image = load_rgb(path)
    return image[TOP:TOP + HEIGHT, LEFT:LEFT + WIDTH]


def load_lensed(path):
    image = load_rgb(path)
    return cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)


def to_tensor(image, device):
    image = image.astype(np.float32) / 255
    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).contiguous().to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lensed_dir", required=True)
    parser.add_argument("--recon_dir", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    mse = MeanSquaredError().to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to(device)

    lensed_dir, recon_dir = Path(args.lensed_dir), Path(args.recon_dir)
    n = 0
    for recon_path in sorted(recon_dir.iterdir()):
        lensed_matches = sorted(lensed_dir.glob(f"{recon_path.stem}.*"))
        if not lensed_matches:
            continue
        recon = to_tensor(load_recon(recon_path), device)
        lensed = to_tensor(load_lensed(lensed_matches[0]), device)
        psnr.update(recon, lensed)
        mse.update(recon, lensed)
        ssim.update(recon, lensed)
        lpips.update(recon, lensed)
        n += 1

    if n == 0:
        print("No matching image pairs found.")
        return

    print(f"images: {n}")
    print(f"PSNR:  {psnr.compute().item():.6f}")
    print(f"MSE:   {mse.compute().item():.6f}")
    print(f"SSIM:  {ssim.compute().item():.6f}")
    print(f"LPIPS: {lpips.compute().item():.6f}")


if __name__ == "__main__":
    main()
