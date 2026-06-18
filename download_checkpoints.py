from pathlib import Path

import gdown

CHECKPOINTS = {
    "modular_prepost": "https://drive.google.com/uc?export=download&id=14AfjpDFrYf0XXX9M53tgpMfWyaZ-f_Om",
    "modular_pre": "https://drive.google.com/uc?export=download&id=1_HGNyBrAuXF8W7mk0GD8E0q9ko0M6aGU",
    "modular_post": "https://drive.google.com/uc?export=download&id=16diBP8V_WansZ4M-ZzqpH-sNgqReDeEj",
    "unrolled": "https://drive.google.com/uc?export=download&id=1B8YexApuCAlJceu_VW_f3beiW-77WHhD",
}


def main():
    out_dir = Path("checkpoints")
    out_dir.mkdir(exist_ok=True)
    for name, url in CHECKPOINTS.items():
        if not url:
            continue
        out = out_dir / f"{name}.pth"
        if out.exists():
            print(f"{out} already exists, skipping")
            continue
        gdown.download(url, str(out), quiet=False, fuzzy=True)


if __name__ == "__main__":
    main()
