# Lensless Computational Imaging

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#checkpoints">Checkpoints</a> •
  <a href="#training">Training</a> •
  <a href="#inference">Inference</a> •
  <a href="#metrics">Metrics</a> •
  <a href="#speed-benchmark">Benchmark</a> •
  <a href="#demo">Demo</a> •
  <a href="#repository-structure">Structure</a> •
  <a href="#credits">Credits</a>
</p>

## About

This repository implements and compares reconstruction algorithms for a
**lensless camera**: a sensor with a programmable mask instead of a lens. The
sensor measurement is the scene convolved with the mask point-spread function
(PSF); recovering the scene is a deconvolution problem solved here with ADMM and
its learned variants.

Models are trained on the `train` split and evaluated on the `test` split of
[bezzam/DigiCam-Mirflickr-MultiMask-10K](https://huggingface.co/datasets/bezzam/DigiCam-Mirflickr-MultiMask-10K),
using **PSNR**, **LPIPS (VGG)**, **MSE**, and **SSIM** (computed on the aligned
region of interest).

Implemented methods:

- **ADMM-100** — classic ADMM, 100 iterations, fixed hyperparameters
  (μᵢ = 10⁻⁴, τ = 2·10⁻⁴), no training.
- **Unrolled ADMM-20** — 20 ADMM iterations with per-iteration trainable
  hyperparameters.
- **8M-modular LeADMM-5** — 5 ADMM iterations wrapped with learned DRUNet
  pre/post-processors, in three variants: pre-only, post-only, and pre+post.

## Installation

```bash
pip install -r requirements.txt
```

The DigiCam dataset (images + masks) is downloaded automatically from the
HuggingFace Hub on first run.

## Checkpoints

Download the trained checkpoints with the provided script:

```bash
python download_checkpoints.py
```

This fetches the checkpoints into `checkpoints/` (for example `checkpoints/modular_prepost.pth`). The Google Drive links are defined at the top of `download_checkpoints.py`. ADMM-100 has no checkpoint — it has no trainable parameters.

## Training

Training is configured with Hydra. Each variant has its own top-level config:

| Variant | Config | Command |
| --- | --- | --- |
| Unrolled ADMM-20 | `leadmm.yaml` | `python train.py` |
| Modular pre-only | `modular_pre.yaml` | `python train.py -cn=modular_pre` |
| Modular post-only | `modular_post.yaml` | `python train.py -cn=modular_post` |
| Modular pre+post | `modular_prepost.yaml` | `python train.py -cn=modular_prepost` |

Any hyperparameter can be overridden from the command line. Released
configuration for the modular pre+post model:

```bash
python train.py -cn=modular_prepost \
    optimizer.lr=1e-4 \
    dataloader.batch_size=6 \
    trainer.n_epochs=50 \
    trainer.epoch_len=300 \
    trainer.save_period=5 \
    writer.run_name=modular_prepost \
    trainer.save_dir=save_dir
```

The best model (by `test_PSNR`) is saved as `model_best.pth`. Losses, metrics,
and reconstruction images are logged to Comet ML.

To resume an interrupted run (continues the same Comet experiment):

```bash
python train.py -cn=modular_prepost \
    writer.run_name=modular_prepost \
    trainer.save_dir=save_dir \
    trainer.resume_from=model_best.pth
```

**Note on `normalize`.** ADMM-100, unrolled, and modular pre-only are evaluated
with min/max output normalization (`normalize=true`); modular post-only and
pre+post learn the output scale via the post-processor and use `normalize=false`
(the value is already set in each config).

## Inference

`inference.py` applies a model to a dataset and saves each reconstruction with
the same `ImageID` as the input (`data/saved/<save_path>/<part>/<ImageID>.png`).

Evaluate **ADMM-100** on the DigiCam `test` split (no checkpoint required):

```bash
python inference.py
```

Evaluate a **trained model** on the `test` split:

```bash
python inference.py -cn=inference \
    model=modular_prepost normalize=false \
    inferencer.from_pretrained=checkpoints/modular_prepost.pth
```

Run inference on a **custom directory** (`lensless/`, `masks/`, optional
`lensed/` with matching `ImageID` filenames):

```bash
python inference.py -cn=inference \
    model=modular_prepost normalize=false \
    datasets=custom_dir datasets.test.data_dir=/path/to/data \
    inferencer.from_pretrained=checkpoints/modular_prepost.pth \
    inferencer.save_path=my_recon
```

Metrics are printed if `lensed/` ground truth is present. By default no
experiment tracker is used; pass `inferencer.log_to_writer=true` to log to
Comet ML.

## Metrics

Compute metrics between a folder of reconstructions and a folder of lensed
images (matched by filename, compared on the region of interest):

```bash
python calculate_metrics.py --lensed_dir /path/to/lensed --recon_dir /path/to/reconstructions
```

Prints PSNR, MSE, SSIM, and LPIPS.

## Speed benchmark

Measure per-image reconstruction time for all methods:

```bash
python benchmark.py
```

## Demo

`demo.ipynb` is an end-to-end inference demo for a fresh Google Colab session:
it clones the repository, installs dependencies, downloads the checkpoints,
downloads and extracts a user-provided `.zip` dataset (Google Drive URL), runs
`inference.py`, visualizes lensed / lensless / reconstruction, and computes
metrics if ground truth is available. Set `DATASET_URL` and run all cells.

## Repository structure

```
src/
  configs/        Hydra configs
  datasets/       DigiCam (HuggingFace) and CustomDir datasets, collate function
  model/          ADMM layer, LeADMM (ADMM-100 / unrolled), DRUNet, modular models
  loss/           MSE + LPIPS reconstruction loss
  metrics/        PSNR / LPIPS / MSE / SSIM image metric
  trainer/        Training and inference loops
  logger/         Comet ML / W&B writers
lensless_helpers/ PSF simulation and dataset preprocessing
train.py                 Training entry point
inference.py             Inference / evaluation entry point
calculate_metrics.py     Standalone metrics script
benchmark.py             Reconstruction speed benchmark
download_checkpoints.py  Checkpoint download script
demo.ipynb               Inference demo notebook
```

## Credits

This repository is based on the [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template)