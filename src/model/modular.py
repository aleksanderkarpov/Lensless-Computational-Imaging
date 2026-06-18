import torch.nn.functional as F
from torch import nn

from src.model.drunet import DRUnet
from src.model.leadmm import LeADMM


def apply_net(net, x):
    x = x.permute(0, 3, 1, 2)
    h, w = x.shape[-2:]
    ph = (8 - h % 8) % 8
    pw = (8 - w % 8) % 8
    x = F.pad(x, (0, pw, 0, ph))
    x = net(x)
    x = x[:, :, :h, :w]
    return x.permute(0, 2, 3, 1)


class ModularPre(nn.Module):
    def __init__(self, iters=5, channels=(32, 64, 128, 256)):
        super().__init__()
        self.pre = DRUnet(channels)
        self.leadmm = LeADMM(iters=iters, trainable=True)

    def forward(self, lensless, psf, **batch):
        lensless = apply_net(self.pre, lensless)
        return self.leadmm(lensless=lensless, psf=psf)


class ModularPost(nn.Module):
    def __init__(self, iters=5, channels=(32, 64, 128, 256)):
        super().__init__()
        self.post = DRUnet(channels)
        self.leadmm = LeADMM(iters=iters, trainable=True)

    def forward(self, lensless, psf, **batch):
        out = self.leadmm(lensless=lensless, psf=psf)
        return {"reconstructed": apply_net(self.post, out["reconstructed"])}


class ModularPrePost(nn.Module):
    def __init__(self, iters=5, channels=(32, 64, 116, 128)):
        super().__init__()
        self.pre = DRUnet(channels)
        self.post = DRUnet(channels)
        self.leadmm = LeADMM(iters=iters, trainable=True)

    def forward(self, lensless, psf, **batch):
        lensless = apply_net(self.pre, lensless)
        out = self.leadmm(lensless=lensless, psf=psf)
        return {"reconstructed": apply_net(self.post, out["reconstructed"])}
