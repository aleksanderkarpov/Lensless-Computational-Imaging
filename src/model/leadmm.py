from torch import nn
import torch
from src.model.admm_layer import ADMMBase, ADMMLayer


class LeADMM(ADMMBase):

    def __init__(self, iters=5, trainable=False):
        super().__init__()
        self.trainable = trainable
        self.admm_module = nn.ModuleList([
            ADMMLayer(trainable) for _ in range(iters)
        ])

    def forward(self, lensless, psf, **batch):
        device = lensless.device
        Ctb, shape = self.pad(lensless)
        B, H, W, C = shape
        psf_transformed = torch.fft.fft2(torch.fft.ifftshift(self.pad(psf)[0], dim=(1, 2)), dim=(1, 2))

        CtC, _ = self.pad(torch.ones(B, H, W, C, device=device))

        if self.trainable:
            xdivider = None
        else:
            mine = torch.zeros(2 * H, 2 * W, device=device)
            mine[0, 0] = 4
            mine[0, 1] = mine[1, 0] = mine[0, -1] = mine[-1, 0] = -1
            psiTpsi = torch.fft.fft2(mine).real[None, ..., None]
            layer0 = self.admm_module[0]
            xdivider = layer0.log_mu1.exp() * psf_transformed.abs() ** 2 + layer0.log_mu2.exp() * psiTpsi + layer0.log_mu3.exp()

        x = torch.zeros_like(Ctb)
        alpha1 = torch.zeros_like(x)
        alpha2 = torch.zeros_like(self.psi(x))
        alpha3 = torch.zeros_like(x)

        for layer in self.admm_module:
            x, alpha1, alpha2, alpha3 = layer(x, alpha1, alpha2, alpha3, Ctb, psf_transformed, CtC, xdivider, H, W, device)

        return {"reconstructed": self.crop(x, shape)}
