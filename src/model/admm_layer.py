from torch import nn
import torch


class ADMMBase(nn.Module):
    def pad(self, x):
        B, H, W, C = x.shape
        out = x.new_zeros(B, 2 * H, 2 * W, C)
        out[:, H // 2:H // 2 + H, W // 2:W // 2 + W] = x
        return out, (B, H, W, C)

    def crop(self, x, shape):
        B, H, W, C = shape
        return x[:, H // 2: H // 2 + H, W // 2: W // 2 + W]

    def H(self, x, psf_transformed):
        x = torch.fft.fft2(x, dim=(1, 2))
        out = x * psf_transformed
        out = torch.fft.ifft2(out, dim=(1, 2)).real
        return out

    def HT(self, x, psf_transformed):
        x = torch.fft.fft2(x, dim=(1, 2))
        out = x * psf_transformed.conj()
        out = torch.fft.ifft2(out, dim=(1, 2)).real
        return out

    def psi(self, x):
        dx = torch.roll(x, -1, dims=2) - x
        dy = torch.roll(x, -1, dims=1) - x
        return torch.stack([dx, dy], dim=-1)

    def psiT(self, x):
        dx = x[..., 0]
        dy = x[..., 1]

        dxT = torch.roll(dx, 1, dims=2) - dx
        dyT = torch.roll(dy, 1, dims=1) - dy

        return dxT + dyT

    def Tau(self, x, threshold):
        return torch.sign(x) * torch.clamp(x.abs() - threshold, min=0)


class ADMMLayer(ADMMBase):
    def __init__(self, trainable=False):
        super().__init__()
        self.trainable = trainable

        if trainable:
            self.log_mu1 = nn.Parameter(torch.tensor(1e-4).log(), requires_grad=True)
            self.log_mu2 = nn.Parameter(torch.tensor(1e-4).log(), requires_grad=True)
            self.log_mu3 = nn.Parameter(torch.tensor(1e-4).log(), requires_grad=True)
            self.log_tau = nn.Parameter(torch.tensor(2e-4).log(), requires_grad=True)
        else:
            self.log_mu1 = nn.Parameter(torch.tensor(1e-4).log(), requires_grad=False)
            self.log_mu2 = nn.Parameter(torch.tensor(1e-4).log(), requires_grad=False)
            self.log_mu3 = nn.Parameter(torch.tensor(1e-4).log(), requires_grad=False)
            self.log_tau = nn.Parameter(torch.tensor(2e-4).log(), requires_grad=False)

    def forward(self, x, alpha1, alpha2, alpha3, Ctb, psf_transformed, CtC, xdivider, H, W, device):
        mu1 = self.log_mu1.exp()
        mu2 = self.log_mu2.exp()
        mu3 = self.log_mu3.exp()
        tau = self.log_tau.exp()
        if self.trainable:
            number = tau
        else:
            number = tau / mu2
        u = self.Tau(self.psi(x) + alpha2 / mu2, number)
        vdivider = CtC + mu1
        v = (alpha1 + mu1 * self.H(x, psf_transformed) + Ctb) / vdivider
        w = torch.clamp(alpha3 / mu3 + x, min=0)
        r = (mu3 * w - alpha3) + self.psiT(mu2 * u - alpha2) + self.HT(mu1 * v - alpha1, psf_transformed)
        if xdivider is None:
            mine = torch.zeros(2 * H, 2 * W, device=device)
            mine[0, 0] = 4
            mine[0, 1] = mine[1, 0] = mine[0, -1] = mine[-1, 0] = -1
            psiTpsi = torch.fft.fft2(mine).real[None, ..., None]
            xdivider = mu1 * psf_transformed.abs() ** 2 + mu2 * psiTpsi + mu3
        x = torch.fft.ifft2(torch.fft.fft2(r, dim=(1, 2)) / xdivider, dim=(1, 2)).real
        alpha1 = alpha1 + mu1 * (self.H(x, psf_transformed) - v)
        alpha2 = alpha2 + mu2 * (self.psi(x) - u)
        alpha3 = alpha3 + mu3 * (x - w)
        return x, alpha1, alpha2, alpha3
