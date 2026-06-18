import lpips
import torch
import torch.nn.functional as F
from torch import nn

from lensless_helpers.preprocessor import ALIGNMENT

TOP, LEFT = ALIGNMENT["top_left"]
HEIGHT, WIDTH = ALIGNMENT["height"], ALIGNMENT["width"]


class Loss(nn.Module):

    def __init__(self):
        super().__init__()
        self.lpips_loss = lpips.LPIPS(net="vgg")

    def _prepare(self, x):
        x = x[:, TOP:TOP + HEIGHT, LEFT:LEFT + WIDTH, :]
        return x.permute(0, 3, 1, 2).contiguous()

    def forward(self, reconstructed: torch.Tensor, lensed: torch.Tensor, **batch):
        pred = self._prepare(reconstructed)
        target = self._prepare(lensed)
        mse = F.mse_loss(pred, target)
        lpips_value = self.lpips_loss(
            pred.clamp(0, 1), target.clamp(0, 1), normalize=True
        ).mean()
        return {"mse_loss": mse, "lpips_loss": lpips_value, "loss": mse + lpips_value}
