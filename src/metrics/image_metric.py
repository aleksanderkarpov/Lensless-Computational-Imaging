import torch

from lensless_helpers.preprocessor import ALIGNMENT
from src.metrics.base_metric import BaseMetric

TOP, LEFT = ALIGNMENT["top_left"]
HEIGHT, WIDTH = ALIGNMENT["height"], ALIGNMENT["width"]


class ImageMetric(BaseMetric):
    def __init__(self, metric, device, normalize=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metric = metric.to(device)
        self.normalize = normalize

    def __call__(self, reconstructed, lensed, **batch):
        return self.metric(self._prepare(reconstructed), self._prepare(lensed))

    def _prepare(self, x):
        x = x[:, TOP:TOP + HEIGHT, LEFT:LEFT + WIDTH, :]
        if self.normalize:
            lo = x.amin(dim=(1, 2, 3), keepdim=True)
            hi = x.amax(dim=(1, 2, 3), keepdim=True)
            x = (x - lo) / (hi - lo + 1e-8)
        else:
            x = x.clamp(0, 1)
        return x.permute(0, 3, 1, 2).contiguous()
