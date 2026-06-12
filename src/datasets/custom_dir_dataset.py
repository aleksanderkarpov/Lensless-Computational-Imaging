from pathlib import Path
import numpy as np
from lensless_helpers.preprocessor import get_dataset_object
from lensless_helpers.psf import simulate_psf_from_mask
from src.datasets.base_dataset import BaseDataset


class CustomDirDataset(BaseDataset):
    def __init__(self, data_dir, *args, **kwargs):
        data_dir = Path(data_dir)
        lensless_dir = data_dir / "lensless"
        masks_dir = data_dir / "masks"
        lensed_dir = data_dir / "lensed"

        index = []
        for lensless_path in sorted(lensless_dir.iterdir()):
            stem = lensless_path.stem
            mask_matches = sorted(masks_dir.glob(f"{stem}.*"))
            entry = {
                "id": stem,
                "lensless": str(lensless_path),
                "mask": str(mask_matches[0]),
            }
            if lensed_dir.exists():
                lensed_matches = sorted(lensed_dir.glob(f"{stem}.*"))
                entry["lensed"] = str(lensed_matches[0])
            index.append(entry)

        super().__init__(index, *args, **kwargs)

    def __getitem__(self, ind):
        data_dict = self._index[ind]

        lensless_object = self._load_image(data_dict["lensless"])
        lensed_object = (self._load_image(data_dict["lensed"]) if "lensed" in data_dict else None)
        psf = self._get_psf(data_dict["mask"])

        lensed, lensless, psf = get_dataset_object(lensed_object, lensless_object, psf)
        instance_data = {
            "id": data_dict["id"],
            "lensless": lensless,
            "lensed": lensed,
            "psf": psf,
        }
        return self.preprocess_data(instance_data)

    def _get_psf(self, mask_path):
        if mask_path not in self.psf_cache:
            mask_vals = np.load(mask_path)
            self.psf_cache[mask_path] = simulate_psf_from_mask(mask_vals)
        return self.psf_cache[mask_path]
