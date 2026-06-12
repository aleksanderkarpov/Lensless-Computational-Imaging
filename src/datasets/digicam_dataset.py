import os
import numpy as np
from datasets import load_dataset
from huggingface_hub import snapshot_download
from lensless_helpers.preprocessor import get_dataset_object
from lensless_helpers.psf import simulate_psf_from_mask
from src.datasets.base_dataset import BaseDataset


class DigiCamDataset(BaseDataset):
    def __init__(self, dataset_name, split="train", *args, **kwargs):
        index = load_dataset(dataset_name, split=split)
        repo_root = snapshot_download(dataset_name, repo_type="dataset", allow_patterns="masks/*")
        self.masks_dir = os.path.join(repo_root, "masks")
        super().__init__(index, *args, **kwargs)

    @staticmethod
    def _assert_index_is_valid(index):
        for field in ["lensless", "lensed", "mask_label"]:
            assert field in index.column_names, f"HF dataset must contain '{field}'."

    @staticmethod
    def _shuffle_and_limit_index(index, limit, shuffle_index):
        if shuffle_index:
            index = index.shuffle(seed=42)
        if limit is not None:
            index = index.select(range(limit))
        return index

    def __getitem__(self, ind):
        item = self._index[ind]
        psf = self._get_psf(item["mask_label"])
        lensed, lensless, psf = get_dataset_object(item["lensed"], item["lensless"], psf)
        instance_data = {
            "id": str(ind),
            "lensless": lensless,
            "lensed": lensed,
            "psf": psf,
        }
        return self.preprocess_data(instance_data)

    def _get_psf(self, mask_label):
        if mask_label not in self.psf_cache:
            mask_vals = np.load(os.path.join(self.masks_dir, f"mask_{mask_label}.npy"))
            self.psf_cache[mask_label] = simulate_psf_from_mask(mask_vals)
        return self.psf_cache[mask_label]
