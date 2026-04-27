"""Dataset loaders for IRSTD benchmark.

Directory layout (per dataset):
    datasets/<dataset_name>/
        images/          # grayscale PNG/BMP
        masks/           # binary PNG/BMP
        img_idx/
            train_<name>.txt
            test_<name>.txt
"""

import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from utils import PadImg, Normalized, get_img_norm_cfg, random_crop


class TrainSetLoader(Dataset):
    def __init__(self, dataset_dir, dataset_name, patch_size, img_norm_cfg=None):
        self.dataset_dir = os.path.join(dataset_dir, dataset_name)
        self.patch_size = patch_size

        split_path = os.path.join(self.dataset_dir, "img_idx", f"train_{dataset_name}.txt")
        if not os.path.exists(split_path):
            split_path = os.path.join(self.dataset_dir, "img_idx", "train.txt")
        with open(split_path, "r") as f:
            self.sample_list = f.read().splitlines()

        self.img_norm_cfg = img_norm_cfg or get_img_norm_cfg(dataset_name, dataset_dir)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        name = self.sample_list[idx]
        img = self._load_image(name)
        mask = self._load_mask(name)

        img, mask = random_crop(img, mask, self.patch_size)
        img, mask = _augment(img, mask)

        img = torch.from_numpy(np.ascontiguousarray(img[np.newaxis, :]))
        mask = torch.from_numpy(np.ascontiguousarray(mask[np.newaxis, :]))
        return img, mask

    def _load_image(self, name):
        for ext in (".png", ".bmp", ".jpg"):
            p = os.path.join(self.dataset_dir, "images", name + ext)
            if os.path.exists(p):
                return Normalized(np.array(Image.open(p).convert("I"), dtype=np.float32), self.img_norm_cfg)
        raise FileNotFoundError(f"Image not found: {name}")

    def _load_mask(self, name):
        for ext in (".png", ".bmp", ".jpg"):
            p = os.path.join(self.dataset_dir, "masks", name + ext)
            if os.path.exists(p):
                mask = np.array(Image.open(p), dtype=np.float32) / 255.0
                if mask.ndim > 2:
                    mask = mask[:, :, 0]
                return mask
        raise FileNotFoundError(f"Mask not found: {name}")


class TestSetLoader(Dataset):
    def __init__(self, dataset_dir, train_dataset_name, test_dataset_name, img_norm_cfg=None):
        self.dataset_dir = os.path.join(dataset_dir, test_dataset_name)
        with open(os.path.join(self.dataset_dir, "img_idx", f"test_{test_dataset_name}.txt"), "r") as f:
            self.sample_list = f.read().splitlines()
        self.img_norm_cfg = img_norm_cfg or get_img_norm_cfg(train_dataset_name, dataset_dir)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        name = self.sample_list[idx]
        img = self._load_image(name)
        mask = self._load_mask(name)

        h, w = img.shape
        img = PadImg(img)
        mask = PadImg(mask)

        img = torch.from_numpy(np.ascontiguousarray(img[np.newaxis, :]))
        mask = torch.from_numpy(np.ascontiguousarray(mask[np.newaxis, :]))
        return img, mask, [h, w], name

    def _load_image(self, name):
        for ext in (".png", ".bmp", ".jpg"):
            p = os.path.join(self.dataset_dir, "images", name + ext)
            if os.path.exists(p):
                return Normalized(np.array(Image.open(p).convert("I"), dtype=np.float32), self.img_norm_cfg)
        raise FileNotFoundError(f"Image not found: {name}")

    def _load_mask(self, name):
        for ext in (".png", ".bmp", ".jpg"):
            p = os.path.join(self.dataset_dir, "masks", name + ext)
            if os.path.exists(p):
                mask = np.array(Image.open(p), dtype=np.float32) / 255.0
                if mask.ndim > 2:
                    mask = mask[:, :, 0]
                return mask
        raise FileNotFoundError(f"Mask not found: {name}")


def _augment(img, mask):
    if random.random() < 0.5:
        img = img[::-1, :].copy()
        mask = mask[::-1, :].copy()
    if random.random() < 0.5:
        img = img[:, ::-1].copy()
        mask = mask[:, ::-1].copy()
    if random.random() < 0.5:
        img = img.T.copy()
        mask = mask.T.copy()
    return img, mask
