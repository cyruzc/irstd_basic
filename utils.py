"""Utility functions for IRSTD benchmark."""

import math
import os
import random

import numpy as np
import torch
from PIL import Image


def seed_pytorch(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def Normalized(img, img_norm_cfg):
    return (img - img_norm_cfg["mean"]) / img_norm_cfg["std"]


def get_img_norm_cfg(dataset_name, dataset_dir):
    presets = {
        "NUAA-SIRST": dict(mean=101.06385040283203, std=34.619606018066406),
        "NUDT-SIRST": dict(mean=107.80905151367188, std=33.02274703979492),
        "IRSTD-1k": dict(mean=87.4661865234375, std=39.71953201293945),
        "IRSTD-1K": dict(mean=87.4661865234375, std=39.71953201293945),
        "NUDT-SIRST-Sea": dict(mean=43.62403869628906, std=18.91838264465332),
        "SIRST3": dict(mean=62.10432052612305, std=23.96998405456543),
        "IRDST-real": dict(mean=101.54053497314453, std=56.49856185913086),
    }
    if dataset_name in presets:
        return presets[dataset_name]

    # compute from data
    train_list_path = os.path.join(dataset_dir, dataset_name, "img_idx", f"train_{dataset_name}.txt")
    if not os.path.exists(train_list_path):
        train_list_path = os.path.join(dataset_dir, dataset_name, "img_idx", "train.txt")
    with open(train_list_path, "r") as f:
        train_list = f.read().splitlines()
    img_dir = os.path.join(dataset_dir, dataset_name, "images")
    mean_list, std_list = [], []
    for name in train_list:
        for ext in (".png", ".jpg", ".bmp"):
            p = os.path.join(img_dir, name + ext)
            if os.path.exists(p):
                img = np.array(Image.open(p).convert("I"), dtype=np.float32)
                mean_list.append(img.mean())
                std_list.append(img.std())
                break
    cfg = dict(mean=float(np.mean(mean_list)), std=float(np.mean(std_list)))
    print(f"{dataset_name}\t{cfg}")
    return cfg


def random_crop(img, mask, patch_size, pos_prob=0.5):
    h, w = img.shape
    if min(h, w) < patch_size:
        img = np.pad(img, ((0, max(h, patch_size) - h), (0, max(w, patch_size) - w)), mode="constant")
        mask = np.pad(mask, ((0, max(h, patch_size) - h), (0, max(w, patch_size) - w)), mode="constant")
        h, w = img.shape

    if random.random() > pos_prob or mask.max() == 0:
        h_start = random.randint(0, h - patch_size)
        w_start = random.randint(0, w - patch_size)
    else:
        loc = np.where(mask > 0)
        idx = random.randint(0, max(len(loc[0]) - 1, 0))
        h_start = random.randint(max(0, loc[0][idx] - patch_size), min(loc[0][idx], h - patch_size))
        w_start = random.randint(max(0, loc[1][idx] - patch_size), min(loc[1][idx], w - patch_size))

    return (
        img[h_start : h_start + patch_size, w_start : w_start + patch_size],
        mask[h_start : h_start + patch_size, w_start : w_start + patch_size],
    )


def PadImg(img, times=32):
    h, w = img.shape
    if h % times != 0:
        img = np.pad(img, ((0, (h // times + 1) * times - h), (0, 0)), mode="constant")
    if w % times != 0:
        img = np.pad(img, ((0, 0), (0, (w // times + 1) * times - w)), mode="constant")
    return img


def get_optimizer(net, optimizer_name, scheduler_name, optimizer_settings, scheduler_settings):
    if optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(
            net.parameters(), lr=optimizer_settings["lr"],
            weight_decay=optimizer_settings.get("weight_decay", 1e-4),
        )
    elif optimizer_name == "Adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=optimizer_settings["lr"])
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(net.parameters(), lr=optimizer_settings["lr"])

    warmup = scheduler_settings.get("warmup", 0)
    total = scheduler_settings.get("epochs", 300)
    min_lr = scheduler_settings.get("min_lr", 1e-6)

    if scheduler_name == "CosineAnnealingLR":
        if warmup > 0:
            scheduler = _warmup_cosine(optimizer, warmup, total, min_lr)
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total, eta_min=min_lr
            )
    elif scheduler_name == "MultiStepLR":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=scheduler_settings["step"], gamma=scheduler_settings["gamma"]
        )
    else:
        scheduler = None

    return optimizer, scheduler


def _warmup_cosine(optimizer, warmup_epochs, total_epochs, min_lr):
    """Linear warmup + cosine annealing via LambdaLR."""
    base_lr = optimizer.param_groups[0]["lr"]

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs  # +1 so epoch 0 gives 1/warmup, not 0
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return min_lr / base_lr + (1 - min_lr / base_lr) * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
