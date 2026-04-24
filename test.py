"""Unified test script for IRSTD benchmark — full evaluation (IoU, nIoU, PD, FA).

Usage:
    # test specific checkpoint
    python test.py --checkpoint log/IRSTD-1K/LightUNet/seed42/best_niou.pth.tar

    # test all seeds for a model+dataset, print mean±std
    python test.py --model_name LightUNet --dataset_name IRSTD-1K
"""

import argparse
import glob
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import TestSetLoader
from metrics import IoU, nIoU, PD_FA
from net import Net

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

parser = argparse.ArgumentParser(description="PyTorch IRSTD Benchmark Test")
parser.add_argument("--model_name", default="LightUNet", type=str)
parser.add_argument("--dataset_name", default="IRSTD-1K", type=str)
parser.add_argument("--dataset_dir", default="./datasets", type=str)
parser.add_argument("--checkpoint", default=None, type=str,
                    help="specific checkpoint path (overrides auto-discovery)")
parser.add_argument("--best_by", default="best_niou", type=str,
                    choices=["best_iou", "best_niou"],
                    help="which best checkpoint to use when auto-discovering")
parser.add_argument("--log_dir", default="./log", type=str)
parser.add_argument("--threshold", type=float, default=0.5)

opt = parser.parse_args()


def evaluate(checkpoint_path):
    ckpt = torch.load(checkpoint_path, weights_only=False)
    model_name = ckpt.get("model_name", opt.model_name)
    dataset_name = ckpt.get("dataset_name", opt.dataset_name)
    loss_name = ckpt.get("loss_name", "BCE+SoftIoU")

    test_set = TestSetLoader(opt.dataset_dir, dataset_name, dataset_name)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)

    net = Net(model_name=model_name, loss_name=loss_name).cuda()
    net.load_state_dict(ckpt["state_dict"])
    net.eval()

    iou_metric = IoU()
    niou_metric = nIoU()
    pd_fa_metric = PD_FA()

    with torch.no_grad():
        for img, gt_mask, size, name in test_loader:
            img = img.cuda()
            gt_mask = gt_mask.cuda()
            logits = net(img)
            prob = logits.sigmoid()
            prob = prob[:, :, : size[0], : size[1]]
            gt_mask = gt_mask[:, :, : size[0], : size[1]]
            binary = (prob > opt.threshold).float()
            iou_metric.update(binary, gt_mask)
            niou_metric.update(binary, gt_mask)
            pd_fa_metric.update(binary[0, 0].cpu(), gt_mask[0, 0].cpu(), size)

    pd, fa = pd_fa_metric.get()
    return {"IoU": iou_metric.get(), "nIoU": niou_metric.get(), "PD": pd, "FA": fa}


if opt.checkpoint:
    r = evaluate(opt.checkpoint)
    print(f"Checkpoint: {opt.checkpoint}")
    print(f"IoU={r['IoU']:.4f}  nIoU={r['nIoU']:.4f}  PD={r['PD']:.4f}  FA={r['FA']:.8f}")
else:
    model_dir = os.path.join(opt.log_dir, opt.dataset_name, opt.model_name)
    seed_dirs = sorted(glob.glob(os.path.join(model_dir, "seed*")))

    if not seed_dirs:
        print(f"No seed directories found in {model_dir}")
        exit(1)

    results = []
    for sd in seed_dirs:
        best_pth = os.path.join(sd, f"{opt.best_by}.pth.tar")
        if not os.path.exists(best_pth):
            print(f"  skipping {sd} (no {opt.best_by}.pth.tar)")
            continue
        seed = os.path.basename(sd).split("_")[0].replace("seed", "")
        r = evaluate(best_pth)
        results.append({"seed": seed, **r})
        print(f"  seed={seed}  IoU={r['IoU']:.4f}  nIoU={r['nIoU']:.4f}  "
              f"PD={r['PD']:.4f}  FA={r['FA']:.8f}")

    if len(results) >= 2:
        print(f"\n{'='*60}")
        for k in ["IoU", "nIoU", "PD", "FA"]:
            vals = [r[k] for r in results]
            fmt = ".6f" if k == "FA" else ".4f"
            print(f"  {k}: {np.mean(vals):{fmt}} +/- {np.std(vals):{fmt}}")
