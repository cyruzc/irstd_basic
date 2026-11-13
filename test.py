"""Unified test script for IRSTD benchmark — full evaluation.

Two evaluation protocols are available:

    basic : the classic protocol (IoU / nIoU / PD / FA, from metrics.py)
            IoU / nIoU (pixel level) + PD / FA (centroid distance < 3, greedy)
    opdc  : the protocol of "Rethinking Evaluation of IRSTD" (NeurIPS'25 D&B), also in metrics.py
            OPDC matching (IoU>=0.5 priority + distance<3 compensation, Hungarian)
            -> PD / FA, plus seg_iou / loc_iou / hIoU
    full  : both, side by side (default). Keeps numbers comparable to Tab.1 of the
            paper while showing what the classic protocol hides.

Usage:
    # test specific checkpoint
    python test.py --checkpoint log/IRSTD-1K/LightUNet/seed42/best_niou.pth.tar

    # test all seeds for a model+dataset, print mean±std
    python test.py --model_name LightUNet --dataset_name IRSTD-1K

    # test MiniUNet head2 variant
    python test.py --model_name MiniUNetHead2 --dataset_name IRSTD-1K

    # classic protocol only (fast, old numbers)
    python test.py --model_name LightUNet --protocol basic

    # dump per-image counters for later threshold-policy analysis
    python test.py --model_name LightUNet --dump out/lightunet_irstd1k.json
"""

import argparse
import glob
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import TestSetLoader
from metrics import HierarchicalErrorAnalysis, IoU, MatchingMetrics, PD_FA, nIoU
from net import MODEL_NAMES, Net

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

parser = argparse.ArgumentParser(
    description="PyTorch IRSTD Benchmark Test (LightUNet / MiniUNet head4-head2 / MicroUNet)"
)
parser.add_argument("--model_name", default="LightUNet", type=str, choices=MODEL_NAMES)
parser.add_argument("--dataset_name", default="IRSTD-1K", type=str)
parser.add_argument("--dataset_dir", default="./datasets", type=str)
parser.add_argument("--checkpoint", default=None, type=str,
                    help="specific checkpoint path (overrides auto-discovery)")
parser.add_argument("--best_by", default="best_niou", type=str,
                    choices=["best_iou", "best_niou"],
                    help="which best checkpoint to use when auto-discovering")
parser.add_argument("--log_dir", default="./log", type=str)
parser.add_argument("--threshold", type=float, default=0.5)
# ---- OPDC / hIoU protocol ---------------------------------------------- #
parser.add_argument("--protocol", default="full", type=str,
                    choices=["basic", "opdc", "full"],
                    help="basic = classic PD/FA; opdc = OPDC + hIoU; full = both")
parser.add_argument("--overlap_threshold", default=0.5, type=float,
                    help="OPDC stage-1 rule: pair IoU >= this value")
parser.add_argument("--distance_threshold", default=3, type=int,
                    help="OPDC stage-2 rule: centroid distance < this value")
parser.add_argument("--penalize_invalid", action="store_true",
                    help="solve OPDC as a single constrained assignment "
                         "(not identical to the official toolkit, can only add TPs)")
parser.add_argument("--error_analysis", action="store_true",
                    help="also report the 7-type hierarchical error decomposition (slow)")
parser.add_argument("--dump", default=None, type=str,
                    help="write per-image counters + config to this json file")

opt = parser.parse_args()


def _make_metrics():
    """Build the metric bundle for the requested protocol."""
    m = {"iou": IoU(), "niou": nIoU()}
    if opt.protocol in ("basic", "full"):
        m["pd_fa"] = PD_FA()
    if opt.protocol in ("opdc", "full"):
        m["opdc"] = MatchingMetrics(
            overlap_threshold=opt.overlap_threshold,
            distance_threshold=opt.distance_threshold,
            penalize_invalid=opt.penalize_invalid,
        )
    if opt.error_analysis:
        m["err"] = HierarchicalErrorAnalysis(
            overlap_threshold=opt.overlap_threshold,
            distance_threshold=opt.distance_threshold,
            penalize_invalid=opt.penalize_invalid,
        )
    return m


def _collect(m) -> dict:
    """Turn the accumulated metric objects into a flat result dict."""
    r = {"IoU": m["iou"].get(), "nIoU": m["niou"].get()}
    if "pd_fa" in m:
        pd, fa = m["pd_fa"].get()
        r["PD"] = pd
        r["FA"] = fa
    if "opdc" in m:
        c = m["opdc"].get()
        r["PD_opdc"] = c["pd"]
        r["FA_opdc"] = c["fa"]
        r["hIoU"] = c["hiou"]
        r["segIoU"] = c["seg_iou"]
        r["locIoU"] = c["loc_iou"]
        r["_counts"] = {k: c[k] for k in ("tp", "fp", "fn", "tp_iou", "fp_area", "total_area")}
    if "err" in m:
        e = m["err"].get()
        for k, v in e.items():
            r[f"err_{k}"] = v
    return r


_METRIC_ORDER = [
    "IoU", "nIoU", "PD", "FA",
    "PD_opdc", "FA_opdc", "hIoU", "segIoU", "locIoU",
    "err_seg_iou", "err_seg_mrg_err", "err_seg_itf_err", "err_seg_pcp_err",
    "err_loc_iou", "err_loc_s2m_err", "err_loc_m2s_err", "err_loc_itf_err", "err_loc_pcp_err",
    "err_hiou",
]


def _fmt(key, val):
    if key in ("FA", "FA_opdc"):
        return f"{val:.4e}"
    return f"{val:.4f}"


def _print_result(r, indent="  "):
    keys = [k for k in _METRIC_ORDER if k in r]
    lines = [f"{k}={_fmt(k, r[k])}" for k in keys]
    # two rows of keys for readability
    for i in range(0, len(lines), 5):
        print(indent + "  ".join(lines[i:i + 5]))


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

    m = _make_metrics()
    names = []

    with torch.no_grad():
        for img, gt_mask, size, name in test_loader:
            img = img.cuda()
            gt_mask = gt_mask.cuda()
            logits = net(img)
            prob = logits.sigmoid()
            prob = prob[:, :, : size[0], : size[1]]
            gt_mask = gt_mask[:, :, : size[0], : size[1]]
            binary = (prob > opt.threshold).float()

            m["iou"].update(binary, gt_mask)
            m["niou"].update(binary, gt_mask)

            pred_np = binary[0, 0].cpu().numpy().astype(bool)
            gt_np = gt_mask[0, 0].cpu().numpy() > 0

            if "pd_fa" in m:
                m["pd_fa"].update(binary[0, 0].cpu(), gt_mask[0, 0].cpu(), size)
            if "opdc" in m:
                m["opdc"].update(pred_np, gt_np, name=name)
            if "err" in m:
                m["err"].update(pred_np, gt_np)
            names.append(name)

    return _collect(m), model_name, dataset_name


def main():
    if opt.checkpoint:
        r, model_name, dataset_name = evaluate(opt.checkpoint)
        print(f"Checkpoint: {opt.checkpoint}")
        print(f"protocol={opt.protocol}  threshold={opt.threshold}  "
              f"overlap={opt.overlap_threshold}  distance={opt.distance_threshold}"
              + ("  penalize_invalid" if opt.penalize_invalid else ""))
        _print_result(r)
        if opt.dump:
            _write_dump(r, extra={"checkpoint": opt.checkpoint,
                                  "model_name": model_name,
                                  "dataset_name": dataset_name})
        return r

    model_dir = os.path.join(opt.log_dir, opt.dataset_name, opt.model_name)
    seed_dirs = sorted(glob.glob(os.path.join(model_dir, "seed*")))

    if not seed_dirs:
        print(f"No seed directories found in {model_dir}")
        exit(1)

    print(f"protocol={opt.protocol}  threshold={opt.threshold}  "
          f"overlap={opt.overlap_threshold}  distance={opt.distance_threshold}")

    results = []
    for sd in seed_dirs:
        best_pth = os.path.join(sd, f"{opt.best_by}.pth.tar")
        if not os.path.exists(best_pth):
            print(f"  skipping {sd} (no {opt.best_by}.pth.tar)")
            continue
        seed = os.path.basename(sd).split("_")[0].replace("seed", "")
        r, _, _ = evaluate(best_pth)
        results.append({"seed": seed, **{k: v for k, v in r.items() if not k.startswith("_")}})
        print(f"\n  [seed={seed}]")
        _print_result(r, indent="    ")
        if opt.dump:
            _write_dump(r, extra={"checkpoint": best_pth, "seed": seed})

    if len(results) >= 2:
        print(f"\n{'=' * 72}")
        keys = [k for k in _METRIC_ORDER if k in results[0]]
        for k in keys:
            vals = [r[k] for r in results]
            fmt = ".4e" if k in ("FA", "FA_opdc") else ".4f"
            print(f"  {k}: {np.mean(vals):{fmt}} +/- {np.std(vals):{fmt}}")

    if opt.dump:
        print(f"\nper-run counters dumped to: {opt.dump}")


def _write_dump(r, extra=None):
    payload = {
        "args": {k: v for k, v in vars(opt).items() if k != "dump"},
        "metrics": {k: v for k, v in r.items() if not k.startswith("_")},
        "counts": r.get("_counts", {}),
        "extra": extra or {},
    }
    d = os.path.dirname(os.path.abspath(opt.dump))
    os.makedirs(d, exist_ok=True)
    with open(opt.dump, "w") as f:
        json.dump(payload, f, indent=2, default=float)


if __name__ == "__main__":
    main()
