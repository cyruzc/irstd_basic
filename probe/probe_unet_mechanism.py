"""Probe hard-case entanglement in UNet-family checkpoints.

This script is designed around MiniUNetHead2-style checkpoints, but works for
any UNet-family checkpoint that exposes `forward_features()`.
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
TMP_ROOT = os.path.join(REPO_ROOT, "tmp")
os.makedirs(TMP_ROOT, exist_ok=True)

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(TMP_ROOT, "matplotlib"))

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation
from skimage import measure
from skimage.feature import peak_local_max
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from dataset import TestSetLoader
from net import Net
from utils import seed_pytorch


CASE_COLORS = {
    "tp_normal": "#1b9e77",
    "fn_entangled": "#d95f02",
    "fn_weak": "#7570b3",
    "fp_clutter": "#e7298a",
    "tn_targetlike": "#a6761d",
    "tn_high_response": "#66a61e",
    "tn_near_boundary": "#1f78b4",
    "tn_easy": "#999999",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Probe UNet hard-case entanglement.")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--dataset_name", default=None, type=str)
    parser.add_argument("--dataset_dir", default="./datasets", type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--canonical_name", default=None, type=str,
                        help="Optional label used in the output summary.")
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--workers", default=0, type=int)
    parser.add_argument("--roi_radius", default=10, type=int,
                        help="ROI radius in full-resolution pixels.")
    parser.add_argument("--match_radius", default=3.0, type=float)
    parser.add_argument("--topk_pixels", default=5, type=int)
    parser.add_argument("--fn_ref_quantile", default=0.25, type=float,
                        help="FN-entangled dec2 threshold uses this quantile of TP dec2 evidence.")
    parser.add_argument("--fn_logit_max", default=0.0, type=float,
                        help="FN-entangled GT-mask mean logit must be <= this value.")
    parser.add_argument("--tn_targetlike_quantile", "--tn_hard_quantile",
                        dest="tn_targetlike_quantile", default=0.995, type=float,
                        help="Per-image dec2 energy quantile used to mine target-like TN peaks.")
    parser.add_argument("--tn_targetlike_per_image", default=2, type=int)
    parser.add_argument("--tn_high_response_per_image", default=3, type=int)
    parser.add_argument("--tn_near_boundary_per_image", default=2, type=int)
    parser.add_argument("--tn_easy_per_image", default=2, type=int)
    parser.add_argument("--tn_boundary_radius", default=6, type=int)
    parser.add_argument("--tn_boundary_quantile", default=0.98, type=float)
    parser.add_argument("--tn_easy_quantile", default=0.25, type=float)
    parser.add_argument("--tn_easy_prob_max", default=0.05, type=float)
    parser.add_argument("--tn_peak_min_distance", default=8, type=int)
    parser.add_argument("--limit_images", default=0, type=int,
                        help="0 = all images.")
    parser.add_argument("--probe_folds", default=5, type=int)
    parser.add_argument("--max_scatter_points", default=4000, type=int)
    parser.add_argument("--save_examples", default=4, type=int)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def crop_map(tensor, h, w):
    return tensor[..., :h, :w]


def upsample_to_input(feat, h, w):
    if feat.shape[-2:] == (h, w):
        return feat
    return F.interpolate(feat, size=(h, w), mode="bilinear", align_corners=False)


def feature_energy(feat_map):
    return torch.sqrt(torch.mean(feat_map * feat_map, dim=0) + 1e-12)


def roi_bounds(cy, cx, radius, h, w):
    y0 = max(cy - radius, 0)
    y1 = min(cy + radius + 1, h)
    x0 = max(cx - radius, 0)
    x1 = min(cx + radius + 1, w)
    return y0, y1, x0, x1


def patch_mean_vector(feat_map, cy, cx, radius):
    h, w = feat_map.shape[-2:]
    y0, y1, x0, x1 = roi_bounds(cy, cx, radius, h, w)
    patch = feat_map[:, y0:y1, x0:x1]
    return patch.reshape(feat_map.shape[0], -1).mean(dim=1)


def patch_center_vector(feat_map, cy, cx):
    h, w = feat_map.shape[-2:]
    cy = int(np.clip(cy, 0, h - 1))
    cx = int(np.clip(cx, 0, w - 1))
    return feat_map[:, cy, cx]


def masked_mean(feat_map, mask):
    flat = feat_map[:, mask]
    if flat.numel() == 0:
        return torch.zeros(feat_map.shape[0], device=feat_map.device)
    return flat.mean(dim=1)


def masked_topk_mean(scalar_map, mask, k):
    vals = scalar_map[mask]
    if vals.numel() == 0:
        return 0.0
    k = min(k, vals.numel())
    return float(torch.topk(vals, k=k).values.mean().item())


def masked_mean_scalar(scalar_map, mask):
    vals = scalar_map[mask]
    if vals.numel() == 0:
        return 0.0
    return float(vals.mean().item())


def compute_structure_features(img_np, cy, cx, radius):
    h, w = img_np.shape
    y0, y1, x0, x1 = roi_bounds(cy, cx, radius, h, w)
    patch = img_np[y0:y1, x0:x1]
    inner1 = img_np[max(cy - 1, 0): min(cy + 2, h), max(cx - 1, 0): min(cx + 2, w)]
    inner2 = img_np[max(cy - 2, 0): min(cy + 3, h), max(cx - 2, 0): min(cx + 3, w)]
    yy, xx = np.gradient(patch)
    grad_mag = np.sqrt(xx * xx + yy * yy)
    gx = np.abs(xx).mean()
    gy = np.abs(yy).mean()
    anisotropy = abs(gx - gy) / (gx + gy + 1e-6)
    lap = np.zeros_like(patch)
    if patch.shape[0] >= 3 and patch.shape[1] >= 3:
        lap[1:-1, 1:-1] = (
            patch[:-2, 1:-1] + patch[2:, 1:-1] + patch[1:-1, :-2] + patch[1:-1, 2:]
            - 4 * patch[1:-1, 1:-1]
        )
    patch_mean = float(patch.mean())
    patch_std = float(patch.std())
    patch_max = float(patch.max())
    patch_min = float(patch.min())
    inner1_mean = float(inner1.mean())
    inner2_mean = float(inner2.mean())
    surround_mean = float((patch.sum() - inner2.sum()) / max(patch.size - inner2.size, 1))
    return np.array([
        float(img_np[cy, cx]),
        patch_mean,
        patch_std,
        patch_max,
        patch_min,
        inner1_mean,
        inner2_mean,
        inner1_mean - surround_mean,
        inner2_mean - surround_mean,
        float(grad_mag.mean()),
        float(grad_mag.std()),
        float(gx),
        float(gy),
        float(anisotropy),
        float(np.mean(np.abs(lap))),
    ], dtype=np.float32)


def structure_feature_names():
    return [
        "center",
        "patch_mean",
        "patch_std",
        "patch_max",
        "patch_min",
        "inner3_mean",
        "inner5_mean",
        "center_surround_3",
        "center_surround_5",
        "grad_mean",
        "grad_std",
        "grad_x_abs_mean",
        "grad_y_abs_mean",
        "grad_anisotropy",
        "lap_abs_mean",
    ]


def build_matches(gt_regions, pred_regions, pred_label, gt_label, match_radius):
    pairs = []
    for gi, g in enumerate(gt_regions):
        gmask = np.zeros_like(gt_label, dtype=bool)
        gmask[g.coords[:, 0], g.coords[:, 1]] = True
        for pi, p in enumerate(pred_regions):
            overlap = int(np.logical_and(gmask, pred_label == p.label).sum())
            dist = float(np.linalg.norm(np.array(g.centroid) - np.array(p.centroid)))
            if overlap > 0 or dist <= match_radius:
                pairs.append((gi, pi, overlap, dist))
    pairs.sort(key=lambda item: (-item[2], item[3]))

    gt_to_pred = {i: None for i in range(len(gt_regions))}
    pred_to_gt = {i: None for i in range(len(pred_regions))}
    used_gt = set()
    used_pred = set()
    for gi, pi, _, _ in pairs:
        if gi in used_gt or pi in used_pred:
            continue
        gt_to_pred[gi] = pi
        pred_to_gt[pi] = gi
        used_gt.add(gi)
        used_pred.add(pi)
    return gt_to_pred, pred_to_gt


def background_mode_group(case_label):
    if case_label in {"tp_normal", "fn_entangled", "fn_weak"}:
        return "target"
    if case_label == "tn_easy":
        return "background_easy"
    return "background_negative_mode"


def mask_quantile(values, mask, quantile, fallback=math.inf):
    vals = values[mask]
    if vals.size == 0:
        return float(fallback)
    return float(np.quantile(vals, quantile))


def mine_peak_coords(score_map, mask, threshold_abs, min_distance, num_peaks):
    if mask.sum() == 0 or num_peaks <= 0:
        return np.zeros((0, 2), dtype=np.int64)
    masked_map = np.where(mask, score_map, -1e9)
    coords = peak_local_max(
        masked_map,
        min_distance=min_distance,
        threshold_abs=threshold_abs,
        num_peaks=num_peaks,
    )
    return coords.astype(np.int64)


def select_spaced_points(coords, max_points, min_distance, rng):
    if len(coords) == 0 or max_points <= 0:
        return np.zeros((0, 2), dtype=np.int64)
    coords = np.array(coords, dtype=np.int64)
    order = rng.permutation(len(coords))
    kept = []
    min_dist_sq = float(min_distance * min_distance)
    for idx in order:
        cy, cx = coords[idx]
        ok = True
        for py, px in kept:
            if (cy - py) ** 2 + (cx - px) ** 2 < min_dist_sq:
                ok = False
                break
        if ok:
            kept.append((int(cy), int(cx)))
            if len(kept) >= max_points:
                break
    return np.array(kept, dtype=np.int64)


def sample_easy_background_coords(score_map, prob_map, mask, quantile, prob_max, max_points, min_distance, rng):
    if mask.sum() == 0 or max_points <= 0:
        return np.zeros((0, 2), dtype=np.int64)
    thr = mask_quantile(score_map, mask, quantile, fallback=0.0)
    candidate_mask = mask & (score_map <= thr) & (prob_map <= prob_max)
    coords = np.argwhere(candidate_mask)
    return select_spaced_points(coords, max_points, min_distance, rng)


def append_background_case(
    case_rows,
    feature_store,
    case_counter,
    layer_maps,
    img_np,
    cy,
    cx,
    roi_radius,
    case_id,
    image_name,
    case_label,
    extra_fields=None,
):
    row = {
        "case_id": case_id,
        "image_name": image_name,
        "case_kind": "background_mode",
        "base_label": case_label,
        "background_mode": case_label,
        "case_label": case_label,
        "mode_group": background_mode_group(case_label),
        "cy": int(cy),
        "cx": int(cx),
        "area": 0,
        "matched_pred": -1,
    }
    if extra_fields:
        row.update(extra_fields)
    scalar_row, case_feats = extract_case_features(layer_maps, img_np, int(cy), int(cx), roi_radius)
    row.update(scalar_row)
    logits = layer_maps["logits"]
    prob = logits.sigmoid()
    row["logit_center"] = float(logits[0, cy, cx].item())
    row["prob_center"] = float(prob[0, cy, cx].item())
    row["priority_score"] = row.get("dec2_peak_value", row["prob_center"])
    case_rows.append(row)
    for name_key, arr in case_feats.items():
        feature_store[name_key].append(arr)
    feature_store["case_id"].append(case_id)
    feature_store["image_name"].append(image_name)
    feature_store["base_label"].append(case_label)
    case_counter[row["case_label"]] += 1


def extract_case_features(layer_maps, img_np, cy, cx, roi_radius):
    row = {}
    feature_arrays = {}
    for name, feat_map in layer_maps.items():
        mean_vec = patch_mean_vector(feat_map, cy, cx, roi_radius)
        center_vec = patch_center_vector(feat_map, cy, cx)
        feature_arrays[name] = mean_vec.detach().cpu().numpy().astype(np.float32)
        row[f"{name}_dim"] = int(mean_vec.shape[0])
        row[f"{name}_center_l2"] = float(torch.sqrt((center_vec * center_vec).sum() + 1e-12).item())
        row[f"{name}_roi_l2"] = float(torch.sqrt((mean_vec * mean_vec).sum() + 1e-12).item())
    feature_arrays["structure"] = compute_structure_features(img_np, cy, cx, roi_radius)
    return row, feature_arrays


def collect_example_maps(img_np, gt_np, pred_np, dec2_energy_np, head_input_np, logit_np):
    maps = {
        "image": img_np,
        "gt": gt_np.astype(np.float32),
        "pred": pred_np.astype(np.float32),
        "dec2_energy": dec2_energy_np,
        "logit": logit_np,
    }
    for ci in range(min(head_input_np.shape[0], 2)):
        maps[f"head_ch{ci + 1}"] = head_input_np[ci]
    return maps


def run_probe_task(name, X, y, folds, seed):
    result = {
        "task": name,
        "num_samples": int(len(y)),
        "num_pos": int(y.sum()),
        "num_neg": int((1 - y).sum()),
    }
    if len(np.unique(y)) < 2:
        result["status"] = "skipped_one_class"
        return result
    min_class = int(min(result["num_pos"], result["num_neg"]))
    if min_class < 2:
        result["status"] = "skipped_too_few"
        return result

    n_splits = min(folds, min_class)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    probs = np.zeros(len(y), dtype=np.float32)
    preds = np.zeros(len(y), dtype=np.int64)

    for train_idx, test_idx in cv.split(X, y):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=seed,
            solver="lbfgs",
        )
        clf.fit(X_train, y[train_idx])
        probs[test_idx] = clf.predict_proba(X_test)[:, 1]
        preds[test_idx] = clf.predict(X_test)

    result.update({
        "status": "ok",
        "roc_auc": float(roc_auc_score(y, probs)),
        "average_precision": float(average_precision_score(y, probs)),
        "balanced_accuracy": float(balanced_accuracy_score(y, preds)),
    })
    return result


def plot_head2_scatter(cases, head_features, weight, bias, out_png, out_csv, max_points, seed):
    rng = np.random.default_rng(seed)
    selected = [c for c in cases if c["case_label"] in CASE_COLORS]
    X = head_features[selected_indices(cases, selected)]
    labels = [c["case_label"] for c in selected]
    ids = [c["case_id"] for c in selected]
    names = [c["image_name"] for c in selected]

    if len(selected) > max_points:
        keep = rng.choice(len(selected), size=max_points, replace=False)
        keep.sort()
        X = X[keep]
        labels = [labels[i] for i in keep]
        ids = [ids[i] for i in keep]
        names = [names[i] for i in keep]

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["case_id", "image_name", "case_label", "x", "y", "margin"])
        for case_id, image_name, label, vec in zip(ids, names, labels, X):
            margin = float(weight[0] * vec[0] + weight[1] * vec[1] + bias)
            writer.writerow([case_id, image_name, label, float(vec[0]), float(vec[1]), margin])

    plt.figure(figsize=(8, 8))
    for label, color in CASE_COLORS.items():
        idx = [i for i, lab in enumerate(labels) if lab == label]
        if not idx:
            continue
        pts = X[idx]
        plt.scatter(pts[:, 0], pts[:, 1], s=18, alpha=0.65, c=color, label=label)

    xmin, xmax = float(X[:, 0].min()), float(X[:, 0].max())
    ymin, ymax = float(X[:, 1].min()), float(X[:, 1].max())
    padx = max((xmax - xmin) * 0.08, 1e-3)
    pady = max((ymax - ymin) * 0.08, 1e-3)
    plt.xlim(xmin - padx, xmax + padx)
    plt.ylim(ymin - pady, ymax + pady)

    if abs(weight[1]) > 1e-8:
        xs = np.array([xmin - padx, xmax + padx], dtype=np.float32)
        ys = -(weight[0] * xs + bias) / weight[1]
        plt.plot(xs, ys, "--", color="black", linewidth=1.5)

    plt.xlabel("Head channel 1")
    plt.ylabel("Head channel 2")
    plt.title("Head2 feature space with learned linear boundary")
    plt.grid(alpha=0.2)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def selected_indices(all_cases, subset_cases):
    index = {case["case_id"]: i for i, case in enumerate(all_cases)}
    return np.array([index[c["case_id"]] for c in subset_cases], dtype=np.int64)


def save_example_figures(cases, image_maps_by_name, out_dir, limit):
    ensure_dir(out_dir)
    grouped = defaultdict(list)
    for case in cases:
        grouped[case["case_label"]].append(case)

    for label, rows in grouped.items():
        rows = sorted(rows, key=lambda row: -row.get("priority_score", 0.0))[:limit]
        for row in rows:
            maps = image_maps_by_name.get(row["image_name"])
            if maps is None:
                continue
            fig, axes = plt.subplots(2, 3, figsize=(10, 6))
            items = [
                ("image", maps["image"]),
                ("gt", maps["gt"]),
                ("pred", maps["pred"]),
                ("dec2_energy", maps["dec2_energy"]),
                ("head_ch1", maps.get("head_ch1")),
                ("logit", maps["logit"]),
            ]
            for ax, (name, arr) in zip(axes.flat, items):
                if arr is None:
                    ax.axis("off")
                    continue
                cmap = "gray" if name in {"image", "gt", "pred"} else "magma"
                ax.imshow(arr, cmap=cmap)
                ax.set_title(name)
                ax.axis("off")
            fig.suptitle(f"{row['case_id']} | {label}", fontsize=10)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"{row['case_id']}.png"), dpi=160)
            plt.close(fig)


def main():
    opt = parse_args()
    seed_pytorch(opt.seed)
    rng = np.random.default_rng(opt.seed)
    ensure_dir(opt.output_dir)
    device = resolve_device(opt.device)

    ckpt = torch.load(opt.checkpoint, map_location=device, weights_only=False)
    model_name = ckpt["model_name"]
    dataset_name = opt.dataset_name or ckpt.get("dataset_name", "IRSTD-1K")
    loss_name = ckpt.get("loss_name", "BCE+SoftIoU")
    canonical_name = opt.canonical_name or model_name

    net = Net(model_name=model_name, loss_name=loss_name).to(device)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()

    test_set = TestSetLoader(opt.dataset_dir, dataset_name, dataset_name)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=opt.workers)

    case_rows = []
    feature_store = defaultdict(list)
    image_maps = {}
    gt_case_ids = []
    tp_dec2_evidence = []
    image_summaries = []
    case_counter = Counter()

    with torch.no_grad():
        for image_idx, (img, gt_mask, size, name) in enumerate(test_loader):
            if opt.limit_images > 0 and image_idx >= opt.limit_images:
                break
            image_name = name[0]
            h = int(size[0][0].item()) if isinstance(size[0], torch.Tensor) else int(size[0][0])
            w = int(size[1][0].item()) if isinstance(size[1], torch.Tensor) else int(size[1][0])

            img = img.to(device)
            gt_mask = gt_mask.to(device)
            feats = net.forward_features(img)

            logits = crop_map(feats["logits"], h, w)[0]
            dec1 = crop_map(feats["dec1"], h, w)[0]
            head_input = crop_map(feats["head_input"], h, w)[0]
            dec2 = upsample_to_input(feats["dec2"], h, w)[0]

            prob = logits.sigmoid()
            pred = (prob > opt.threshold).float()
            gt = crop_map(gt_mask, h, w)[0]

            img_np = crop_map(img, h, w)[0, 0].detach().cpu().numpy().astype(np.float32)
            gt_np = gt[0].detach().cpu().numpy().astype(np.uint8)
            pred_np = pred[0].detach().cpu().numpy().astype(np.uint8)
            logit_np = logits[0].detach().cpu().numpy().astype(np.float32)
            dec2_energy = feature_energy(dec2)
            dec2_energy_np = dec2_energy.detach().cpu().numpy().astype(np.float32)
            head_input_np = head_input.detach().cpu().numpy().astype(np.float32)

            gt_label = measure.label(gt_np, connectivity=2)
            pred_label = measure.label(pred_np, connectivity=2)
            gt_regions = measure.regionprops(gt_label)
            pred_regions = measure.regionprops(pred_label)
            gt_to_pred, pred_to_gt = build_matches(gt_regions, pred_regions, pred_label, gt_label, opt.match_radius)

            layer_maps = {
                "dec2": dec2,
                "dec1": dec1,
                "head_input": head_input,
                "logits": logits,
            }

            for gi, region in enumerate(gt_regions):
                cy, cx = int(round(region.centroid[0])), int(round(region.centroid[1]))
                case_id = f"{image_name}_gt{gi:02d}"
                base_label = "gt_tp" if gt_to_pred[gi] is not None else "gt_fn"
                priority = 0.0

                row = {
                    "case_id": case_id,
                    "image_name": image_name,
                    "case_kind": "gt_target",
                    "base_label": base_label,
                    "background_mode": "",
                    "case_label": "tp_normal" if base_label == "gt_tp" else "fn_pending",
                    "mode_group": "target",
                    "cy": cy,
                    "cx": cx,
                    "area": int(region.area),
                    "matched_pred": -1 if gt_to_pred[gi] is None else int(gt_to_pred[gi]),
                }
                scalar_row, case_feats = extract_case_features(layer_maps, img_np, cy, cx, opt.roi_radius)
                row.update(scalar_row)

                gt_region_mask = np.zeros_like(gt_np, dtype=bool)
                gt_region_mask[region.coords[:, 0], region.coords[:, 1]] = True
                gt_region_mask_t = torch.from_numpy(gt_region_mask).to(device)
                row["gt_logit_mean"] = masked_mean_scalar(logits[0], gt_region_mask_t)
                row["gt_prob_mean"] = masked_mean_scalar(prob[0], gt_region_mask_t)
                row["gt_dec2_energy_topk"] = masked_topk_mean(dec2_energy, gt_region_mask_t, opt.topk_pixels)
                row["gt_dec2_energy_mean"] = masked_mean_scalar(dec2_energy, gt_region_mask_t)
                row["logit_center"] = float(logits[0, cy, cx].item())
                row["prob_center"] = float(prob[0, cy, cx].item())
                priority = row["gt_dec2_energy_topk"]
                row["priority_score"] = priority

                case_rows.append(row)
                gt_case_ids.append(case_id)
                if base_label == "gt_tp":
                    tp_dec2_evidence.append(row["gt_dec2_energy_topk"])
                for name_key, arr in case_feats.items():
                    feature_store[name_key].append(arr)
                feature_store["case_id"].append(case_id)
                feature_store["image_name"].append(image_name)
                feature_store["base_label"].append(base_label)
                case_counter[row["case_label"]] += 1

            for pi, region in enumerate(pred_regions):
                if pred_to_gt[pi] is not None:
                    continue
                cy, cx = int(round(region.centroid[0])), int(round(region.centroid[1]))
                case_id = f"{image_name}_fp{pi:02d}"
                row = {
                    "case_id": case_id,
                    "image_name": image_name,
                    "case_kind": "pred_component",
                    "base_label": "fp_clutter",
                    "background_mode": "fp_clutter",
                    "case_label": "fp_clutter",
                    "mode_group": background_mode_group("fp_clutter"),
                    "cy": cy,
                    "cx": cx,
                    "area": int(region.area),
                    "matched_pred": int(pi),
                }
                scalar_row, case_feats = extract_case_features(layer_maps, img_np, cy, cx, opt.roi_radius)
                row.update(scalar_row)
                row["logit_center"] = float(logits[0, cy, cx].item())
                row["prob_center"] = float(prob[0, cy, cx].item())
                row["priority_score"] = row["prob_center"]

                case_rows.append(row)
                for name_key, arr in case_feats.items():
                    feature_store[name_key].append(arr)
                feature_store["case_id"].append(case_id)
                feature_store["image_name"].append(image_name)
                feature_store["base_label"].append("fp_clutter")
                case_counter[row["case_label"]] += 1

            background_mask = (gt_np == 0) & (pred_np == 0)
            gt_bool = gt_np.astype(bool)
            near_boundary_mask = binary_dilation(gt_bool, iterations=opt.tn_boundary_radius) & (~gt_bool) & background_mask
            far_background_mask = background_mask & (~near_boundary_mask)

            high_thr = mask_quantile(dec2_energy_np, far_background_mask, opt.tn_targetlike_quantile)
            candidate_peaks = mine_peak_coords(
                dec2_energy_np,
                far_background_mask,
                high_thr,
                opt.tn_peak_min_distance,
                max((opt.tn_targetlike_per_image + opt.tn_high_response_per_image) * 4, 8),
            )
            candidate_rows = []
            for peak_idx, (cy, cx) in enumerate(candidate_peaks):
                y0, y1, x0, x1 = roi_bounds(int(cy), int(cx), opt.roi_radius, h, w)
                if gt_np[y0:y1, x0:x1].any() or pred_np[y0:y1, x0:x1].any():
                    continue
                if logit_np[cy, cx] > 0:
                    continue
                candidate_rows.append({
                    "cy": int(cy),
                    "cx": int(cx),
                    "peak_rank": int(peak_idx),
                    "dec2_peak_value": float(dec2_energy_np[cy, cx]),
                    "logit_center": float(logit_np[cy, cx]),
                    "prob_center": float(prob[0, cy, cx].item()),
                })
            candidate_rows.sort(key=lambda row: (-row["prob_center"], -row["dec2_peak_value"]))

            kept_targetlike = 0
            kept_high = 0
            for cand in candidate_rows:
                if kept_targetlike < opt.tn_targetlike_per_image:
                    label = "tn_targetlike"
                    kept_targetlike += 1
                elif kept_high < opt.tn_high_response_per_image:
                    label = "tn_high_response"
                    kept_high += 1
                else:
                    break
                append_background_case(
                    case_rows,
                    feature_store,
                    case_counter,
                    layer_maps,
                    img_np,
                    cand["cy"],
                    cand["cx"],
                    opt.roi_radius,
                    f"{image_name}_{label}{(kept_targetlike if label == 'tn_targetlike' else kept_high) - 1:02d}",
                    image_name,
                    label,
                    {
                        "peak_rank": cand["peak_rank"],
                        "dec2_peak_value": cand["dec2_peak_value"],
                        "tn_quantile_threshold": high_thr,
                    },
                )

            boundary_thr = mask_quantile(dec2_energy_np, near_boundary_mask, opt.tn_boundary_quantile)
            boundary_peaks = mine_peak_coords(
                dec2_energy_np,
                near_boundary_mask,
                boundary_thr,
                opt.tn_peak_min_distance,
                max(opt.tn_near_boundary_per_image * 4, opt.tn_near_boundary_per_image),
            )
            kept_boundary = 0
            for peak_idx, (cy, cx) in enumerate(boundary_peaks):
                if kept_boundary >= opt.tn_near_boundary_per_image:
                    break
                if logit_np[cy, cx] > 0:
                    continue
                append_background_case(
                    case_rows,
                    feature_store,
                    case_counter,
                    layer_maps,
                    img_np,
                    int(cy),
                    int(cx),
                    opt.roi_radius,
                    f"{image_name}_tn_near_boundary{kept_boundary:02d}",
                    image_name,
                    "tn_near_boundary",
                    {
                        "peak_rank": int(peak_idx),
                        "dec2_peak_value": float(dec2_energy_np[cy, cx]),
                        "tn_boundary_threshold": boundary_thr,
                    },
                )
                kept_boundary += 1

            easy_coords = sample_easy_background_coords(
                dec2_energy_np,
                prob[0].detach().cpu().numpy().astype(np.float32),
                far_background_mask,
                opt.tn_easy_quantile,
                opt.tn_easy_prob_max,
                opt.tn_easy_per_image,
                opt.tn_peak_min_distance,
                rng,
            )
            kept_easy = 0
            for cy, cx in easy_coords:
                append_background_case(
                    case_rows,
                    feature_store,
                    case_counter,
                    layer_maps,
                    img_np,
                    int(cy),
                    int(cx),
                    opt.roi_radius,
                    f"{image_name}_tn_easy{kept_easy:02d}",
                    image_name,
                    "tn_easy",
                    {
                        "dec2_peak_value": float(dec2_energy_np[cy, cx]),
                        "tn_easy_quantile_threshold": mask_quantile(dec2_energy_np, far_background_mask, opt.tn_easy_quantile, fallback=0.0),
                    },
                )
                kept_easy += 1

            image_maps[image_name] = collect_example_maps(
                img_np, gt_np, pred_np, dec2_energy_np, head_input_np, logit_np
            )

            image_summaries.append({
                "image_name": image_name,
                "num_gt": len(gt_regions),
                "num_pred_components": len(pred_regions),
                "num_tn_targetlike": kept_targetlike,
                "num_tn_high_response": kept_high,
                "num_tn_near_boundary": kept_boundary,
                "num_tn_easy": kept_easy,
                "tn_high_threshold": high_thr,
                "tn_boundary_threshold": boundary_thr,
            })

    tp_ref = float(np.quantile(tp_dec2_evidence, opt.fn_ref_quantile)) if tp_dec2_evidence else 0.0
    for row in case_rows:
        if row["base_label"] != "gt_fn":
            continue
        if row["gt_dec2_energy_topk"] >= tp_ref and row["gt_logit_mean"] <= opt.fn_logit_max:
            row["case_label"] = "fn_entangled"
        else:
            row["case_label"] = "fn_weak"
        row["mode_group"] = background_mode_group(row["case_label"])

    case_counter = Counter(row["case_label"] for row in case_rows)

    case_index = {case_id: i for i, case_id in enumerate(feature_store["case_id"])}
    dec2_features = np.stack(feature_store["dec2"], axis=0)
    dec1_features = np.stack(feature_store["dec1"], axis=0)
    head_features = np.stack(feature_store["head_input"], axis=0)
    logit_features = np.stack(feature_store["logits"], axis=0)
    structure_features = np.stack(feature_store["structure"], axis=0)
    labels = np.array([row["case_label"] for row in case_rows], dtype=object)

    for row in case_rows:
        idx = case_index[row["case_id"]]
        row["head_margin"] = float(logit_features[idx][0])

    fieldnames = sorted({key for row in case_rows for key in row.keys()})
    with open(os.path.join(opt.output_dir, "cases.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(case_rows)

    np.savez_compressed(
        os.path.join(opt.output_dir, "features.npz"),
        case_id=np.array(feature_store["case_id"], dtype=object),
        image_name=np.array(feature_store["image_name"], dtype=object),
        base_label=np.array(feature_store["base_label"], dtype=object),
        case_label=labels,
        dec2=dec2_features,
        dec1=dec1_features,
        head_input=head_features,
        logits=logit_features,
        structure=structure_features,
        structure_names=np.array(structure_feature_names(), dtype=object),
    )

    tasks = []
    layer_sources = {
        "dec2": dec2_features,
        "dec1": dec1_features,
        "head_input": head_features,
        "structure": structure_features,
    }
    task_defs = [
        ("target_vs_shared_negative_modes", ["tp_normal", "fn_entangled", "fn_weak"],
         ["fp_clutter", "tn_targetlike", "tn_high_response", "tn_near_boundary"]),
        ("target_vs_tn_easy", ["tp_normal", "fn_entangled", "fn_weak"], ["tn_easy"]),
        ("fn_entangled_vs_shared_negative_modes", ["fn_entangled"],
         ["fp_clutter", "tn_targetlike", "tn_high_response", "tn_near_boundary"]),
        ("fn_entangled_vs_fp_clutter", ["fn_entangled"], ["fp_clutter"]),
        ("fn_entangled_vs_tn_targetlike", ["fn_entangled"], ["tn_targetlike"]),
        ("fn_entangled_vs_tn_high_response", ["fn_entangled"], ["tn_high_response"]),
        ("fn_entangled_vs_tn_near_boundary", ["fn_entangled"], ["tn_near_boundary"]),
        ("fn_entangled_vs_tn_easy", ["fn_entangled"], ["tn_easy"]),
        ("tp_normal_vs_fn_entangled", ["tp_normal"], ["fn_entangled"]),
    ]
    for layer_name, X_full in layer_sources.items():
        for task_name, pos_labels, neg_labels in task_defs:
            idx = [
                i for i, lab in enumerate(labels)
                if lab in pos_labels or lab in neg_labels
            ]
            if not idx:
                continue
            X = X_full[idx]
            y = np.array([1 if labels[i] in pos_labels else 0 for i in idx], dtype=np.int64)
            task = run_probe_task(f"{layer_name}:{task_name}", X, y, opt.probe_folds, opt.seed)
            task["layer"] = layer_name
            task["task_name"] = task_name
            tasks.append(task)

    with open(os.path.join(opt.output_dir, "layerwise_probe.json"), "w") as f:
        json.dump(to_jsonable(tasks), f, indent=2)

    with open(os.path.join(opt.output_dir, "layerwise_probe.csv"), "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["layer", "task_name", "status", "num_samples", "num_pos", "num_neg",
                        "roc_auc", "average_precision", "balanced_accuracy"],
        )
        writer.writeheader()
        for task in tasks:
            writer.writerow({k: task.get(k) for k in writer.fieldnames})

    channel_semantics = {
        "head_dim": int(head_features.shape[1]),
        "structure_feature_names": structure_feature_names(),
        "case_stats": {},
    }
    if head_features.shape[1] == 2:
        head_weight = net.model.head.weight.detach().cpu().numpy()[0, :, 0, 0].astype(np.float32)
        head_bias = float(net.model.head.bias.detach().cpu().numpy()[0])
        channel_semantics["linear_head"] = {
            "weight": head_weight.tolist(),
            "bias": head_bias,
        }
        for label in sorted(CASE_COLORS):
            idx = np.where(labels == label)[0]
            if len(idx) == 0:
                continue
            mean_feat = head_features[idx].mean(axis=0)
            channel_semantics["case_stats"][label] = {
                "count": int(len(idx)),
                "mean_head_feature": mean_feat.tolist(),
                "std_head_feature": head_features[idx].std(axis=0).tolist(),
                "mean_contribution": (head_weight * mean_feat).tolist(),
                "mean_margin": float(np.mean(head_features[idx] @ head_weight + head_bias)),
            }
        plot_head2_scatter(
            case_rows,
            head_features,
            head_weight,
            head_bias,
            os.path.join(opt.output_dir, "head2_scatter.png"),
            os.path.join(opt.output_dir, "head2_scatter.csv"),
            opt.max_scatter_points,
            opt.seed,
        )

    with open(os.path.join(opt.output_dir, "channel_semantics.json"), "w") as f:
        json.dump(to_jsonable(channel_semantics), f, indent=2)

    save_example_figures(case_rows, image_maps, os.path.join(opt.output_dir, "example_cases"), opt.save_examples)

    summary = {
        "checkpoint": opt.checkpoint,
        "model_name": model_name,
        "canonical_name": canonical_name,
        "dataset_name": dataset_name,
        "device": str(device),
        "num_images": len(image_summaries),
        "counts": dict(case_counter),
        "thresholds": {
            "fn_entangled_tp_dec2_reference_quantile": opt.fn_ref_quantile,
            "fn_entangled_tp_dec2_reference_value": tp_ref,
            "fn_entangled_logit_max": opt.fn_logit_max,
            "tn_targetlike_quantile": opt.tn_targetlike_quantile,
            "tn_boundary_quantile": opt.tn_boundary_quantile,
            "tn_boundary_radius": opt.tn_boundary_radius,
            "tn_easy_quantile": opt.tn_easy_quantile,
            "tn_easy_prob_max": opt.tn_easy_prob_max,
        },
        "background_mode_labels": [
            "fp_clutter",
            "tn_targetlike",
            "tn_high_response",
            "tn_near_boundary",
            "tn_easy",
        ],
        "image_summaries": image_summaries,
    }
    with open(os.path.join(opt.output_dir, "summary.json"), "w") as f:
        json.dump(to_jsonable(summary), f, indent=2)

    print(json.dumps(to_jsonable(summary), indent=2))


if __name__ == "__main__":
    main()
