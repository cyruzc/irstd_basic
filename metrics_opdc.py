"""OPDC matching + hierarchical IoU (hIoU) for IRSTD evaluation.

Reference
---------
Youwei Pang, Xiaoqi Zhao, Lihe Zhang, Huchuan Lu, Georges El Fakhri,
Xiaofeng Liu, Shijian Lu.
"Rethinking Evaluation of Infrared Small Target Detection."
NeurIPS 2025 (Datasets & Benchmarks).  arXiv:2509.16888
Official toolkit: https://github.com/lartpang/PyIRSTDMetrics

What this module adds on top of ``metrics.py``
---------------------------------------------
``metrics.py`` keeps the two classic protocols:
    - ``PD_FA``  : centroid-distance matching (``distance < 3``), greedy, order dependent
                   -> Pd / Fa
    - ``IoU()`` / ``nIoU()`` : pure pixel-level, blind to target identity

This module implements the paper's two fixes:

1. **OPDC** ("Overlap Priority with Distance Compensation") — repairs the *matching
   primitive* that Pd/Fa depend on:
       stage 1: Hungarian assignment on centroid distance, keep pairs with IoU >= 0.5
       stage 2: on the leftovers only, Hungarian assignment, keep pairs with dist < 3
   IoU is scale-invariant (fixes "large target + few-pixel shift = total miss"),
   the distance stage rescues tiny targets whose IoU is inherently low, and the
   Hungarian assignment makes the result order-invariant (the old greedy loop is not).

2. **hIoU** — repairs the *score* after matching::

       hIoU = IoU_loc * IoU_seg
            = [TP / (TP + FP + FN)] * [sum(pair IoU) / TP]
            = sum(pair IoU) / (TP + FP + FN)          <-- TP cancels out

   i.e. "target-level IoU in which a matched pair contributes its *pixel* IoU
   instead of 1". A single number that simultaneously penalises missed targets
   (FN), spurious targets (FP) and imprecise masks (numerator). It closes the
   blind spot of Pd/Fa: a matched prediction with a bloated mask gives
   Pd = 1, Fa = 0, but hIoU < 1.

Notes
-----
* Everything here is CPU / numpy; it is meant for *post-training* evaluation
  (connected components are slow), matching the design of ``metrics.py``.
* Convention: ``pred`` and ``gt`` are **binary** arrays of the same shape,
  already cropped to the original image size. Thresholding is the caller's job,
  so threshold-policy experiments stay possible.
* ``OPDCMatching(penalize_invalid=True)`` switches stage 1+2 to a single
  Hungarian solve with legal pairs only. It is *not* what the official toolkit
  does (which assigns first and filters afterwards and can therefore lose a
  legal pair); it is provided so the difference can be measured.
* ``python metrics_opdc.py`` runs a built-in self test.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage import measure

__all__ = [
    "OPDCMatching",
    "DistanceOnlyMatching",
    "MatchingMetrics",
    "HierarchicalErrorAnalysis",
    "label_targets",
    "pair_geometry",
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def label_targets(binary: np.ndarray) -> tuple[np.ndarray, list]:
    """8-connected connected-component labelling (same as the official toolkit)."""
    lab = measure.label(np.asarray(binary).astype(np.uint8), connectivity=2)
    return lab, measure.regionprops(lab)


def pair_geometry(gt_map, gt_props, pred_map, pred_props, image_area: int):
    """Centroid distance and pixel IoU for every (GT, prediction) pair.

    Returns
    -------
    dist : (M, N) float array, centroid Euclidean distance.
           Unmatchable / empty cases are filled with ``image_area`` (a value that
           is larger than any possible centroid distance in the image).
    iou  : (M, N) float array, IoU of the two components, 0 when bboxes do not
           even intersect (cheap pruning).
    """
    m, n = len(gt_props), len(pred_props)
    dist = np.full((max(m, 1), max(n, 1)), float(image_area))
    iou = np.zeros((max(m, 1), max(n, 1)), dtype=np.float64)
    if m == 0 or n == 0:
        return dist[:m, :n], iou[:m, :n]

    gt_c = np.asarray([p.centroid for p in gt_props], dtype=np.float64)
    pr_c = np.asarray([p.centroid for p in pred_props], dtype=np.float64)
    dist[:m, :n] = np.linalg.norm(gt_c[:, None, :] - pr_c[None, :, :], axis=2)

    for i, g in enumerate(gt_props):
        gr0, gc0, gr1, gc1 = g.bbox
        for j, p in enumerate(pred_props):
            pr0, pc0, pr1, pc1 = p.bbox
            r0, c0 = max(gr0, pr0), max(gc0, pc0)
            r1, c1 = min(gr1, pr1), min(gc1, pc1)
            if r0 >= r1 or c0 >= c1:  # bbox disjoint -> IoU is 0, skip pixels
                continue
            inter = np.count_nonzero(
                (gt_map[r0:r1, c0:c1] == g.label) & (pred_map[r0:r1, c0:c1] == p.label)
            )
            if inter:
                iou[i, j] = inter / (g.area + p.area - inter)
    return dist, iou


def _assert_one_to_one(matched: np.ndarray) -> None:
    if not np.isin(matched.sum(axis=0), (0, 1)).all():
        raise ValueError("Some predicted targets are matched to multiple GT targets.")
    if not np.isin(matched.sum(axis=1), (0, 1)).all():
        raise ValueError("Some GT targets are matched to multiple predicted targets.")


# --------------------------------------------------------------------------- #
# matching strategies
# --------------------------------------------------------------------------- #
class DistanceOnlyMatching:
    """The classic protocol: greedy centroid-distance matching (``distance < 3``).

    Kept for A/B comparison. It is greedy over the label enumeration order, so
    its result is not order invariant.
    """

    def __init__(self, distance_threshold: int = 3):
        if not isinstance(distance_threshold, int) or distance_threshold <= 0:
            raise ValueError("distance_threshold must be a positive int")
        self.distance_threshold = distance_threshold

    def __call__(self, gt_map, gt_props, pred_map, pred_props, image_area: int):
        m, n = len(gt_props), len(pred_props)
        matched = np.zeros((m, n), dtype=bool)
        for i, g in enumerate(gt_props):
            for j, p in enumerate(pred_props):
                if matched[:, j].any():
                    continue
                if np.linalg.norm(np.subtract(g.centroid, p.centroid)) < self.distance_threshold:
                    matched[i, j] = True
                    break
        _, iou = pair_geometry(gt_map, gt_props, pred_map, pred_props, image_area)
        return matched, iou


class OPDCMatching:
    """Overlap Priority with Distance Compensation.

    Parameters
    ----------
    overlap_threshold : float
        Stage-1 acceptance rule, ``pair IoU >= overlap_threshold``.
    distance_threshold : int
        Stage-2 acceptance rule, ``centroid distance < distance_threshold``.
    penalize_invalid : bool
        If True, solve one Hungarian problem on the legal pairs only instead of
        "assign on everything, then filter". Gives the same or one more TP;
        differs from the official implementation on ~0.2% of random cases.
    """

    def __init__(
        self,
        overlap_threshold: float = 0.5,
        distance_threshold: int = 3,
        penalize_invalid: bool = False,
    ):
        if not 0 <= overlap_threshold < 1:
            raise ValueError("overlap_threshold must be in [0, 1)")
        if not isinstance(distance_threshold, int) or distance_threshold <= 0:
            raise ValueError("distance_threshold must be a positive int")
        self.overlap_threshold = overlap_threshold
        self.distance_threshold = distance_threshold
        self.penalize_invalid = penalize_invalid

    # -- used by both MatchingMetrics and HierarchicalErrorAnalysis ---------- #
    def legal_pairs(self, dist: np.ndarray, iou: np.ndarray):
        valid_iou = iou >= self.overlap_threshold
        valid_dist = dist < self.distance_threshold
        return valid_iou, valid_dist

    def match(self, dist: np.ndarray, iou: np.ndarray) -> np.ndarray:
        m, n = dist.shape
        matched = np.zeros((m, n), dtype=bool)
        if m == 0 or n == 0:
            return matched
        valid_iou, valid_dist = self.legal_pairs(dist, iou)

        if self.penalize_invalid:
            valid = valid_iou | valid_dist
            if not valid.any():
                return matched
            # any legal pair is worth more than all distances together
            big = (min(m, n) + 1) * (float(dist.max()) + 1.0) + 1.0
            rows, cols = linear_sum_assignment(np.where(valid, dist, big))
            keep = valid[rows, cols]
            matched[rows[keep], cols[keep]] = True
            return matched

        # stage 1: overlap priority (official behaviour: assign, then filter)
        if valid_iou.any():
            rows, cols = linear_sum_assignment(dist)
            matched[rows, cols] = True
            matched &= valid_iou

        # stage 2: distance compensation, leftovers only
        if valid_dist.any():
            free_gt = np.flatnonzero(matched.sum(axis=1) == 0)
            free_pred = np.flatnonzero(matched.sum(axis=0) == 0)
            if free_gt.size and free_pred.size:
                sub = dist[np.ix_(free_gt, free_pred)]
                rows, cols = linear_sum_assignment(sub)
                for r, c in zip(rows, cols):
                    if valid_dist[free_gt[r], free_pred[c]]:
                        matched[free_gt[r], free_pred[c]] = True
        return matched

    def __call__(self, gt_map, gt_props, pred_map, pred_props, image_area: int):
        dist, iou = pair_geometry(gt_map, gt_props, pred_map, pred_props, image_area)
        return self.match(dist, iou), iou


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
class MatchingMetrics:
    """Target-level Pd / Fa plus hierarchical hIoU.

    ``update`` consumes binary masks; ``get`` returns::

        pd       = TP / (TP + FN)                     (target level)
        fa       = FP_area / total_area               (pixel level, as in the field)
        seg_iou  = sum(pair IoU) / TP                 (quality of matched targets)
        loc_iou  = TP / (TP + FP + FN)                (did we find the targets)
        hiou     = seg_iou * loc_iou
                 = sum(pair IoU) / (TP + FP + FN)     (TP cancels out)

    Counters are accumulated over the whole dataset, so `IoU`-style global
    aggregation is used (consistent with ``metrics.py``).
    """

    def __init__(
        self,
        overlap_threshold: float = 0.5,
        distance_threshold: int = 3,
        penalize_invalid: bool = False,
        matcher=None,
    ):
        self.matcher = matcher or OPDCMatching(
            overlap_threshold=overlap_threshold,
            distance_threshold=distance_threshold,
            penalize_invalid=penalize_invalid,
        )
        self.reset()

    def reset(self):
        self.tp = 0            # matched GT/prediction pairs
        self.fp = 0            # unmatched predicted components
        self.fn = 0            # unmatched GT components
        self.tp_iou = 0.0      # sum of pair IoU over matched pairs
        self.fp_area = 0       # pixels in unmatched predicted components
        self.total_area = 0
        self.n_images = 0
        # per-image bookkeeping, handy for threshold-policy analysis
        self.per_image: list[dict] = []

    def update(self, pred: np.ndarray, gt: np.ndarray, name: str | None = None):
        pred = np.asarray(pred).astype(bool)
        gt = np.asarray(gt).astype(bool)
        if pred.shape != gt.shape:
            raise ValueError(f"shape mismatch: pred {pred.shape} vs gt {gt.shape}")

        gt_map, gt_props = label_targets(gt)
        pred_map, pred_props = label_targets(pred)
        area = int(pred.size)

        matched, iou = self.matcher(gt_map, gt_props, pred_map, pred_props, area)
        _assert_one_to_one(matched)

        tp = int(matched.sum())
        tp_iou = float(iou[matched].sum()) if tp else 0.0
        pred_matched = matched.sum(axis=0).astype(bool)
        fp = int((~pred_matched).sum())
        fn = int(gt_props.__len__() - tp)
        fp_area = int(sum(p.area for j, p in enumerate(pred_props) if not pred_matched[j]))

        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.tp_iou += tp_iou
        self.fp_area += fp_area
        self.total_area += area
        self.n_images += 1
        self.per_image.append(dict(
            name=name, tp=tp, fp=fp, fn=fn, tp_iou=tp_iou,
            fp_area=fp_area, area=area,
            n_gt=len(gt_props), n_pred=len(pred_props),
        ))

    def counts(self) -> dict:
        seg = self.tp_iou / self.tp if self.tp else 0.0
        loc = self.tp / (self.tp + self.fp + self.fn) if (self.tp + self.fp + self.fn) else 0.0
        return dict(
            tp=self.tp, fp=self.fp, fn=self.fn,
            tp_iou=self.tp_iou, fp_area=self.fp_area, total_area=self.total_area,
            pd=self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0,
            fa=self.fp_area / self.total_area if self.total_area else 0.0,
            seg_iou=seg,
            loc_iou=loc,
            hiou=seg * loc,
        )

    def get(self) -> dict:
        return self.counts()


class HierarchicalErrorAnalysis:
    """Decompose the loss of hIoU into 7 interpretable error types.

    Target level (sums to ``1 - loc_iou``)::

        loc_s2m_err : one prediction satisfied several GT targets, extras lost
        loc_m2s_err : several predictions competed for one GT target
        loc_itf_err : false targets from clutter / noise
        loc_pcp_err : real targets not perceived

    Pixel level (sums to ``1 - seg_iou``), only over matched pairs::

        seg_mrg_err : predicted pixels that leaked into a *neighbouring* GT target
        seg_itf_err : predicted pixels outside any GT target (near the target)
        seg_pcp_err : GT pixels inside the matched target that were missed
    """

    def __init__(self, overlap_threshold: float = 0.5, distance_threshold: int = 3,
                 penalize_invalid: bool = False):
        self.matching = OPDCMatching(overlap_threshold, distance_threshold, penalize_invalid)
        self.reset()

    def reset(self):
        self.tp = self.fp = self.fn = 0
        self.tp_iou = 0.0
        self.seg_mrg = self.seg_itf = self.seg_pcp = 0.0
        self.n_itf = self.n_pcp = self.n_m2s = self.n_s2m = 0

    def update(self, pred: np.ndarray, gt: np.ndarray):
        pred = np.asarray(pred).astype(bool)
        gt = np.asarray(gt).astype(bool)
        gt_map, gt_props = label_targets(gt)
        pred_map, pred_props = label_targets(pred)
        area = int(pred.size)

        dist, iou = pair_geometry(gt_map, gt_props, pred_map, pred_props, area)
        valid_iou, valid_dist = self.matching.legal_pairs(dist, iou)
        initial = valid_iou | valid_dist          # pairs that *could* have matched
        matched = self.matching.match(dist, iou)
        _assert_one_to_one(matched)

        for i, j in zip(*np.nonzero(matched)):
            g_mask = gt_map == gt_props[i].label
            p_mask = pred_map == pred_props[j].label
            other_gt = gt & ~g_mask

            tp_pix = np.count_nonzero(p_mask & g_mask)
            fp_pix = np.count_nonzero(p_mask & ~g_mask)
            fp_other = np.count_nonzero(p_mask & other_gt)
            fp_self = fp_pix - fp_other
            fn_pix = np.count_nonzero(~p_mask & g_mask)
            union = np.count_nonzero(p_mask | g_mask)

            self.tp_iou += tp_pix / union if union else 0.0
            self.seg_mrg += fp_other / union if union else 0.0
            self.seg_itf += fp_self / union if union else 0.0
            self.seg_pcp += fn_pix / union if union else 0.0

        init_free_gt = int((initial.sum(axis=1) == 0).sum())
        init_free_pred = int((initial.sum(axis=0) == 0).sum())
        final_free_gt = int((matched.sum(axis=1) == 0).sum())
        final_free_pred = int((matched.sum(axis=0) == 0).sum())

        self.n_itf += init_free_pred
        self.n_pcp += init_free_gt
        self.n_m2s += final_free_pred - init_free_pred
        self.n_s2m += final_free_gt - init_free_gt

        self.tp += int(matched.sum())
        self.fp += final_free_pred
        self.fn += final_free_gt

    def get(self) -> dict:
        tp = self.tp
        denom = self.tp + self.fp + self.fn
        seg_iou = self.tp_iou / tp if tp else 0.0
        loc_iou = tp / denom if denom else 0.0

        def frac(x):
            return x / tp if tp else 0.0

        def tfrac(x):
            return x / denom if denom else 0.0

        return dict(
            seg_iou=seg_iou,
            seg_mrg_err=frac(self.seg_mrg),
            seg_itf_err=frac(self.seg_itf),
            seg_pcp_err=frac(self.seg_pcp),
            loc_iou=loc_iou,
            loc_s2m_err=tfrac(self.n_s2m),
            loc_m2s_err=tfrac(self.n_m2s),
            loc_itf_err=tfrac(self.n_itf),
            loc_pcp_err=tfrac(self.n_pcp),
            hiou=seg_iou * loc_iou,
        )


# --------------------------------------------------------------------------- #
# self test
# --------------------------------------------------------------------------- #
def _box(rows, cols, h, w, size=64):
    m = np.zeros((size, size), dtype=bool)
    m[rows:rows + h, cols:cols + w] = True
    return m


def _self_test() -> None:
    def run(gt, pred, matcher):
        gm, gp = label_targets(gt)
        pm, pp = label_targets(pred)
        matched, iou = matcher(gm, gp, pm, pp, gt.size)
        return int(matched.sum()), iou[matched] if matched.any() else np.array([])

    print("=" * 72)
    print("case A: 20x20 target shifted by 3 px  (pixel IoU = 0.739)")
    gt, pr = _box(20, 20, 20, 20), _box(20, 23, 20, 20)
    tp_old, _ = run(gt, pr, DistanceOnlyMatching(3))
    tp_new, ious = run(gt, pr, OPDCMatching())
    print(f"  distance-only TP={tp_old}/1   OPDC TP={tp_new}/1 (matched IoU={ious[0]:.3f})")
    assert (tp_old, tp_new) == (0, 1)

    print("case B: 2x2 target shifted by 1 px  (pixel IoU = 0.333)")
    gt, pr = _box(20, 20, 2, 2), _box(20, 21, 2, 2)
    tp_overlap, _ = run(gt, pr, OPDCMatching(distance_threshold=1))  # overlap stage only
    tp_opdc, _ = run(gt, pr, OPDCMatching())
    print(f"  overlap-only TP={tp_overlap}/1   OPDC TP={tp_opdc}/1")
    assert (tp_overlap, tp_opdc) == (0, 1)

    print("case C: greedy is order dependent, Hungarian is not")
    gt = _box(10, 10, 1, 1) | _box(10, 13, 1, 1)
    pr = _box(10, 11, 1, 1) | _box(12, 9, 1, 1)
    tp_greedy, _ = run(gt, pr, DistanceOnlyMatching(3))
    tp_hung, _ = run(gt, pr, OPDCMatching())
    print(f"  greedy TP={tp_greedy}/2   OPDC TP={tp_hung}/2")
    assert (tp_greedy, tp_hung) == (1, 2)

    print("case D: matched-but-blobby -> Pd=1, Fa=0, but hIoU=0.444")
    gt, pr = _box(20, 20, 4, 4), _box(19, 19, 6, 6)
    m = MatchingMetrics()
    m.update(pr, gt)
    c = m.get()
    print(f"  Pd={c['pd']:.3f}  Fa={c['fa']:.2e}  seg={c['seg_iou']:.3f}  "
          f"loc={c['loc_iou']:.3f}  hIoU={c['hiou']:.3f}")
    assert abs(c["pd"] - 1.0) < 1e-12 and abs(c["fa"]) < 1e-12
    assert abs(c["hiou"] - 16 / 36) < 1e-6

    print("case E: closed form  hIoU = sum(pair IoU) / (TP+FP+FN)")
    cases = {
        "perfect":    (_box(10, 10, 4, 4), _box(10, 10, 4, 4), 1.0000),
        "too big":    (_box(10, 10, 4, 4), _box(9, 9, 6, 6), 0.4444),
        "false alarm":(_box(10, 10, 4, 4), _box(10, 10, 4, 4) | _box(40, 40, 3, 3), 0.5000),
        "missed one": (_box(10, 10, 4, 4) | _box(40, 40, 4, 4), _box(10, 10, 4, 4), 0.5000),
        "two merged": (_box(10, 10, 4, 4) | _box(10, 15, 4, 4), _box(10, 10, 4, 9), 0.2222),
    }
    for tag, (gt, pr, want) in cases.items():
        m = MatchingMetrics()
        m.update(pr, gt)
        closed = m.tp_iou / (m.tp + m.fp + m.fn) if (m.tp + m.fp + m.fn) else 0.0
        c = m.get()
        print(f"  {tag:<12} TP/FP/FN={m.tp}/{m.fp}/{m.fn}  hIoU={c['hiou']:.4f} "
              f"closed={closed:.4f} (expect {want:.4f})")
        assert abs(c["hiou"] - closed) < 1e-12
        assert abs(c["hiou"] - want) < 1e-4

    print("case F: 1 - loc_iou == sum of target-level errors")
    gt = _box(10, 10, 4, 4) | _box(10, 20, 4, 4) | _box(40, 40, 4, 4)
    pr = _box(10, 10, 5, 5) | _box(40, 40, 4, 4) | _box(50, 5, 2, 2)
    ea = HierarchicalErrorAnalysis()
    ea.update(pr, gt)
    e = ea.get()
    loc_sum = e["loc_s2m_err"] + e["loc_m2s_err"] + e["loc_itf_err"] + e["loc_pcp_err"]
    seg_sum = e["seg_mrg_err"] + e["seg_itf_err"] + e["seg_pcp_err"]
    print(f"  loc_iou={e['loc_iou']:.4f}  sum(loc_err)={loc_sum:.4f}  (1-loc)={1 - e['loc_iou']:.4f}")
    print(f"  seg_iou={e['seg_iou']:.4f}  sum(seg_err)={seg_sum:.4f}  (1-seg)={1 - e['seg_iou']:.4f}")
    assert abs(loc_sum - (1 - e["loc_iou"])) < 1e-9
    assert abs(seg_sum - (1 - e["seg_iou"])) < 1e-9

    print("=" * 72)
    print("all self tests passed")


if __name__ == "__main__":
    _self_test()
