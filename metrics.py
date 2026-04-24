"""Evaluation metrics for IR small target segmentation.

GPU metrics (fast, used every epoch):
    - IoU:   global Intersection-over-Union (accumulate all pixels)
    - nIoU:  per-sample IoU then average (each image weighted equally)

CPU metrics (slow, used once after training):
    - PD_FA: Probability of Detection / False Alarm (skimage connected components)
"""

import numpy as np
import torch
from skimage import measure


class IoU:
    """Global IoU — accumulate intersection & union across all samples, pure GPU."""

    def __init__(self):
        self.reset()

    def update(self, preds, labels):
        # preds, labels: [B, 1, H, W], already binarized by caller
        predict = preds.float()
        target = labels.float()

        inter = (predict * target).sum()
        self.total_inter += inter
        self.total_union += predict.sum() + target.sum() - inter

    def get(self):
        eps = 1e-10
        return float(self.total_inter / (eps + self.total_union))

    def reset(self):
        self.total_inter = 0.0
        self.total_union = 0.0


class nIoU:
    """Normalized IoU — per-sample IoU then average, pure GPU (vectorized)."""

    def __init__(self):
        self.reset()

    def update(self, preds, labels):
        # preds, labels: [B, 1, H, W], already binarized by caller
        predict = preds.float()
        target = labels.float()

        inter = (predict * target).sum(dim=(1, 2, 3))       # [B]
        union = predict.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - inter  # [B]

        # empty pred + empty gt → perfect match → IoU = 1.0
        both_empty = (union == 0)
        ious = torch.where(both_empty, torch.ones_like(inter), inter / (union + 1e-10))

        self.total_iou += ious.sum().item()
        self.count += ious.shape[0]

    def get(self):
        return float(self.total_iou / max(self.count, 1))

    def reset(self):
        self.total_iou = 0.0
        self.count = 0


class PD_FA:
    """PD / FA — requires CPU for skimage connected component analysis."""

    def __init__(self):
        self.reset()

    def update(self, pred, label, size):
        predits = np.array(pred.cpu()).astype("int64")
        labelss = np.array(label.cpu()).astype("int64")

        image = measure.label(predits, connectivity=2)
        coord_image = measure.regionprops(image)
        label_img = measure.label(labelss, connectivity=2)
        coord_label = measure.regionprops(label_img)

        self.target += len(coord_label)

        distance_match = []
        true_img = np.zeros(predits.shape)

        for i in range(len(coord_label)):
            centroid_label = np.array(list(coord_label[i].centroid))
            for m in range(len(coord_image)):
                centroid_image = np.array(list(coord_image[m].centroid))
                distance = np.linalg.norm(centroid_image - centroid_label)
                if distance < 3:
                    distance_match.append(distance)
                    true_img[coord_image[m].coords[:, 0], coord_image[m].coords[:, 1]] = 1
                    del coord_image[m]
                    break

        self.dismatch_pixel += (predits - true_img).sum()
        self.all_pixel += size[0] * size[1]
        self.PD += len(distance_match)

    def get(self):
        fa = self.dismatch_pixel / max(self.all_pixel, 1)
        pd = self.PD / max(self.target, 1)
        return float(pd), float(fa)

    def reset(self):
        self.dismatch_pixel = 0
        self.all_pixel = 0
        self.PD = 0
        self.target = 0
