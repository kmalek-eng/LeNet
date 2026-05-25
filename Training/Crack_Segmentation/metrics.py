import numpy as np


def _safe_divide(numerator, denominator):
    if denominator == 0:
        return 1.0
    return numerator / denominator


def f1_score(groundtruth_mask, pred_mask):
    gt = np.asarray(groundtruth_mask).astype(bool)
    pred = np.asarray(pred_mask).astype(bool)

    intersection = np.logical_and(gt, pred).sum()
    total = gt.sum() + pred.sum()

    return round(_safe_divide(2 * intersection, total), 3)


def iou_score(groundtruth_mask, pred_mask):
    gt = np.asarray(groundtruth_mask).astype(bool)
    pred = np.asarray(pred_mask).astype(bool)

    intersection = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()

    return round(_safe_divide(intersection, union), 3)