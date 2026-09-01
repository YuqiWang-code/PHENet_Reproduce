"""Binary change-detection metrics accumulated through a confusion matrix."""

import numpy as np


class Evaluator:
    def __init__(self):
        self.reset()

    def reset(self):
        # Rows are ground truth, columns are prediction.
        self.confusion_matrix = np.zeros((2, 2), dtype=np.int64)

    def add_batch(self, target, prediction):
        target = np.asarray(target).astype(np.int64, copy=False)
        prediction = np.asarray(prediction).astype(np.int64, copy=False)
        if target.shape != prediction.shape:
            raise ValueError(f"Metric shape mismatch: {target.shape} != {prediction.shape}")
        valid = (target >= 0) & (target < 2)
        encoded = 2 * target[valid] + prediction[valid]
        self.confusion_matrix += np.bincount(encoded, minlength=4).reshape(2, 2)

    def compute(self):
        tn, fp, fn, tp = self.confusion_matrix.ravel().astype(np.float64)
        eps = np.finfo(np.float64).eps
        recall = tp / max(tp + fn, eps)
        precision = tp / max(tp + fp, eps)
        oa = (tp + tn) / max(tp + tn + fp + fn, eps)
        f1 = 2.0 * precision * recall / max(precision + recall, eps)
        iou = tp / max(tp + fp + fn, eps)
        total = tp + tn + fp + fn
        pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / max(total * total, eps)
        kappa = (oa - pe) / max(1.0 - pe, eps)
        return {
            "Recall": recall,
            "Precision": precision,
            "OA": oa,
            "F1": f1,
            "IoU": iou,
            "Kappa": kappa,
        }
