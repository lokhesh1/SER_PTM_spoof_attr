#!/usr/bin/env python3
"""Evaluation metrics & result saving for layer-wise probing.

Part of the *spoof speech source attribution* project. Given softmax
probabilities and integer ground-truth labels, computes:

    * overall accuracy,
    * per-class accuracy (recall),
    * 7x7 confusion matrix,
    * binary EER (bonafide vs spoof) -- the classic ASVspoof number,
    * macro one-vs-rest EER (mean of per-class OvR EERs) + per-class table,

and saves them to a directory as ``metrics.json``, ``confusion_matrix.{npy,csv,
png}`` and ``per_class.csv``.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from feature_dataset import BONAFIDE_INDEX

logger = logging.getLogger("ser.probe_metrics")


# --------------------------------------------------------------------------- #
# Core metrics
# --------------------------------------------------------------------------- #
def confusion(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """``num_classes`` x ``num_classes`` confusion matrix (rows=true, cols=pred)."""
    from sklearn.metrics import confusion_matrix

    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))


def per_class_accuracy(cm: np.ndarray) -> np.ndarray:
    """Per-class recall = diagonal / row-sum (0 for classes with no support)."""
    rows = cm.sum(axis=1)
    return np.where(rows > 0, np.diag(cm) / np.maximum(rows, 1), 0.0)


def compute_eer(scores: np.ndarray, labels: np.ndarray):
    """Equal Error Rate for a binary problem. ``scores`` higher => more positive;
    ``labels`` in {0,1}. Returns ``(eer, threshold)``; NaN if a class is absent."""
    from sklearn.metrics import roc_curve

    labels = np.asarray(labels)
    if np.unique(labels).size < 2:
        return float("nan"), float("nan")
    fpr, tpr, thr = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    i = int(np.nanargmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2.0), float(thr[i])


def evaluate(
    probs: np.ndarray,
    y_true: np.ndarray,
    class_names: List[str],
    bonafide_index: Optional[int] = BONAFIDE_INDEX,
) -> Dict:
    """Compute the full metric bundle for one (model, layer, head).

    ``bonafide_index`` is the column of the bonafide class for the binary
    bonafide-vs-spoof EER; pass ``None`` for label spaces with no bonafide class
    (e.g. the attr17 protocol), in which case ``binary_eer`` is reported as NaN.
    """
    num = len(class_names)
    y_pred = probs.argmax(axis=1)
    cm = confusion(y_true, y_pred, num)
    pca = per_class_accuracy(cm)

    # Binary bonafide-vs-spoof EER: score = P(spoof) = 1 - P(bonafide).
    if bonafide_index is not None and 0 <= bonafide_index < num:
        spoof_score = 1.0 - probs[:, bonafide_index]
        spoof_label = (y_true != bonafide_index).astype(int)
        binary_eer, _ = compute_eer(spoof_score, spoof_label)
    else:
        binary_eer = float("nan")

    # Macro one-vs-rest EER.
    ovr = {
        class_names[c]: compute_eer(probs[:, c], (y_true == c).astype(int))[0]
        for c in range(num)
    }
    macro_eer = float(np.nanmean([v for v in ovr.values()]))

    return {
        "accuracy": float((y_pred == y_true).mean()),
        "binary_eer": binary_eer,
        "macro_ovr_eer": macro_eer,
        "per_class_accuracy": {class_names[i]: float(pca[i]) for i in range(num)},
        "per_class_ovr_eer": {k: float(v) for k, v in ovr.items()},
        "support": {class_names[i]: int((y_true == i).sum()) for i in range(num)},
        "confusion_matrix": cm,
    }


# --------------------------------------------------------------------------- #
# Saving
# --------------------------------------------------------------------------- #
def save_evaluation(out_dir, metrics: Dict, class_names: List[str]) -> None:
    """Write metrics.json, confusion_matrix.{npy,csv,png}, per_class.csv."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cm = np.asarray(metrics["confusion_matrix"])

    # metrics.json (everything except the matrix array)
    summary = {k: v for k, v in metrics.items() if k != "confusion_matrix"}
    (out / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # confusion matrix: npy + labelled csv
    np.save(out / "confusion_matrix.npy", cm)
    with open(out / "confusion_matrix.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["true\\pred", *class_names])
        for name, row in zip(class_names, cm):
            wr.writerow([name, *row.tolist()])

    # per-class table
    with open(out / "per_class.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["class", "support", "accuracy", "ovr_eer"])
        for c in class_names:
            wr.writerow([
                c, metrics["support"][c],
                f"{metrics['per_class_accuracy'][c]:.6f}",
                f"{metrics['per_class_ovr_eer'][c]:.6f}",
            ])

    _save_confusion_png(out / "confusion_matrix.png", cm, class_names)


def _save_confusion_png(path, cm: np.ndarray, class_names: List[str]) -> None:
    """Row-normalised confusion-matrix heatmap (best-effort; skips if no mpl)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.debug("matplotlib unavailable (%s); skipping confusion PNG.", exc)
        return

    rows = cm.sum(axis=1, keepdims=True)
    norm = cm / np.maximum(rows, 1)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if norm[i, j] > 0.5 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row-normalised")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
