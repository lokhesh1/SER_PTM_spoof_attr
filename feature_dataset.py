#!/usr/bin/env python3
"""Feature data loading for layer-wise probing.

Part of the *spoof speech source attribution* project. Loads the per-layer
pooled features written by :mod:`extract_asvspoof` -- one ``layer_KK.npz`` per
layer, each holding ``features (N, D)`` plus the aligned ``attacks (N,)`` /
``labels`` / ``ids`` / ``speakers`` -- and turns a single layer into stratified
train / eval tensors for the classification heads in :mod:`classifier_heads`.

Attribution label space is the fixed 7-class set shared by ASVspoof19 LA
train/dev: ``bonafide`` + ``A01..A06``. Eval attacks (A07+) are rejected with a
clear error so they can't silently corrupt the label encoding.

Programmatic use::

    from feature_dataset import make_layer_loaders
    data = make_layer_loaders("feats_test/asvspoof_la_train__wavlm_base_emotion__mean",
                              layer=6, batch_size=256, device="cuda")
    for xb, yb in data.train_loader:   # xb: (B, 768), yb: (B,)
        ...
"""

from __future__ import annotations

import glob
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger("ser.feature_dataset")

# Fixed 7-class attribution label space (train/dev share these exact labels).
CLASS_NAMES: List[str] = ["bonafide", "A01", "A02", "A03", "A04", "A05", "A06"]
CLASS_TO_IDX: Dict[str, int] = {c: i for i, c in enumerate(CLASS_NAMES)}
BONAFIDE_INDEX: int = CLASS_TO_IDX["bonafide"]


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def discover_model_dirs(
    features_root: os.PathLike, models: Optional[Sequence[str]] = None
) -> List[Path]:
    """Return per-layer feature directories under ``features_root``.

    A directory qualifies if it contains at least one ``layer_*.npz``. ``models``
    optionally filters by substring of the directory name (e.g. ``"wavlm"``).
    """
    root = Path(features_root)
    dirs = sorted(p for p in root.glob("*") if p.is_dir() and any(p.glob("layer_*.npz")))
    if models:
        dirs = [d for d in dirs if any(m in d.name for m in models)]
    return dirs


def model_key_from_dir(feature_dir: os.PathLike) -> str:
    """``asvspoof_la_train__wavlm_base_emotion__mean`` -> ``wavlm_base_emotion``."""
    parts = Path(feature_dir).name.split("__")
    return parts[1] if len(parts) >= 2 else Path(feature_dir).name


def list_layers(feature_dir: os.PathLike) -> List[int]:
    """Sorted layer indices available in a feature directory."""
    layers: List[int] = []
    for f in glob.glob(os.path.join(str(feature_dir), "layer_*.npz")):
        try:
            layers.append(int(Path(f).stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(layers)


# --------------------------------------------------------------------------- #
# Loading & splitting
# --------------------------------------------------------------------------- #
def load_layer(feature_dir: os.PathLike, layer: int) -> Tuple[np.ndarray, np.ndarray]:
    """Load one layer's ``(X, y)`` -- features ``(N, D)`` float32 and integer
    attribution labels ``(N,)`` in ``0..6`` per :data:`CLASS_TO_IDX`."""
    f = Path(feature_dir) / f"layer_{layer:02d}.npz"
    if not f.exists():
        raise FileNotFoundError(f"No such layer file: {f}")
    z = np.load(f, allow_pickle=True)
    X = np.asarray(z["features"], dtype=np.float32)
    attacks = [str(a) for a in z["attacks"]]
    unknown = sorted({a for a in attacks if a not in CLASS_TO_IDX})
    if unknown:
        raise ValueError(
            f"{f}: attack labels outside the 7-class set {CLASS_NAMES}: {unknown}. "
            "(eval attacks A07+ are a different label space and not supported here.)"
        )
    y = np.fromiter((CLASS_TO_IDX[a] for a in attacks), dtype=np.int64, count=len(attacks))
    return X, y


def stratified_split(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Class-stratified train/eval split (eval doubles as test for now)."""
    from sklearn.model_selection import train_test_split

    return train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )


def standardize_fit(X_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-feature ``(mean, std)`` from the training split only."""
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-6
    return mean, std


def balanced_class_weights(counts: np.ndarray) -> torch.Tensor:
    """Inverse-frequency class weights ``n / (C * count_c)`` for imbalanced CE."""
    counts = np.asarray(counts, dtype=np.float64)
    n, c = counts.sum(), len(counts)
    w = np.where(counts > 0, n / (c * np.maximum(counts, 1.0)), 0.0)
    return torch.tensor(w, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Torch dataset / loaders
# --------------------------------------------------------------------------- #
class ArrayDataset(Dataset):
    """In-memory ``(X, y)`` tensor dataset; features are tiny (768-d) so the
    whole split lives on CPU RAM and is cheap to index."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(np.ascontiguousarray(X)).float()
        self.y = torch.from_numpy(np.ascontiguousarray(y)).long()

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, i: int):
        return self.X[i], self.y[i]


@dataclass
class LayerData:
    """Everything a training run needs for one (model, layer)."""

    train_loader: DataLoader
    test_loader: DataLoader
    y_train: np.ndarray
    y_test: np.ndarray
    input_dim: int
    num_classes: int
    class_names: List[str]
    train_class_counts: np.ndarray
    standardization: Optional[Tuple[np.ndarray, np.ndarray]] = None


def make_layer_loaders(
    feature_dir: os.PathLike,
    layer: int,
    *,
    batch_size: int = 256,
    test_size: float = 0.2,
    seed: int = 42,
    standardize: bool = True,
    num_workers: int = 0,
    device: str = "cpu",
) -> LayerData:
    """Build stratified train/eval ``DataLoader``s for one layer."""
    X, y = load_layer(feature_dir, layer)
    X_tr, X_te, y_tr, y_te = stratified_split(X, y, test_size=test_size, seed=seed)

    stats = None
    if standardize:
        mean, std = standardize_fit(X_tr)
        X_tr = (X_tr - mean) / std
        X_te = (X_te - mean) / std
        stats = (mean, std)

    pin = str(device).startswith("cuda")
    train_loader = DataLoader(
        ArrayDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin, drop_last=False,
    )
    test_loader = DataLoader(
        ArrayDataset(X_te, y_te), batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )
    counts = np.bincount(y_tr, minlength=len(CLASS_NAMES))
    return LayerData(
        train_loader=train_loader, test_loader=test_loader,
        y_train=y_tr, y_test=y_te, input_dim=int(X.shape[1]),
        num_classes=len(CLASS_NAMES), class_names=list(CLASS_NAMES),
        train_class_counts=counts, standardization=stats,
    )


# --------------------------------------------------------------------------- #
# Tiny CLI: inspect a features directory
# --------------------------------------------------------------------------- #
def _main() -> int:
    import argparse
    from collections import Counter

    p = argparse.ArgumentParser(description="Inspect a per-layer feature directory.")
    p.add_argument("features_dir", help="e.g. feats_test or a single model dir.")
    args = p.parse_args()

    root = Path(args.features_dir)
    dirs = [root] if any(root.glob("layer_*.npz")) else discover_model_dirs(root)
    if not dirs:
        print(f"No per-layer feature dirs found under {root}")
        return 1
    for d in dirs:
        layers = list_layers(d)
        X, y = load_layer(d, layers[0])
        dist = {CLASS_NAMES[i]: c for i, c in
                sorted(Counter(y.tolist()).items())}
        print(f"{d.name}: layers {layers[0]}..{layers[-1]} ({len(layers)}), "
              f"N={X.shape[0]}, dim={X.shape[1]}")
        print(f"    classes: {dist}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
