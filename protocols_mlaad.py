#!/usr/bin/env python3
"""MLAAD v5 ratio-based source-tracing split (closed-set, language-disjoint test).

Self-contained companion to :mod:`protocols` (ASVspoof). This module does **not**
import from or modify ``protocols.py`` / ``extract_asvspoof.py``; it reuses only
the backend-agnostic helpers in :mod:`feature_dataset` (:class:`ArrayDataset`,
:func:`standardize_fit`) for loader construction, exactly as ``protocols.py``
does.

**Pool** ("B"): every clip under ``<MLAAD_ROOT>/fake/<lang>/<model>/*.wav``
(154,000 clips, 82 generators, 38 languages). The attribution label is the TTS
``model_name`` -- a **closed-set, 82-class** target (no bonafide).

**Split** (seed 42, ratio 60:10:30):

* 7 languages -- ``ar, hi, ja, ko, th, tk, vi`` -- are held out **entirely** to
  test (the unseen-language condition). They were chosen because together they
  capture *no* generator exclusively, so every one of the 82 classes still
  appears in train: closed-set is preserved and nothing is dropped.
* the remaining 31 languages are split **stratified by model** so the *global*
  ratio lands at 60:10:30 (held-out clips, ~8.4% of the pool, count toward the
  test 30%; the rest of test is drawn from the shared languages).

Feature extraction is **split-agnostic**: :mod:`extract_mlaad` writes one
``.npz`` per clip mirroring the MLAAD tree under ``feat_<encoder>_<layer>/``.
This module only decides the partition (a manifest CSV) and, given those feature
trees, builds train/val/test loaders.

CLI::

    python protocols_mlaad.py --summary           # split sizes + closed-set check
    python protocols_mlaad.py --write-manifest mlaad_split_seed42.csv
"""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from torch.utils.data import DataLoader

from feature_dataset import ArrayDataset, standardize_fit

logger = logging.getLogger("ser.protocols_mlaad")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
MLAAD_ROOT_DEFAULT = Path("MLAAD/Mlaad_v5/mlaad_v5")

# Languages held out 100% to test. These 7 share every one of their generators
# with at least one non-held-out language, so holding them out removes no class
# from train (closed-set safe). Adding an 8th would orphan >=1 generator.
HELDOUT_TEST_LANGUAGES: List[str] = ["ar", "hi", "ja", "ko", "th", "tk", "vi"]

RATIOS: Tuple[float, float, float] = (0.6, 0.1, 0.3)  # train, val, test
DEFAULT_SEED = 42

ENCODER_DEFAULT = "hubert_emotion"
# feat_<encoder>_<suffix> dir -> what that layer is (cnn-mode key it came from).
FEATURE_LAYER_DIRS: Dict[str, str] = {
    "00": "cnn[-1] last conv (512-d)",
    "01": "tf_0 = hidden_states[0] (1024-d)",
}


# --------------------------------------------------------------------------- #
# Pool scan (shared with extract_mlaad)
# --------------------------------------------------------------------------- #
def _model_name_of_dir(model_dir: Path) -> str:
    """Read the generator ``model_name`` for a ``fake/<lang>/<model>/`` dir from
    its ``meta.csv`` (``|``-delimited); fall back to the directory name."""
    meta = model_dir / "meta.csv"
    if meta.exists():
        try:
            with open(meta, newline="") as f:
                row = next(csv.DictReader(f, delimiter="|"))
                if row.get("model_name"):
                    return row["model_name"]
        except (StopIteration, KeyError, OSError):
            pass
    logger.warning("No usable meta.csv in %s; using dir name as label.", model_dir)
    return model_dir.name


def scan_pool(root: os.PathLike = MLAAD_ROOT_DEFAULT) -> List[Tuple[str, str, str]]:
    """Walk ``<root>/fake/<lang>/<model>/*.wav`` and return one
    ``(rel_path, model_name, language)`` per clip, where ``rel_path`` is relative
    to ``root`` (e.g. ``fake/en/tts_models_en_.../foo.wav``). Deterministic order."""
    root = Path(root)
    fake = root / "fake"
    if not fake.is_dir():
        raise FileNotFoundError(f"MLAAD fake/ tree not found under {root}")
    recs: List[Tuple[str, str, str]] = []
    for lang_dir in sorted(p for p in fake.iterdir() if p.is_dir()):
        language = lang_dir.name
        for model_dir in sorted(p for p in lang_dir.iterdir() if p.is_dir()):
            model_name = _model_name_of_dir(model_dir)
            for wav in sorted(model_dir.glob("*.wav")):
                recs.append((str(wav.relative_to(root)), model_name, language))
    return recs


# --------------------------------------------------------------------------- #
# Split
# --------------------------------------------------------------------------- #
@dataclass
class MlaadSplit:
    """One built ratio split over the pool."""

    rel_paths: np.ndarray   # (N,) str, relative to MLAAD root
    models: np.ndarray      # (N,) str, the model_name label
    languages: np.ndarray   # (N,) str
    split: np.ndarray       # (N,) str in {"train","val","test"}
    class_names: List[str]  # sorted unique models (closed-set label space)

    def mask(self, name: str) -> np.ndarray:
        return self.split == name


def build_ratio_split(
    root: os.PathLike = MLAAD_ROOT_DEFAULT,
    *,
    seed: int = DEFAULT_SEED,
    heldout_languages: Sequence[str] = HELDOUT_TEST_LANGUAGES,
    ratios: Tuple[float, float, float] = RATIOS,
) -> MlaadSplit:
    """Build the 60:10:30 split: held-out languages -> test, the rest split
    stratified by model so the global ratio is met. Deterministic given ``seed``."""
    recs = scan_pool(root)
    rels = np.asarray([r for r, _, _ in recs])
    models = np.asarray([m for _, m, _ in recs])
    langs = np.asarray([l for _, _, l in recs])
    N = len(recs)

    held = set(heldout_languages)
    is_held = np.isin(langs, list(held))
    n_held = int(is_held.sum())

    f_tr, f_val, f_te = ratios
    n_train = int(round(f_tr * N))
    n_val = int(round(f_val * N))
    n_test = N - n_train - n_val
    test_from_shared = n_test - n_held
    if test_from_shared < 0:
        raise ValueError(
            f"Held-out languages are {100 * n_held / N:.1f}% of the pool, which "
            f"exceeds the test ratio {100 * f_te:.0f}%. Choose smaller languages."
        )
    shared_N = N - n_held
    # Per-model fractions applied within the shared languages so the GLOBAL totals
    # hit (n_train, n_val, n_test). They sum to 1 by construction.
    sf_tr = n_train / shared_N
    sf_val = n_val / shared_N

    split = np.empty(N, dtype=object)
    split[is_held] = "test"

    rng = np.random.default_rng(seed)
    shared_idx = np.where(~is_held)[0]
    for m in np.unique(models[shared_idx]):
        idx = shared_idx[models[shared_idx] == m]
        idx = rng.permutation(idx)
        n = len(idx)
        c_tr = min(int(round(sf_tr * n)), n)
        c_val = min(int(round(sf_val * n)), n - c_tr)
        split[idx[:c_tr]] = "train"
        split[idx[c_tr:c_tr + c_val]] = "val"
        split[idx[c_tr + c_val:]] = "test"

    return MlaadSplit(
        rel_paths=rels, models=models, languages=langs,
        split=split.astype(str), class_names=sorted(set(models.tolist())),
    )


# --------------------------------------------------------------------------- #
# Manifest I/O
# --------------------------------------------------------------------------- #
_MANIFEST_FIELDS = ["path", "model_name", "language", "split"]


def write_manifest(split: MlaadSplit, path: os.PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(_MANIFEST_FIELDS)
        for r, m, l, s in zip(split.rel_paths, split.models, split.languages, split.split):
            w.writerow([r, m, l, s])
    logger.info("Wrote manifest (%d rows) -> %s", len(split.rel_paths), path)


def load_manifest(path: os.PathLike) -> MlaadSplit:
    rels, models, langs, splits = [], [], [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rels.append(row["path"]); models.append(row["model_name"])
            langs.append(row["language"]); splits.append(row["split"])
    models_arr = np.asarray(models)
    return MlaadSplit(
        rel_paths=np.asarray(rels), models=models_arr, languages=np.asarray(langs),
        split=np.asarray(splits), class_names=sorted(set(models_arr.tolist())),
    )


# --------------------------------------------------------------------------- #
# Loaders (read the per-clip feature tree back into train/val/test)
# --------------------------------------------------------------------------- #
@dataclass
class MlaadData:
    """Mirrors :class:`protocols.ProtocolData` field names so the existing
    ``train_probes`` evaluation path can consume it unchanged."""

    train_loader: DataLoader
    val_loader: Optional[DataLoader]
    test_loader: DataLoader
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    input_dim: int
    num_classes: int
    class_names: List[str]
    train_class_counts: np.ndarray
    bonafide_index: Optional[int]  # always None for MLAAD (no bonafide class)
    standardization: Optional[Tuple[np.ndarray, np.ndarray]] = None


def _load_feature_split(
    feature_dir: Path, rels: np.ndarray
) -> np.ndarray:
    """Stack the per-clip ``features`` vectors for ``rels`` from ``feature_dir``
    (the ``feat_<encoder>_<suffix>/`` tree mirroring the MLAAD layout)."""
    vecs = []
    missing = 0
    first_missing = None
    for r in rels.tolist():
        f = feature_dir / Path(r).with_suffix(".npz")
        if not f.exists():
            missing += 1
            if first_missing is None:
                first_missing = f
            continue
        vecs.append(np.load(f)["features"])
    if missing:
        raise FileNotFoundError(
            f"{missing} feature files missing under {feature_dir} (e.g. {first_missing}). "
            f"Run extract_mlaad.py first."
        )
    return np.asarray(vecs, dtype=np.float32)


def make_mlaad_loaders(
    feature_dir: os.PathLike,
    split: MlaadSplit,
    *,
    batch_size: int = 256,
    standardize: bool = True,
    num_workers: int = 0,
    device: str = "cpu",
) -> MlaadData:
    """Build train/val/test loaders for one ``feat_<encoder>_<suffix>/`` tree.

    ``feature_dir`` is the directory whose contents mirror the MLAAD ``fake/``
    tree (one ``.npz`` per clip). ``split`` is a built/loaded :class:`MlaadSplit`.
    """
    feature_dir = Path(feature_dir)
    class_to_idx = {c: i for i, c in enumerate(split.class_names)}

    def _xy(name: str) -> Tuple[np.ndarray, np.ndarray]:
        m = split.mask(name)
        X = _load_feature_split(feature_dir, split.rel_paths[m])
        y = np.fromiter((class_to_idx[a] for a in split.models[m].tolist()),
                        dtype=np.int64, count=int(m.sum()))
        return X, y

    X_tr, y_tr = _xy("train")
    if X_tr.size == 0:
        raise ValueError(f"{feature_dir.name}: empty train split -- nothing to fit on.")
    dim = X_tr.shape[1]

    def _as2d(X: np.ndarray) -> np.ndarray:  # empty splits come back (0,); make (0, D)
        return X if X.size else np.zeros((0, dim), dtype=np.float32)

    X_val, y_val = _xy("val"); X_val = _as2d(X_val)
    X_te, y_te = _xy("test"); X_te = _as2d(X_te)

    stats = None
    if standardize:
        mean, std = standardize_fit(X_tr)
        X_tr = (X_tr - mean) / std
        X_val = (X_val - mean) / std
        X_te = (X_te - mean) / std
        stats = (mean, std)

    pin = str(device).startswith("cuda")

    def _loader(X, y, shuffle):
        return DataLoader(ArrayDataset(X, y), batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=pin)

    counts = np.bincount(y_tr, minlength=len(split.class_names))
    logger.info("mlaad [%s]: train=%d val=%d test=%d, %d classes, dim=%d",
                feature_dir.name, len(y_tr), len(y_val), len(y_te),
                len(split.class_names), int(X_tr.shape[1]))
    return MlaadData(
        train_loader=_loader(X_tr, y_tr, True),
        val_loader=_loader(X_val, y_val, False) if len(y_val) else None,
        test_loader=_loader(X_te, y_te, False),
        y_train=y_tr, y_val=y_val, y_test=y_te,
        input_dim=int(X_tr.shape[1]), num_classes=len(split.class_names),
        class_names=list(split.class_names), train_class_counts=counts,
        bonafide_index=None, standardization=stats,
    )


# --------------------------------------------------------------------------- #
# CLI: summary / manifest / closed-set verification
# --------------------------------------------------------------------------- #
def _summary(split: MlaadSplit) -> None:
    from collections import Counter
    N = len(split.rel_paths)
    train_models = set(split.models[split.mask("train")].tolist())
    print(f"pool: {N} clips, {len(split.class_names)} classes, "
          f"{len(set(split.languages.tolist()))} languages")
    for name in ("train", "val", "test"):
        m = split.mask(name)
        langs = sorted(set(split.languages[m].tolist()))
        n_models = len(set(split.models[m].tolist()))
        print(f"  {name:5s} N={int(m.sum()):6d} ({100*m.sum()/N:4.1f}%)  "
              f"classes={n_models:3d}  languages={len(langs)}")
    # closed-set check: every test class present in train
    test_models = set(split.models[split.mask("test")].tolist())
    val_models = set(split.models[split.mask("val")].tolist())
    missing_te = sorted(test_models - train_models)
    missing_val = sorted(val_models - train_models)
    held = set(HELDOUT_TEST_LANGUAGES)
    train_val_langs = set(split.languages[split.mask("train") | split.mask("val")].tolist())
    print(f"\nclosed-set: train covers all classes = {len(train_models) == len(split.class_names)} "
          f"({len(train_models)}/{len(split.class_names)})")
    if missing_te: print(f"  WARNING test classes absent from train: {missing_te}")
    if missing_val: print(f"  WARNING val classes absent from train: {missing_val}")
    print(f"held-out languages strictly test-only = {held.isdisjoint(train_val_langs)} "
          f"({sorted(held)})")


def _main() -> int:
    import argparse
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(MLAAD_ROOT_DEFAULT))
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--summary", action="store_true", help="print split sizes + closed-set check")
    p.add_argument("--write-manifest", metavar="PATH", default=None,
                   help="build the split and write it to a manifest CSV")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    split = build_ratio_split(args.root, seed=args.seed)
    if args.summary or not args.write_manifest:
        _summary(split)
    if args.write_manifest:
        write_manifest(split, args.write_manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
