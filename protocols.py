#!/usr/bin/env python3
"""Custom train / val / test partitioning protocols for spoof source attribution.

Part of the *spoof speech source attribution* project. The four protocols below
are *re-partitioning + relabeling* schemes layered on top of the per-layer
pooled features written by :mod:`extract_asvspoof` (read-only). Each protocol
defines its own label space, its own train/val/test assignment, and (where the
spec calls for it) speaker-disjoint splitting -- then hands back
``train``/``val``/``test`` :class:`~torch.utils.data.DataLoader`\\ s ready for the
heads in :mod:`classifier_heads`, exactly like :mod:`feature_dataset` does for
the single-subset case.

The four protocols (ASVspoof2019 LA; train/dev = A01-A06, eval = A07-A19, and
A16/A19 reuse the A04/A06 algorithms):

* ``csr1``  -- 7 classes (bonafide + A01-A06). train = 80% of orig train,
  val = 20% of orig train (stratified random split, speakers may overlap),
  test = entire orig dev.
* ``csr2``  -- 14 classes (bonafide + A07-A19). orig eval split 60:20:20
  (stratified random split, speakers may overlap) into train/val/test.
* ``attr2`` -- closed-set, 6-class space (A01-A06, **no bonafide**). train =
  orig train, val = orig dev (both bonafide-filtered), test = orig eval
  restricted to the two *known* attacks A16->A04 and A19->A06.
* ``attr17``-- 17 classes, **no bonafide** (A01-A15, A17, A18; A16->A04,
  A19->A06 merged). A01-A06: orig train split 80:20 (train/val), orig dev = test.
  A07-A19 (from orig eval): 39 "speaker-common" speakers split 50:10:40 across
  train/val/test, 9 "speaker-disjoint" speakers held out to the test partition
  only (speaker-unseen condition).
* ``cv5``   -- 5-fold cross-validation, 20 classes (bonafide + A01-A19, all
  attack ids kept distinct, no merge). train+dev+eval are merged and split into
  5 class-balanced folds (StratifiedKFold; speakers may overlap -- set
  :data:`CV5_SPEAKER_DISJOINT` for speaker-disjoint folds instead). Each fold
  trains on 80% (4 folds) and uses the held-out 20% as both val and test. Driven
  by :data:`PROTOCOL_FOLDS`; the trainer loops folds and reports mean +/- std.

Programmatic use::

    from protocols import make_protocol_loaders
    data = make_protocol_loaders("feats_test", "csr1",
                                 model="wavlm_base_emotion", pooling="mean",
                                 layer=6, device="cuda")
    for xb, yb in data.train_loader:   # xb: (B, D), yb: (B,)
        ...
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from torch.utils.data import DataLoader

from feature_dataset import ArrayDataset, standardize_fit

logger = logging.getLogger("ser.protocols")


# --------------------------------------------------------------------------- #
# Label spaces & protocol constants
# --------------------------------------------------------------------------- #
def _A(*nums: int) -> List[str]:
    return [f"A{n:02d}" for n in nums]


# Eval attack -> algorithm-equivalent train attack. A16 reuses A04's algorithm
# and A19 reuses A06's, so the "known" attacks collapse onto the train labels
# (used by attr2 for the closed-set test and by attr17 for the 17-class merge).
KNOWN_ATTACK_MERGE: Dict[str, str] = {"A16": "A04", "A19": "A06"}

CSR1_CLASSES: List[str] = ["bonafide", *_A(1, 2, 3, 4, 5, 6)]                       # 7
CSR2_CLASSES: List[str] = ["bonafide", *_A(7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19)]  # 14
ATTR2_CLASSES: List[str] = _A(1, 2, 3, 4, 5, 6)                                    # 6, no bonafide (test: A04/A06 only)
ATTR17_CLASSES: List[str] = _A(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18)  # 17, no bonafide
# cv5: merge train+dev+eval; full set = bonafide + all 19 attack ids kept
# distinct (A16/A19 NOT merged). 20 classes.
CV5_CLASSES: List[str] = ["bonafide", *_A(*range(1, 20))]                           # 20

PROTOCOL_CLASSES: Dict[str, List[str]] = {
    "csr1": CSR1_CLASSES, "csr2": CSR2_CLASSES,
    "attr2": ATTR2_CLASSES, "attr17": ATTR17_CLASSES, "cv5": CV5_CLASSES,
}
# Index of the bonafide class, or None when the protocol has no bonafide class
# (controls whether a binary bonafide-vs-spoof EER is meaningful downstream).
PROTOCOL_BONAFIDE: Dict[str, Optional[int]] = {
    "csr1": 0, "csr2": 0, "attr2": None, "attr17": None, "cv5": 0,
}
# Original subsets each protocol needs features extracted for.
PROTOCOL_SUBSETS: Dict[str, List[str]] = {
    "csr1": ["train", "dev"],
    "csr2": ["eval"],
    "attr2": ["train", "dev", "eval"],
    "attr17": ["train", "dev", "eval"],
    "cv5": ["train", "dev", "eval"],
}
# Protocols that run k-fold cross-validation (n folds); others run a single split.
PROTOCOL_FOLDS: Dict[str, int] = {"cv5": 5}

# cv5 fold construction. Default: class-balanced folds (StratifiedKFold by attack
# label; speakers MAY overlap across folds). Set True for speaker-disjoint folds
# (no speaker shared across folds) at the cost of less exact per-fold class balance.
CV5_SPEAKER_DISJOINT: bool = False

DEFAULT_SEED = 42

# attr17: how many of the eval speakers are held out as "speaker-disjoint"
# (placed only in the test partition). To reproduce a specific paper's exact
# split, hard-code the speaker ids in ATTR17_DISJOINT_SPEAKER_IDS below; when
# that is None the ids are chosen deterministically from a seeded shuffle.
ATTR17_DISJOINT_SPEAKERS: int = 9
ATTR17_DISJOINT_SPEAKER_IDS: Optional[List[str]] = None


# --------------------------------------------------------------------------- #
# Feature loading (per subset / model / pooling / layer)
# --------------------------------------------------------------------------- #
def parse_dir_name(name: str) -> Optional[Tuple[str, str, str, bool]]:
    """``asvspoof_la_train__wavlm_base_emotion__mean[__cnn]`` ->
    ``(subset, model, pooling, cnn)``; ``None`` if the name doesn't match."""
    if not name.startswith("asvspoof_la_"):
        return None
    parts = name.split("__")
    if len(parts) < 3:
        return None
    subset = parts[0][len("asvspoof_la_"):]
    model, pooling = parts[1], parts[2]
    cnn = len(parts) >= 4 and parts[3] == "cnn"
    return subset, model, pooling, cnn


def _subset_dir(root: os.PathLike, subset: str, model: str, pooling: str, cnn: bool) -> Path:
    suffix = "__cnn" if cnn else ""
    return Path(root) / f"asvspoof_la_{subset}__{model}__{pooling}{suffix}"


@dataclass
class _Pool:
    """All utterances of one subset for one (model, layer): features + metadata."""

    X: np.ndarray          # (N, D) float32
    attacks: np.ndarray    # (N,) str  -- "bonafide" | "A01".."A19"
    speakers: np.ndarray   # (N,) str


def _load_subset(
    root: os.PathLike, subset: str, model: str, pooling: str, cnn: bool, layer: int
) -> _Pool:
    f = _subset_dir(root, subset, model, pooling, cnn) / f"layer_{layer:02d}.npz"
    if not f.exists():
        raise FileNotFoundError(
            f"Protocol needs '{subset}' features but the layer file is missing: {f}. "
            f"Extract it first, e.g. `extract_asvspoof.py --subset {subset} ...`."
        )
    z = np.load(f, allow_pickle=True)
    return _Pool(
        X=np.asarray(z["features"], dtype=np.float32),
        attacks=np.asarray([str(a) for a in z["attacks"]]),
        speakers=np.asarray([str(s) for s in z["speakers"]]),
    )


# --------------------------------------------------------------------------- #
# Speaker-disjoint splitting
# --------------------------------------------------------------------------- #
def speaker_disjoint_split(
    speakers: np.ndarray,
    fractions: Sequence[float],
    seed: int,
    force_bucket: Optional[Dict[str, int]] = None,
) -> List[np.ndarray]:
    """Partition utterance indices into ``len(fractions)`` buckets so that no
    speaker appears in more than one bucket, approximating the target utterance
    ``fractions`` (which should sum to 1).

    Speakers are visited in a seeded-random order and each is greedily assigned
    to whichever bucket currently has the largest utterance-count deficit
    (target minus filled), skipping buckets with a zero target. ``force_bucket``
    pre-assigns specific speakers to a fixed bucket (e.g. the attr17
    speaker-disjoint set -> the test bucket) before the greedy pass, so those
    speakers' utterances count toward that bucket's target.

    Returns one ``int`` index array per bucket, in ``fractions`` order.
    """
    k = len(fractions)
    uniq, counts = np.unique(speakers, return_counts=True)
    count_map = {s: int(c) for s, c in zip(uniq.tolist(), counts.tolist())}
    order = uniq.tolist()
    np.random.default_rng(seed).shuffle(order)

    total = int(len(speakers))
    targets = [f * total for f in fractions]
    filled = [0.0] * k
    bucket_of: Dict[str, int] = {}

    force_bucket = force_bucket or {}
    for s in order:
        if s in force_bucket:
            b = force_bucket[s]
            bucket_of[s] = b
            filled[b] += count_map[s]

    for s in order:
        if s in bucket_of:
            continue
        deficits = [
            (targets[b] - filled[b]) if targets[b] > 0 else -np.inf for b in range(k)
        ]
        b = int(np.argmax(deficits))
        bucket_of[s] = b
        filled[b] += count_map[s]

    buckets: List[List[int]] = [[] for _ in range(k)]
    for i, s in enumerate(speakers.tolist()):
        buckets[bucket_of[s]].append(i)
    return [np.asarray(b, dtype=int) for b in buckets]


def random_split(
    labels: np.ndarray,
    fractions: Sequence[float],
    seed: int,
) -> List[np.ndarray]:
    """Partition utterance indices into ``len(fractions)`` buckets with a
    *stratified* random split: within each class the indices are shuffled and
    sliced by the cumulative fractions, so every bucket keeps (approximately) the
    same per-class proportions. Speakers are ignored, so the same speaker may
    appear in more than one bucket (NOT speaker-disjoint).

    Returns one ``int`` index array per bucket, in ``fractions`` order.
    """
    k = len(fractions)
    fr = np.asarray(fractions, dtype=float)
    fr = fr / fr.sum()
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    buckets: List[List[int]] = [[] for _ in range(k)]
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        n = len(idx)
        cuts = np.floor(np.cumsum(fr) * n).astype(int)
        prev = 0
        for b in range(k):
            end = n if b == k - 1 else int(cuts[b])
            buckets[b].extend(idx[prev:end].tolist())
            prev = end
    return [np.asarray(sorted(b), dtype=int) for b in buckets]


def _select_disjoint_speakers(speakers: np.ndarray, seed: int) -> List[str]:
    """Choose the attr17 speaker-disjoint (eval-only) speaker ids."""
    uniq = np.unique(speakers).tolist()
    if ATTR17_DISJOINT_SPEAKER_IDS is not None:
        chosen = [s for s in ATTR17_DISJOINT_SPEAKER_IDS if s in set(uniq)]
        missing = [s for s in ATTR17_DISJOINT_SPEAKER_IDS if s not in set(uniq)]
        if missing:
            logger.warning("attr17: configured disjoint speakers not present in eval: %s", missing)
        return sorted(chosen)
    order = list(uniq)
    np.random.default_rng(seed + 999).shuffle(order)
    n = min(ATTR17_DISJOINT_SPEAKERS, max(0, len(order) - 1))
    chosen = sorted(order[:n])
    logger.info(
        "attr17: %d eval speakers, holding out %d as speaker-disjoint (test-only): %s",
        len(uniq), len(chosen), chosen,
    )
    return chosen


def speaker_kfold(speakers: np.ndarray, n_folds: int, seed: int) -> List[np.ndarray]:
    """Partition utterance indices into ``n_folds`` speaker-disjoint groups of
    roughly equal utterance count.

    Speakers are visited largest-first (ties broken by a seeded shuffle) and each
    is assigned to the currently emptiest fold, so no speaker spans two folds and
    the folds stay balanced by utterance count. Returns one ``int`` index array
    per fold.
    """
    uniq, counts = np.unique(speakers, return_counts=True)
    perm = np.random.default_rng(seed).permutation(len(uniq))
    uniq, counts = uniq[perm], counts[perm]
    order = np.argsort(-counts, kind="stable")  # largest first, seed-shuffled ties

    fold_load = [0] * n_folds
    spk_fold: Dict[str, int] = {}
    for j in order.tolist():
        f = int(np.argmin(fold_load))
        spk_fold[uniq[j]] = f
        fold_load[f] += int(counts[j])

    buckets: List[List[int]] = [[] for _ in range(n_folds)]
    for i, s in enumerate(speakers.tolist()):
        buckets[spk_fold[s]].append(i)
    return [np.asarray(b, dtype=int) for b in buckets]


# --------------------------------------------------------------------------- #
# Per-protocol split builders
#   Each returns {"train": (X, attacks), "val": (X, attacks), "test": (X, attacks)}
#   with attacks already relabeled/merged/filtered for that protocol; encoding to
#   integer class ids happens once in make_protocol_loaders.
# --------------------------------------------------------------------------- #
Split = Dict[str, Tuple[np.ndarray, np.ndarray]]


def _merge_known(attacks: np.ndarray) -> np.ndarray:
    """Apply A16->A04 / A19->A06 algorithm merge to an attack-id array."""
    return np.asarray([KNOWN_ATTACK_MERGE.get(a, a) for a in attacks.tolist()])


def _build_csr1(root, model, pooling, cnn, layer, seed) -> Split:
    tr = _load_subset(root, "train", model, pooling, cnn, layer)
    dev = _load_subset(root, "dev", model, pooling, cnn, layer)
    b_tr, b_val = random_split(tr.attacks, [0.8, 0.2], seed)
    return {
        "train": (tr.X[b_tr], tr.attacks[b_tr]),
        "val": (tr.X[b_val], tr.attacks[b_val]),
        "test": (dev.X, dev.attacks),
    }


def _build_csr2(root, model, pooling, cnn, layer, seed) -> Split:
    ev = _load_subset(root, "eval", model, pooling, cnn, layer)
    b_tr, b_val, b_te = random_split(ev.attacks, [0.6, 0.2, 0.2], seed)
    return {
        "train": (ev.X[b_tr], ev.attacks[b_tr]),
        "val": (ev.X[b_val], ev.attacks[b_val]),
        "test": (ev.X[b_te], ev.attacks[b_te]),
    }


def _build_attr2(root, model, pooling, cnn, layer, seed) -> Split:
    del seed  # uses the original partitions only (no random split); seed unused
    tr = _load_subset(root, "train", model, pooling, cnn, layer)
    dev = _load_subset(root, "dev", model, pooling, cnn, layer)
    ev = _load_subset(root, "eval", model, pooling, cnn, layer)
    # No bonafide anywhere: train/val are A01-A06 only; closed-set test is the two
    # known attacks A16/A19 relabeled to A04/A06.
    tr_mask = tr.attacks != "bonafide"
    dev_mask = dev.attacks != "bonafide"
    test_mask = np.isin(ev.attacks, list(KNOWN_ATTACK_MERGE))  # A16, A19
    return {
        "train": (tr.X[tr_mask], tr.attacks[tr_mask]),   # A01-A06
        "val": (dev.X[dev_mask], dev.attacks[dev_mask]),
        "test": (ev.X[test_mask], _merge_known(ev.attacks[test_mask])),
    }


def _build_attr17(root, model, pooling, cnn, layer, seed) -> Split:
    tr = _load_subset(root, "train", model, pooling, cnn, layer)
    dev = _load_subset(root, "dev", model, pooling, cnn, layer)
    ev = _load_subset(root, "eval", model, pooling, cnn, layer)

    train_X, train_a = [], []
    val_X, val_a = [], []
    test_X, test_a = [], []

    # --- A01-A06 portion (bonafide excluded): train split 80:20, dev -> test ---
    tr_mask = tr.attacks != "bonafide"
    b_tr, b_val = speaker_disjoint_split(tr.speakers[tr_mask], [0.8, 0.2], seed)
    Xa, aa = tr.X[tr_mask], tr.attacks[tr_mask]
    train_X.append(Xa[b_tr]); train_a.append(aa[b_tr])
    val_X.append(Xa[b_val]); val_a.append(aa[b_val])
    dev_mask = dev.attacks != "bonafide"
    test_X.append(dev.X[dev_mask]); test_a.append(dev.attacks[dev_mask])

    # --- A07-A19 portion (bonafide excluded, A16->A04 / A19->A06 merged) ------
    ev_mask = ev.attacks != "bonafide"
    Xe = ev.X[ev_mask]
    ae = _merge_known(ev.attacks[ev_mask])
    se = ev.speakers[ev_mask]
    disjoint = set(_select_disjoint_speakers(se, seed))
    force = {s: 2 for s in disjoint}  # bucket 2 == test
    e_tr, e_val, e_te = speaker_disjoint_split(se, [0.5, 0.1, 0.4], seed, force_bucket=force)
    train_X.append(Xe[e_tr]); train_a.append(ae[e_tr])
    val_X.append(Xe[e_val]); val_a.append(ae[e_val])
    test_X.append(Xe[e_te]); test_a.append(ae[e_te])

    return {
        "train": (np.concatenate(train_X), np.concatenate(train_a)),
        "val": (np.concatenate(val_X), np.concatenate(val_a)),
        "test": (np.concatenate(test_X), np.concatenate(test_a)),
    }


_BUILDERS = {
    "csr1": _build_csr1, "csr2": _build_csr2,
    "attr2": _build_attr2, "attr17": _build_attr17,
}


def _build_cv5(root, model, pooling, cnn, layer, seed, fold, n_folds) -> Split:
    """One fold of the cv5 cross-validation split.

    Merges train+dev+eval into one pool (all 19 attack ids kept distinct, no
    merge; bonafide kept -> 20 classes), partitions it into ``n_folds`` folds,
    and returns fold ``fold`` as the held-out 20% used for **both** val and test
    (per the protocol spec), with the other folds as train.

    Folds are **class-balanced** by default (``StratifiedKFold`` on the attack
    label; the same class distribution in every fold, speakers may overlap). Set
    :data:`CV5_SPEAKER_DISJOINT` to make the folds speaker-disjoint instead
    (speaker ids namespaced by subset so the ASVspoof partitions can't collide).
    """
    if not 0 <= fold < n_folds:
        raise ValueError(f"cv5 fold {fold} out of range [0, {n_folds}).")
    Xs, attacks_list, speakers_list = [], [], []
    for subset in PROTOCOL_SUBSETS["cv5"]:
        pool = _load_subset(root, subset, model, pooling, cnn, layer)
        Xs.append(pool.X)
        attacks_list.append(pool.attacks)  # all 19 attack ids kept distinct
        speakers_list.append(np.asarray([f"{subset}:{s}" for s in pool.speakers.tolist()]))
    X = np.concatenate(Xs)
    attacks = np.concatenate(attacks_list)
    speakers = np.concatenate(speakers_list)

    if CV5_SPEAKER_DISJOINT:
        folds = speaker_kfold(speakers, n_folds, seed)
        test_idx = folds[fold]
        train_idx = np.concatenate([folds[i] for i in range(n_folds) if i != fold])
    else:
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        train_idx, test_idx = list(skf.split(np.zeros(len(attacks)), attacks))[fold]
    return {
        "train": (X[train_idx], attacks[train_idx]),
        "val": (X[test_idx], attacks[test_idx]),    # val == test (held-out 20%)
        "test": (X[test_idx], attacks[test_idx]),
    }


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
@dataclass
class ProtocolData:
    """Everything a training run needs for one (protocol, model, layer).

    Field names mirror :class:`feature_dataset.LayerData` (so the same training
    loop consumes both), plus a ``val_loader`` / ``y_val`` for protocol-defined
    validation and ``bonafide_index`` (``None`` when there is no bonafide class).
    """

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
    bonafide_index: Optional[int]
    standardization: Optional[Tuple[np.ndarray, np.ndarray]] = None


def _encode(attacks: np.ndarray, class_to_idx: Dict[str, int], protocol: str, split: str) -> np.ndarray:
    unknown = sorted({a for a in attacks.tolist() if a not in class_to_idx})
    if unknown:
        raise ValueError(
            f"{protocol}/{split}: attack labels outside the class set "
            f"{sorted(class_to_idx)}: {unknown}."
        )
    return np.fromiter((class_to_idx[a] for a in attacks.tolist()),
                       dtype=np.int64, count=len(attacks))


def make_protocol_loaders(
    features_root: os.PathLike,
    protocol: str,
    *,
    model: str,
    pooling: str,
    layer: int,
    cnn: bool = False,
    fold: int = 0,
    n_folds: int = 5,
    batch_size: int = 256,
    seed: int = DEFAULT_SEED,
    standardize: bool = True,
    num_workers: int = 0,
    device: str = "cpu",
) -> ProtocolData:
    """Build train/val/test loaders for one protocol, model and layer.

    ``fold`` / ``n_folds`` are only used by cross-validation protocols (see
    :data:`PROTOCOL_FOLDS`, e.g. ``cv5``); single-split protocols ignore them.
    """
    protocol = protocol.lower()
    if protocol not in PROTOCOL_CLASSES:
        raise KeyError(f"Unknown protocol '{protocol}'. Choose from {sorted(PROTOCOL_CLASSES)}.")

    classes = PROTOCOL_CLASSES[protocol]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    if protocol in PROTOCOL_FOLDS:
        splits = _build_cv5(features_root, model, pooling, cnn, layer, seed, fold, n_folds)
    else:
        splits = _BUILDERS[protocol](features_root, model, pooling, cnn, layer, seed)

    X_tr, a_tr = splits["train"]
    X_val, a_val = splits["val"]
    X_te, a_te = splits["test"]
    y_tr = _encode(a_tr, class_to_idx, protocol, "train")
    y_val = _encode(a_val, class_to_idx, protocol, "val")
    y_te = _encode(a_te, class_to_idx, protocol, "test")

    stats = None
    if standardize:
        mean, std = standardize_fit(X_tr)
        X_tr = (X_tr - mean) / std
        X_val = (X_val - mean) / std
        X_te = (X_te - mean) / std
        stats = (mean, std)

    pin = str(device).startswith("cuda")

    def _loader(X, y, shuffle):
        return DataLoader(
            ArrayDataset(X, y), batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin,
        )

    counts = np.bincount(y_tr, minlength=len(classes))
    tag = f"{'cnn' if cnn else pooling}"
    if protocol in PROTOCOL_FOLDS:
        tag += f" fold {fold + 1}/{n_folds}"
    logger.info(
        "%s [%s L%02d %s]: train=%d val=%d test=%d, %d classes, dim=%d",
        protocol, model, layer, tag,
        len(y_tr), len(y_val), len(y_te), len(classes), int(X_tr.shape[1]),
    )
    return ProtocolData(
        train_loader=_loader(X_tr, y_tr, True),
        val_loader=_loader(X_val, y_val, False) if len(y_val) else None,
        test_loader=_loader(X_te, y_te, False),
        y_train=y_tr, y_val=y_val, y_test=y_te,
        input_dim=int(X_tr.shape[1]), num_classes=len(classes),
        class_names=list(classes), train_class_counts=counts,
        bonafide_index=PROTOCOL_BONAFIDE[protocol], standardization=stats,
    )


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def discover_feature_groups(
    features_root: os.PathLike, models: Optional[Sequence[str]] = None
) -> Dict[Tuple[str, str, bool], Dict[str, List[int]]]:
    """Group the feature dirs under ``features_root`` by ``(model, pooling,
    cnn)`` and, for each group, map every present ``subset`` to its layer list.

    ``models`` optionally filters by substring of the model key.
    """
    root = Path(features_root)
    groups: Dict[Tuple[str, str, bool], Dict[str, List[int]]] = {}
    for d in sorted(root.glob("asvspoof_la_*")):
        if not d.is_dir():
            continue
        parsed = parse_dir_name(d.name)
        if parsed is None:
            continue
        subset, model, pooling, cnn = parsed
        if models and not any(m in model for m in models):
            continue
        layers = sorted(
            int(p.stem.split("_")[1])
            for p in d.glob("layer_*.npz")
            if p.stem.split("_")[1].isdigit()
        )
        if layers:
            groups.setdefault((model, pooling, cnn), {})[subset] = layers
    return groups


def protocol_layers(group_subsets: Dict[str, List[int]], protocol: str) -> List[int]:
    """Layers usable for ``protocol`` in a feature group: the intersection of the
    layer sets of every subset the protocol requires (empty if a subset is
    missing)."""
    needed = PROTOCOL_SUBSETS[protocol.lower()]
    if any(s not in group_subsets for s in needed):
        return []
    common = set(group_subsets[needed[0]])
    for s in needed[1:]:
        common &= set(group_subsets[s])
    return sorted(common)


# --------------------------------------------------------------------------- #
# Tiny CLI: inspect a protocol's splits for one model/layer
# --------------------------------------------------------------------------- #
def _main() -> int:
    import argparse
    from collections import Counter

    p = argparse.ArgumentParser(description="Inspect a protocol's train/val/test split.")
    p.add_argument("features_root")
    p.add_argument("protocol", choices=sorted(PROTOCOL_CLASSES))
    p.add_argument("--model", required=True)
    p.add_argument("--pooling", default="mean")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--cnn", action="store_true")
    p.add_argument("--fold", type=int, default=0, help="CV fold to inspect (cv5).")
    p.add_argument("--cv-folds", type=int, default=5, help="CV fold count (cv5).")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    data = make_protocol_loaders(
        args.features_root, args.protocol, model=args.model, pooling=args.pooling,
        layer=args.layer, cnn=args.cnn, fold=args.fold, n_folds=args.cv_folds,
        seed=args.seed,
    )
    names = data.class_names
    for split, y in (("train", data.y_train), ("val", data.y_val), ("test", data.y_test)):
        dist = {names[i]: c for i, c in sorted(Counter(y.tolist()).items())}
        print(f"{split:5s} N={len(y):6d}  {dist}")
    print(f"classes ({data.num_classes}): {names}")
    print(f"bonafide_index: {data.bonafide_index}   input_dim: {data.input_dim}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
