#!/usr/bin/env python3
"""Batch SER feature extraction over ASVspoof2019 LA *train*.

Part of the *spoof speech source attribution* project. This driver extracts
features from **all encoder layers** of one or more registered SER models for
every utterance in the ASVspoof2019 LA training subset, pools each layer over
time (mean+std by default), and writes one consolidated ``.npz`` per model.

Why consolidated (not one file per utterance): 25,380 utterances x 2 models
would be 50k tiny files. Instead each model produces a single archive holding
``layer_k`` arrays of shape ``(N, P)`` plus aligned ``ids`` / ``labels`` /
``attacks`` -- exactly the layout a layer-wise attribution classifier wants.

The CM protocol is the source of truth for the file list *and* the labels:

    LA_0079 LA_T_1138215 - -   bonafide      # cols: spk utt _ attack label
    LA_0096 LA_T_4724763 - A01 spoof         # attack in {A01..A06} for train

so each saved row carries its attack system id (A01-A06) / ``bonafide`` -- the
target for source attribution.

Output (per model ``<m>``, in ``--out-dir``). Default ``--layout per-layer``
writes one self-contained file per layer so per-layer training loads exactly
one file::

    asvspoof_la_<subset>__<m>__<pooling>/         # <subset> in {train,dev,eval}
        layer_00.npz, layer_01.npz, ...   # each:
            features : (N, P)  pooled features, P = pool_dim * D (mean -> D)
            layer    : ()      this layer's index
            ids      : (N,)    utterance ids   (e.g. LA_T_1138215)
            labels   : (N,)    bonafide | spoof
            attacks  : (N,)    A01..A06 | bonafide   (attribution target)
            speakers : (N,)    speaker ids

``--layout consolidated`` instead writes a single ``<base>.npz`` holding
``layer_0 .. layer_L`` members plus the shared ids/labels/attacks/speakers
(still lazily loadable one layer at a time: ``np.load(f)["layer_6"]``).

Train on a single layer::

    import numpy as np
    d = np.load("features_asvspoof_train/asvspoof_la_train__wavlm_base_emotion__mean/layer_06.npz")
    X, y = d["features"], d["attacks"]      # (N, 768), (N,) in {bonafide, A01..A06}

Examples
--------
Both target models, defaults (train only, per-layer, mean -> 768 dim, float32)::

    python extract_asvspoof.py --data-root /home/hp/Desktop/Spoof_source_attr/LA

Include the dev subset too (dev shares attacks A01-A06 with train)::

    python extract_asvspoof.py --data-root .../LA --subset train dev

Smoke-test on 50 utterances first::

    python extract_asvspoof.py --data-root .../LA --limit 50 --out-dir feats_test

All layers in one file per model instead::

    python extract_asvspoof.py --data-root .../LA --layout consolidated
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from model_loader import configure_logging, load_model
from ser_feature_extractor import ExtractionConfig, extract_features

logger = logging.getLogger("ser.extract_asvspoof")

# Default models for this study: WavLM-base-emotion + emotion2vec+ base.
DEFAULT_MODELS = ["wavlm_base_emotion", "emotion2vec_plus_base"]

# Subset -> (flac subdir, CM protocol filename) under <data-root>, for the
# standard ASVspoof2019 LA release. train/dev share attacks A01-A06; eval uses
# the disjoint A07-A19. (train protocol is .trn.txt; dev/eval are .trl.txt.)
PROTOCOL_DIR = "ASVspoof2019_LA_cm_protocols"
SUBSETS: Dict[str, tuple] = {
    "train": ("ASVspoof2019_LA_train/flac", "ASVspoof2019.LA.cm.train.trn.txt"),
    "dev":   ("ASVspoof2019_LA_dev/flac",   "ASVspoof2019.LA.cm.dev.trl.txt"),
    "eval":  ("ASVspoof2019_LA_eval/flac",  "ASVspoof2019.LA.cm.eval.trl.txt"),
}


# --------------------------------------------------------------------------- #
# Protocol parsing
# --------------------------------------------------------------------------- #
@dataclass
class Utt:
    """One protocol entry."""

    utt_id: str
    speaker: str
    attack: str  # "A01".."A06" or "bonafide"
    label: str   # "bonafide" | "spoof"


def read_protocol(protocol_path: Path) -> List[Utt]:
    """Parse a CM protocol file into a list of :class:`Utt`.

    Columns (whitespace separated): ``speaker utt _ attack label``. For bonafide
    rows the attack column is ``-``; we normalise that to ``"bonafide"`` so the
    ``attacks`` array is a clean categorical attribution target.
    """
    utts: List[Utt] = []
    with open(protocol_path, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            parts = raw.split()
            if not parts:
                continue
            if len(parts) < 5:
                logger.warning("Protocol line %d malformed, skipping: %r", lineno, raw)
                continue
            speaker, utt_id, _gen, attack, label = parts[:5]
            attack = "bonafide" if label == "bonafide" else attack
            utts.append(Utt(utt_id=utt_id, speaker=speaker, attack=attack, label=label))
    return utts


# --------------------------------------------------------------------------- #
# Per-model extraction
# --------------------------------------------------------------------------- #
def model_output_base(
    out_dir: Path, subset: str, model_key: str, pooling: Optional[str],
    extract_cnn: bool = False,
) -> Path:
    """Extension-less base path for a (subset, model) output (a ``.npz`` file or
    a directory, depending on the chosen layout).

    CNN-layer runs get a ``__cnn`` suffix so their conv/early-transformer
    features never collide with (or get skipped by) a full transformer-layer run
    in the same ``--out-dir``."""
    suffix = "__cnn" if extract_cnn else ""
    return Path(out_dir) / f"asvspoof_la_{subset}__{model_key}__{pooling}{suffix}"


def output_exists(base: Path, layout: str) -> bool:
    """Whether a previous run already produced this model's output."""
    if layout == "consolidated":
        return Path(f"{base}.npz").exists()
    return base.is_dir() and any(base.glob("layer_*.npz"))


def extract_model(
    model_key: str,
    subset: str,
    utts: Sequence[Utt],
    flac_dir: Path,
    out_dir: Path,
    *,
    config: ExtractionConfig,
    dtype: str,
    device: str,
    layout: str,
    log_every: int,
    extract_cnn: bool = False,
) -> None:
    """Extract pooled all-layer features for ``utts`` with one model and save.

    ``layout`` controls the on-disk shape:
      * ``"per-layer"``  -> a directory ``<base>/`` with one self-contained
        ``layer_KK.npz`` per layer (each carries ``features`` + the shared
        ``ids/labels/attacks/speakers``), so training on a single layer means
        loading exactly one file.
      * ``"consolidated"`` -> a single ``<base>.npz`` with ``layer_K`` members
        (still lazily loadable one layer at a time via ``np.load(f)["layer_6"]``).
    """
    base = model_output_base(out_dir, subset, model_key, config.pooling, extract_cnn)
    logger.info("=== [%s] %s -> %s ===", subset, model_key, base)
    model = load_model(model_key, device=device, extract_cnn=extract_cnn)

    n = len(utts)
    # Pre-allocated per-layer arrays, sized on the first successful utterance
    # (we don't know the layer set / pooled dim until we run one forward pass).
    buffers: Optional[Dict[int, np.ndarray]] = None
    layer_keys: Optional[List[int]] = None

    ids: List[str] = []
    speakers: List[str] = []
    attacks: List[str] = []
    labels: List[str] = []
    failed: List[str] = []

    w = 0  # write pointer into the pre-allocated buffers
    t0 = time.time()
    for i, utt in enumerate(utts):
        flac = flac_dir / f"{utt.utt_id}.flac"
        if not flac.exists():
            logger.warning("Missing file, skipping: %s", flac)
            failed.append(utt.utt_id)
            continue
        try:
            result = extract_features(model, flac, config)
        except Exception as exc:  # one bad file shouldn't kill a 25k-file run
            logger.warning("Extraction failed for %s (%s); skipping.", utt.utt_id, exc)
            failed.append(utt.utt_id)
            continue

        feats = result.features  # {layer_idx: (P,) pooled vector}
        if buffers is None:
            layer_keys = sorted(feats)
            buffers = {
                k: np.empty((n, feats[k].shape[-1]), dtype=dtype) for k in layer_keys
            }
            logger.info(
                "Layers %s, pooled dim per layer %s",
                layer_keys, {k: int(buffers[k].shape[1]) for k in layer_keys},
            )
        elif sorted(feats) != layer_keys:
            logger.warning(
                "%s exposes layers %s != expected %s; skipping.",
                utt.utt_id, sorted(feats), layer_keys,
            )
            failed.append(utt.utt_id)
            continue

        for k in layer_keys:
            buffers[k][w] = feats[k].astype(dtype, copy=False)
        ids.append(utt.utt_id)
        speakers.append(utt.speaker)
        attacks.append(utt.attack)
        labels.append(utt.label)
        w += 1

        if (i + 1) % log_every == 0 or (i + 1) == n:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - (i + 1)) / rate if rate > 0 else float("nan")
            logger.info(
                "%s: %d/%d  (%.1f utt/s, ETA %.1f min, %d failed)",
                model_key, i + 1, n, rate, eta / 60.0, len(failed),
            )

    if buffers is None or w == 0:
        logger.error("%s: no features extracted; nothing saved.", model_key)
        return

    # Labels/ids shared by every layer (trimmed to the rows actually filled).
    label_meta = {
        "ids": np.asarray(ids),
        "speakers": np.asarray(speakers),
        "attacks": np.asarray(attacks),
        "labels": np.asarray(labels),
    }
    info = {
        "model": np.asarray(model_key),
        "pooling": np.asarray(str(config.pooling)),
        "dtype": np.asarray(dtype),
    }

    if layout == "consolidated":
        out_path = Path(f"{base}.npz")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"layer_{k}": buffers[k][:w] for k in layer_keys}
        np.savez(out_path, **arrays, **label_meta,
                 layers=np.asarray(layer_keys, dtype=np.int64), **info)
        saved = [out_path]
        target = out_path
    else:  # per-layer: one self-contained file per layer
        layer_dir = Path(base)
        layer_dir.mkdir(parents=True, exist_ok=True)
        saved = []
        for k in layer_keys:
            # int keys -> zero-padded layer_06.npz; CNN-mode string keys
            # (cnn_0/tf_0) are used verbatim -> layer_cnn_0.npz / layer_tf_0.npz.
            name = f"layer_{k:02d}" if isinstance(k, int) else f"layer_{k}"
            lp = layer_dir / f"{name}.npz"
            np.savez(lp, features=buffers[k][:w], layer=np.asarray(k),
                     **label_meta, **info)
            saved.append(lp)
        target = layer_dir

    total_gib = sum(p.stat().st_size for p in saved) / 2**30
    logger.info(
        "Saved %s: %d utts, %d layers (%s), %s float -> %s (%.2f GiB)%s",
        model_key, w, len(layer_keys), layout, dtype, target, total_gib,
        f", {len(failed)} failed" if failed else "",
    )
    if failed:
        fail_log = Path(f"{base}.failed.txt")
        fail_log.parent.mkdir(parents=True, exist_ok=True)
        fail_log.write_text("\n".join(failed) + "\n", encoding="utf-8")
        logger.info("Wrote %d failed ids -> %s", len(failed), fail_log)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract all-layer SER features over ASVspoof2019 LA "
                    "(train by default; add dev/eval via --subset).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data-root", required=True,
        help="ASVspoof2019 LA root containing ASVspoof2019_LA_{train,dev,eval}/ "
             "and ASVspoof2019_LA_cm_protocols/.",
    )
    p.add_argument(
        "--subset", nargs="+", choices=list(SUBSETS), default=["train"],
        help="Which subset(s) to extract. Default: train only. "
             "Pass e.g. '--subset train dev' to include dev.",
    )
    p.add_argument(
        "--flac-dir", default=None,
        help="Override flac dir (requires a single --subset; "
             "default <data-root>/ASVspoof2019_LA_<subset>/flac).",
    )
    p.add_argument(
        "--protocol", default=None,
        help="Override CM protocol path (requires a single --subset).",
    )
    p.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS,
        help="Registered model keys to extract with.",
    )
    p.add_argument(
        "--layers", nargs="+", default=None,
        help="Layer indices to keep (e.g. 0 6 12) or omit for ALL layers. "
             "Ignored when --cnn-layers is set.",
    )
    p.add_argument(
        "--cnn-layers", action="store_true",
        help="Extract every CNN feature-encoder layer (cnn_0..cnn_N) plus the "
             "first three hidden states (tf_0/tf_1/tf_2 = input embedding + "
             "transformer blocks 1-2) instead of all transformer layers. "
             "transformers backend only (WavLM / HuBERT / wav2vec2). Output dir "
             "gets a __cnn suffix.",
    )
    p.add_argument(
        "--pooling", choices=["mean", "max", "mean_std"], default="mean",
        help="Temporal pooling applied to every layer. 'mean'/'max' keep the "
             "native hidden dim (768); 'mean_std' concatenates mean+std (1536).",
    )
    p.add_argument(
        "--dtype", choices=["float16", "float32"], default="float32",
        help="On-disk numeric precision for the feature arrays.",
    )
    p.add_argument(
        "--layout", choices=["per-layer", "consolidated"], default="per-layer",
        help="per-layer: one self-contained file per layer (best for per-layer "
             "training). consolidated: all layers in one .npz per model.",
    )
    p.add_argument("--out-dir", default="features_asvspoof_train", help="Output directory.")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:0 ...")
    p.add_argument("--limit", type=int, default=None, help="Process only the first N utts (smoke test).")
    p.add_argument("--overwrite", action="store_true", help="Re-extract even if output exists.")
    p.add_argument("--log-every", type=int, default=500, help="Progress log interval (utts).")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    configure_logging(verbose=args.verbose)

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    subsets = args.subset

    if (args.flac_dir or args.protocol) and len(subsets) != 1:
        logger.error(
            "--flac-dir/--protocol overrides require exactly one --subset; got %s",
            subsets,
        )
        return 2

    layers = [int(x) for x in args.layers] if args.layers else None
    if args.cnn_layers and layers is not None:
        logger.warning("--cnn-layers ignores --layers; extracting all cnn_*/tf_* keys.")
        layers = None
    config = ExtractionConfig(layers=layers, pooling=args.pooling, save=False)

    for subset in subsets:
        flac_subdir, proto_name = SUBSETS[subset]
        flac_dir = Path(args.flac_dir) if args.flac_dir else data_root / flac_subdir
        protocol = (
            Path(args.protocol) if args.protocol
            else data_root / PROTOCOL_DIR / proto_name
        )

        missing = [str(p) for p in (flac_dir, protocol) if not p.exists()]
        if missing:
            logger.error("[%s] not found, skipping subset: %s", subset, missing)
            continue

        utts = read_protocol(protocol)
        if args.limit:
            utts = utts[: args.limit]
        n_bona = sum(u.label == "bonafide" for u in utts)
        logger.info(
            "[%s] Loaded %d utts from %s (%d bonafide, %d spoof). Models: %s",
            subset, len(utts), protocol.name, n_bona, len(utts) - n_bona, args.models,
        )

        for model_key in args.models:
            base = model_output_base(out_dir, subset, model_key, args.pooling,
                                     args.cnn_layers)
            if output_exists(base, args.layout) and not args.overwrite:
                logger.info("Exists, skipping (use --overwrite): %s", base)
                continue
            extract_model(
                model_key, subset, utts, flac_dir, out_dir,
                config=config, dtype=args.dtype, device=args.device,
                layout=args.layout, log_every=args.log_every,
                extract_cnn=args.cnn_layers,
            )

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
