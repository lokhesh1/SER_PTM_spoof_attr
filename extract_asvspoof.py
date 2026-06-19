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

Output (per model ``<m>``, in ``--out-dir``)::

    asvspoof_la_train__<m>__mean_std.npz
        ids      : (N,)  utterance ids        (e.g. LA_T_1138215)
        labels   : (N,)  bonafide | spoof
        attacks  : (N,)  A01..A06 | bonafide   (attribution target)
        speakers : (N,)  speaker ids
        layers   : (L,)  layer indices present (e.g. 0..12)
        layer_0 .. layer_L : (N, P)  pooled features, P = pool_dim * D

Examples
--------
Both target models, defaults (mean_std, float32, all layers)::

    python extract_asvspoof.py --data-root /home/hp/Desktop/Spoof_source_attr/LA

Smoke-test on 50 utterances first::

    python extract_asvspoof.py --data-root .../LA --limit 50 --out-dir feats_test
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

from model_loader import load_model
from ser_feature_extractor import ExtractionConfig, extract_features

logger = logging.getLogger("ser.extract_asvspoof")

# Default models for this study: WavLM-base-emotion + emotion2vec+ base.
DEFAULT_MODELS = ["wavlm_base_emotion", "emotion2vec_plus_base"]

# Layout relative to the LA data root (the standard ASVspoof2019 LA release).
TRAIN_FLAC_SUBDIR = "ASVspoof2019_LA_train/flac"
TRAIN_PROTOCOL_SUBDIR = "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"


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
def extract_model(
    model_key: str,
    utts: Sequence[Utt],
    flac_dir: Path,
    out_path: Path,
    *,
    config: ExtractionConfig,
    dtype: str,
    device: str,
    log_every: int,
) -> None:
    """Extract pooled all-layer features for ``utts`` with one model and save."""
    logger.info("=== %s -> %s ===", model_key, out_path)
    model = load_model(model_key, device=device)

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

    # Trim pre-allocated rows down to what was actually filled (skips/failures).
    arrays = {f"layer_{k}": buffers[k][:w] for k in layer_keys}
    meta = {
        "ids": np.asarray(ids),
        "speakers": np.asarray(speakers),
        "attacks": np.asarray(attacks),
        "labels": np.asarray(labels),
        "layers": np.asarray(layer_keys, dtype=np.int64),
        "model": np.asarray(model_key),
        "pooling": np.asarray(str(config.pooling)),
        "dtype": np.asarray(dtype),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **arrays, **meta)

    size_gib = out_path.stat().st_size / 2**30
    logger.info(
        "Saved %s: %d utts, %d layers, %s float -> %s (%.2f GiB)%s",
        model_key, w, len(layer_keys), dtype, out_path, size_gib,
        f", {len(failed)} failed" if failed else "",
    )
    if failed:
        fail_log = out_path.with_suffix(".failed.txt")
        fail_log.write_text("\n".join(failed) + "\n", encoding="utf-8")
        logger.info("Wrote %d failed ids -> %s", len(failed), fail_log)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract all-layer SER features over ASVspoof2019 LA train.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data-root", required=True,
        help="ASVspoof2019 LA root containing ASVspoof2019_LA_train/ and "
             "ASVspoof2019_LA_cm_protocols/.",
    )
    p.add_argument(
        "--flac-dir", default=None,
        help=f"Override flac dir (default <data-root>/{TRAIN_FLAC_SUBDIR}).",
    )
    p.add_argument(
        "--protocol", default=None,
        help=f"Override CM protocol (default <data-root>/{TRAIN_PROTOCOL_SUBDIR}).",
    )
    p.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS,
        help="Registered model keys to extract with.",
    )
    p.add_argument(
        "--layers", nargs="+", default=None,
        help="Layer indices to keep (e.g. 0 6 12) or omit for ALL layers.",
    )
    p.add_argument(
        "--pooling", choices=["mean", "max", "mean_std"], default="mean_std",
        help="Temporal pooling applied to every layer.",
    )
    p.add_argument(
        "--dtype", choices=["float16", "float32"], default="float32",
        help="On-disk numeric precision for the feature arrays.",
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
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    data_root = Path(args.data_root)
    flac_dir = Path(args.flac_dir) if args.flac_dir else data_root / TRAIN_FLAC_SUBDIR
    protocol = Path(args.protocol) if args.protocol else data_root / TRAIN_PROTOCOL_SUBDIR
    out_dir = Path(args.out_dir)

    for path, what in [(flac_dir, "flac dir"), (protocol, "protocol")]:
        if not path.exists():
            logger.error("%s not found: %s", what, path)
            return 2

    utts = read_protocol(protocol)
    if args.limit:
        utts = utts[: args.limit]
    n_bona = sum(u.label == "bonafide" for u in utts)
    logger.info(
        "Loaded %d utts from %s (%d bonafide, %d spoof). Models: %s",
        len(utts), protocol.name, n_bona, len(utts) - n_bona, args.models,
    )

    layers = None
    if args.layers:
        layers = [int(x) for x in args.layers]

    config = ExtractionConfig(layers=layers, pooling=args.pooling, save=False)

    for model_key in args.models:
        out_path = out_dir / f"asvspoof_la_train__{model_key}__{args.pooling}.npz"
        if out_path.exists() and not args.overwrite:
            logger.info("Exists, skipping (use --overwrite): %s", out_path)
            continue
        extract_model(
            model_key, utts, flac_dir, out_path,
            config=config, dtype=args.dtype, device=args.device,
            log_every=args.log_every,
        )

    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
