#!/usr/bin/env python3
"""Per-clip SER feature extraction over the MLAAD v5 dataset (source tracing).

Companion to :mod:`extract_asvspoof`, kept **separate** (that file is untouched).
It reuses the same dataset-agnostic core -- :func:`model_loader.load_model` and
:func:`ser_feature_extractor.extract_features` -- and only swaps the dataset
source and the on-disk layout.

What it extracts (per the project's MLAAD source-tracing setup):

* encoder: ``hubert_emotion`` (HuBERT-large), ``mean`` pooling;
* the ``extract_cnn`` path is run once per clip and **two** of its four layers
  are kept -- the last conv layer and the transformer input embedding:

      cnn-mode key 1 = cnn[-1]  last conv feature-encoder layer  (512-d)
      cnn-mode key 2 = tf_0      = hidden_states[0]               (1024-d)

Layout -- **one ``.npz`` per clip**, mirroring the MLAAD ``fake/`` tree, with
each layer in its own top-level folder ``feat_<encoder>_<suffix>/``::

    feat_hubert_emotion_00/   # <- cnn[-1]   (key 1)
        fake/<lang>/<model_dir>/<file>.npz
    feat_hubert_emotion_01/   # <- tf_0      (key 2)
        fake/<lang>/<model_dir>/<file>.npz

Each ``.npz`` holds ``features (D,)`` plus ``path`` / ``model_name`` /
``language`` / ``layer`` / ``source`` / ``encoder`` / ``pooling``. Extraction is
**split-agnostic** -- the train/val/test partition is computed separately by
:mod:`protocols_mlaad` from the clip ``path``.

Examples
--------
Smoke-test on 20 clips, CPU::

    python extract_mlaad.py --limit 20 --device cpu

Full pool B (resumable -- re-run to fill in anything that failed)::

    python extract_mlaad.py --device auto
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from model_loader import configure_logging, load_model
from ser_feature_extractor import ExtractionConfig, extract_features
from protocols_mlaad import ENCODER_DEFAULT, MLAAD_ROOT_DEFAULT, scan_pool

logger = logging.getLogger("ser.extract_mlaad")

# Output folder suffix -> the cnn-mode layer key it stores (see module docstring).
#   key 1 = cnn[-1] last conv (512-d);  key 2 = tf_0 = hidden_states[0] (1024-d)
LAYER_DIRS: Dict[str, int] = {"00": 1, "01": 2}
LAYER_SOURCE: Dict[str, str] = {"00": "cnn[-1]", "01": "tf_0"}


def _out_path(out_dir: Path, encoder: str, suffix: str, rel: str) -> Path:
    return out_dir / f"feat_{encoder}_{suffix}" / Path(rel).with_suffix(".npz")


def extract_pool(
    *, root: Path, out_dir: Path, encoder: str, pooling: str, device: str,
    limit: Optional[int], overwrite: bool, log_every: int,
) -> None:
    recs = scan_pool(root)
    if limit:
        recs = recs[:limit]
    n = len(recs)
    logger.info("MLAAD pool: %d clips under %s -> %s (encoder=%s, pooling=%s)",
                n, root, out_dir, encoder, pooling)

    model = load_model(encoder, device=device, extract_cnn=True)
    config = ExtractionConfig(layers=None, pooling=pooling)  # all cnn-mode keys; we pick 1 & 2

    done = skipped = 0
    failed: List[str] = []
    t0 = time.time()
    for i, (rel, model_name, language) in enumerate(recs):
        outs = {sfx: _out_path(out_dir, encoder, sfx, rel) for sfx in LAYER_DIRS}
        if not overwrite and all(p.exists() for p in outs.values()):
            skipped += 1
        else:
            abs_path = root / rel
            try:
                result = extract_features(model, abs_path, config)
            except Exception as exc:  # one bad clip shouldn't kill a 154k-clip run
                logger.warning("Extraction failed for %s (%s); skipping.", rel, exc)
                failed.append(rel)
                continue
            ok = True
            for sfx, key in LAYER_DIRS.items():
                if key not in result.features:
                    logger.warning("%s: cnn-mode key %d absent (got %s); skipping clip.",
                                   rel, key, sorted(result.features))
                    ok = False
                    break
            if not ok:
                failed.append(rel)
                continue
            for sfx, key in LAYER_DIRS.items():
                outs[sfx].parent.mkdir(parents=True, exist_ok=True)
                np.savez(
                    outs[sfx],
                    features=result.features[key].astype(np.float32, copy=False),
                    layer=np.asarray(int(sfx)),
                    source=np.asarray(LAYER_SOURCE[sfx]),
                    path=np.asarray(rel),
                    model_name=np.asarray(model_name),
                    language=np.asarray(language),
                    encoder=np.asarray(encoder),
                    pooling=np.asarray(pooling),
                )
            done += 1

        if (i + 1) % log_every == 0 or (i + 1) == n:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0.0
            eta = (n - (i + 1)) / rate if rate > 0 else float("nan")
            logger.info("%d/%d  (%.1f clip/s, ETA %.1f min, %d new, %d skipped, %d failed)",
                        i + 1, n, rate, eta / 60.0, done, skipped, len(failed))

    logger.info("Done: %d extracted, %d skipped (already present), %d failed.",
                done, skipped, len(failed))
    if failed:
        fail_log = out_dir / f"feat_{encoder}.failed.txt"
        fail_log.parent.mkdir(parents=True, exist_ok=True)
        fail_log.write_text("\n".join(failed) + "\n", encoding="utf-8")
        logger.info("Wrote %d failed paths -> %s", len(failed), fail_log)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract per-clip MLAAD features (last conv + tf_0), mirroring "
                    "the dataset tree under feat_<encoder>_<layer>/.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mlaad-root", default=str(MLAAD_ROOT_DEFAULT),
                   help="MLAAD v5 root containing fake/<lang>/<model>/.")
    p.add_argument("--out-dir", default="feats_mlaad",
                   help="Parent dir; feat_<encoder>_<layer>/ folders are created inside it.")
    p.add_argument("--model", default=ENCODER_DEFAULT, help="Registered encoder key.")
    p.add_argument("--pooling", choices=["mean", "max", "mean_std"], default="mean")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:0 ...")
    p.add_argument("--limit", type=int, default=None, help="Process only the first N clips (smoke test).")
    p.add_argument("--overwrite", action="store_true", help="Re-extract even if the .npz exists.")
    p.add_argument("--log-every", type=int, default=500, help="Progress log interval (clips).")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    configure_logging(verbose=args.verbose)
    extract_pool(
        root=Path(args.mlaad_root), out_dir=Path(args.out_dir), encoder=args.model,
        pooling=args.pooling, device=args.device, limit=args.limit,
        overwrite=args.overwrite, log_every=args.log_every,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
