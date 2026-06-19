#!/usr/bin/env python3
"""Train & evaluate classification heads per layer for spoof source attribution.

Part of the *spoof speech source attribution* project. For every
(model x layer x head) combination it:

    1. loads that layer's pooled features (:mod:`feature_dataset`),
    2. trains the chosen head (:mod:`classifier_heads`) on a stratified split,
    3. evaluates on the held-out split (:mod:`probe_metrics`): accuracy,
       per-class accuracy, confusion matrix, binary EER + macro OvR EER,
    4. saves everything under ``results/<model>/<head>/layer_KK/`` and appends a
       row to ``results/summary.csv``.

Reads the per-layer feature directories produced by :mod:`extract_asvspoof`
(default root ``feats_test``); it never modifies them.

Examples
--------
All models / layers / heads, defaults::

    python train_probes.py --features-dir feats_test --results-dir results

One model, two layers, quick smoke test::

    python train_probes.py --features-dir feats_test --models wavlm_base_emotion \
        --layers 0 6 --epochs 3 --results-dir /tmp/results_smoke
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from classifier_heads import HEADS, build_head
from feature_dataset import (
    LayerData,
    balanced_class_weights,
    discover_model_dirs,
    list_layers,
    make_layer_loaders,
    model_key_from_dir,
)
from model_loader import configure_logging
from probe_metrics import evaluate, save_evaluation

logger = logging.getLogger("ser.train_probes")

SUMMARY_COLUMNS = [
    "model", "layer", "head", "n_train", "n_test",
    "accuracy", "binary_eer", "macro_ovr_eer",
]


# --------------------------------------------------------------------------- #
# Train / predict
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_and_predict(
    head_name: str,
    data: LayerData,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: str,
    class_weights: Optional[torch.Tensor],
    head_cfg: Dict,
    seed: int,
) -> np.ndarray:
    """Train one head and return softmax probabilities on the eval split, in the
    same order as ``data.y_test`` (eval loader is not shuffled)."""
    set_seed(seed)
    model = build_head(head_name, data.input_dim, data.num_classes, **head_cfg).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for epoch in range(epochs):
        running, n = 0.0, 0
        for xb, yb in data.train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
            n += xb.size(0)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("    %s epoch %d/%d  loss=%.4f", head_name, epoch + 1, epochs, running / n)

    model.eval()
    probs: List[np.ndarray] = []
    with torch.no_grad():
        for xb, _ in data.test_loader:
            logits = model(xb.to(device))
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probs, axis=0)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(args: argparse.Namespace) -> int:
    device = resolve_device(args.device)
    logger.info("Device: %s", device)

    feature_root = Path(args.features_dir)
    model_dirs = discover_model_dirs(feature_root, args.models)
    if not model_dirs:
        logger.error("No per-layer feature dirs under %s (filter=%s).",
                     feature_root, args.models)
        return 2

    heads = args.heads or list(HEADS)
    head_cfg = dict(
        d_model=args.d_model, d_state=args.d_state,
        n_layers=args.n_layers, n_heads=args.n_heads, dropout=args.dropout,
    )
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    summary_rows: List[Dict] = []
    t_start = time.time()

    for model_dir in model_dirs:
        model_key = model_key_from_dir(model_dir)
        layers = args.layers if args.layers is not None else list_layers(model_dir)
        logger.info("=== %s : layers %s x heads %s ===", model_key, layers, heads)

        for layer in layers:
            data = make_layer_loaders(
                model_dir, layer,
                batch_size=args.batch_size, test_size=args.test_size,
                seed=args.seed, standardize=not args.no_standardize,
                num_workers=args.num_workers, device=device,
            )
            class_weights = (
                None if args.no_class_weight
                else balanced_class_weights(data.train_class_counts)
            )
            for head_name in heads:
                t0 = time.time()
                probs = train_and_predict(
                    head_name, data,
                    epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                    device=device, class_weights=class_weights,
                    head_cfg=head_cfg, seed=args.seed,
                )
                metrics = evaluate(probs, data.y_test, data.class_names)
                out_dir = results_dir / model_key / head_name / f"layer_{layer:02d}"
                save_evaluation(out_dir, metrics, data.class_names)

                summary_rows.append({
                    "model": model_key, "layer": layer, "head": head_name,
                    "n_train": len(data.y_train), "n_test": len(data.y_test),
                    "accuracy": metrics["accuracy"],
                    "binary_eer": metrics["binary_eer"],
                    "macro_ovr_eer": metrics["macro_ovr_eer"],
                })
                logger.info(
                    "  L%02d %-14s acc=%.4f  binEER=%.4f  macroEER=%.4f  (%.1fs)",
                    layer, head_name, metrics["accuracy"], metrics["binary_eer"],
                    metrics["macro_ovr_eer"], time.time() - t0,
                )

    write_summary(results_dir / "summary.csv", summary_rows)
    logger.info("Done: %d runs in %.1f min -> %s",
                len(summary_rows), (time.time() - t_start) / 60.0, results_dir)
    return 0


def write_summary(path: Path, rows: List[Dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=SUMMARY_COLUMNS)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r[k] for k in SUMMARY_COLUMNS})
    logger.info("Wrote summary (%d rows) -> %s", len(rows), path)


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train per-layer attribution probes (3 heads) and save metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--features-dir", default="feats_test",
                   help="Root holding per-layer feature dirs (read-only).")
    p.add_argument("--models", nargs="+", default=None,
                   help="Filter model dirs by substring (default: all found).")
    p.add_argument("--layers", nargs="+", type=int, default=None,
                   help="Layer indices to probe (default: all layers per model).")
    p.add_argument("--heads", nargs="+", choices=sorted(HEADS), default=None,
                   help="Heads to train (default: all three).")
    # training
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--test-size", type=float, default=0.2, help="Eval (=test) fraction.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-standardize", action="store_true",
                   help="Disable per-feature z-scoring (fit on train).")
    p.add_argument("--no-class-weight", action="store_true",
                   help="Disable balanced class weights in the loss.")
    # head hyper-params
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--d-state", type=int, default=64, help="S4D state size.")
    p.add_argument("--n-layers", type=int, default=2, help="S4D block count.")
    p.add_argument("--n-heads", type=int, default=4, help="Attention heads.")
    p.add_argument("--dropout", type=float, default=0.1)
    # io
    p.add_argument("--results-dir", default="results")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:0 ...")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    configure_logging(verbose=args.verbose)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
