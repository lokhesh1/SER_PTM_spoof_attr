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
    BONAFIDE_INDEX,
    LayerData,
    balanced_class_weights,
    discover_model_dirs,
    list_layers,
    make_layer_loaders,
    model_key_from_dir,
)
from model_loader import configure_logging
from probe_metrics import evaluate, save_evaluation
from protocols import (
    PROTOCOL_FOLDS,
    PROTOCOL_SUBSETS,
    discover_feature_groups,
    make_protocol_loaders,
    protocol_layers,
)
from protocols_mlaad import (
    MLAAD_ROOT_DEFAULT,
    build_ratio_split,
    load_manifest,
    make_mlaad_loaders,
)

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
    same order as ``data.y_test`` (eval loader is not shuffled).

    When ``data`` carries a protocol-defined ``val_loader`` (see
    :class:`protocols.ProtocolData`), the epoch with the best validation accuracy
    is restored before predicting the test set; otherwise the final-epoch model
    is used (the single-subset :class:`~feature_dataset.LayerData` path)."""
    set_seed(seed)
    model = build_head(head_name, data.input_dim, data.num_classes, **head_cfg).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    val_loader = getattr(data, "val_loader", None)
    best_val, best_state = -1.0, None

    for epoch in range(epochs):
        model.train()
        running, n = 0.0, 0
        for xb, yb in data.train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
            n += xb.size(0)
        if val_loader is not None:
            val_acc = _accuracy(model, val_loader, device)
            if val_acc >= best_val:
                best_val = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if logger.isEnabledFor(logging.DEBUG):
            msg = "    %s epoch %d/%d  loss=%.4f" % (head_name, epoch + 1, epochs, running / n)
            if val_loader is not None:
                msg += "  val_acc=%.4f" % best_val
            logger.debug(msg)

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    probs: List[np.ndarray] = []
    with torch.no_grad():
        for xb, _ in data.test_loader:
            logits = model(xb.to(device))
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(probs, axis=0)


def _accuracy(model: nn.Module, loader, device: str) -> float:
    """Top-1 accuracy of ``model`` over a (non-shuffled) loader."""
    model.eval()
    correct, n = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb.to(device)).argmax(dim=1).cpu()
            correct += int((pred == yb).sum())
            n += yb.size(0)
    return correct / max(n, 1)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _train_heads_on(
    data,
    model_tag: str,
    layer: int,
    *,
    heads: Sequence[str],
    head_cfg: Dict,
    args: argparse.Namespace,
    device: str,
    results_dir: Path,
    summary_rows: List[Dict],
) -> None:
    """Train every head on one (model, layer) ``data`` bundle, save per-head
    metrics under ``results_dir/<model_tag>/<head>/layer_KK/`` and append summary
    rows. Works for both :class:`feature_dataset.LayerData` and
    :class:`protocols.ProtocolData` (the latter adds a val set + bonafide index)."""
    class_weights = (
        None if args.no_class_weight else balanced_class_weights(data.train_class_counts)
    )
    bonafide_index = getattr(data, "bonafide_index", BONAFIDE_INDEX)
    for head_name in heads:
        t0 = time.time()
        probs = train_and_predict(
            head_name, data,
            epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
            device=device, class_weights=class_weights,
            head_cfg=head_cfg, seed=args.seed,
        )
        metrics = evaluate(probs, data.y_test, data.class_names,
                           bonafide_index=bonafide_index)
        out_dir = results_dir / model_tag / head_name / f"layer_{layer:02d}"
        save_evaluation(out_dir, metrics, data.class_names)

        summary_rows.append({
            "model": model_tag, "layer": layer, "head": head_name,
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


def _run_cv_layer(
    protocol: str, feature_root: Path, model: str, pooling: str, cnn: bool,
    layer: int, model_tag: str, *,
    heads: Sequence[str], head_cfg: Dict, args: argparse.Namespace, device: str,
    results_dir: Path, summary_rows: List[Dict], n_folds: int,
) -> None:
    """Cross-validation for one (model, layer): train every head on every fold,
    save each fold's metrics under ``.../layer_KK/fold_F/``, then write a
    ``cv_metrics.json`` (mean +/- std + per-fold) per head and a mean summary row."""
    per_head: Dict[str, Dict[str, List[float]]] = {
        h: {"accuracy": [], "binary_eer": [], "macro_ovr_eer": []} for h in heads
    }
    n_train = n_test = 0
    for fold in range(n_folds):
        data = make_protocol_loaders(
            feature_root, protocol, model=model, pooling=pooling, layer=layer,
            cnn=cnn, fold=fold, n_folds=n_folds, batch_size=args.batch_size,
            seed=args.seed, standardize=not args.no_standardize,
            num_workers=args.num_workers, device=device,
        )
        n_train, n_test = len(data.y_train), len(data.y_test)
        class_weights = (
            None if args.no_class_weight else balanced_class_weights(data.train_class_counts)
        )
        for head_name in heads:
            t0 = time.time()
            probs = train_and_predict(
                head_name, data,
                epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                device=device, class_weights=class_weights,
                head_cfg=head_cfg, seed=args.seed,
            )
            metrics = evaluate(probs, data.y_test, data.class_names,
                               bonafide_index=data.bonafide_index)
            out_dir = results_dir / model_tag / head_name / f"layer_{layer:02d}" / f"fold_{fold}"
            save_evaluation(out_dir, metrics, data.class_names)
            for key in per_head[head_name]:
                per_head[head_name][key].append(metrics[key])
            logger.info(
                "  L%02d %-14s fold %d/%d acc=%.4f  (%.1fs)",
                layer, head_name, fold + 1, n_folds, metrics["accuracy"], time.time() - t0,
            )

    for head_name in heads:
        vals = {k: np.asarray(v, dtype=float) for k, v in per_head[head_name].items()}
        agg = {"n_folds": n_folds}
        for k, v in vals.items():
            agg[f"{k}_mean"] = float(np.nanmean(v))
            agg[f"{k}_std"] = float(np.nanstd(v))
            agg[f"per_fold_{k}"] = v.tolist()
        out_dir = results_dir / model_tag / head_name / f"layer_{layer:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "cv_metrics.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")

        summary_rows.append({
            "model": model_tag, "layer": layer, "head": head_name,
            "n_train": n_train, "n_test": n_test,
            "accuracy": agg["accuracy_mean"],
            "binary_eer": agg["binary_eer_mean"],
            "macro_ovr_eer": agg["macro_ovr_eer_mean"],
        })
        logger.info(
            "  L%02d %-14s CV%d acc=%.4f+/-%.4f  binEER=%.4f+/-%.4f  macroEER=%.4f+/-%.4f",
            layer, head_name, n_folds,
            agg["accuracy_mean"], agg["accuracy_std"],
            agg["binary_eer_mean"], agg["binary_eer_std"],
            agg["macro_ovr_eer_mean"], agg["macro_ovr_eer_std"],
        )


def run(args: argparse.Namespace) -> int:
    device = resolve_device(args.device)
    logger.info("Device: %s", device)

    feature_root = Path(args.features_dir)
    heads = args.heads or list(HEADS)
    head_cfg = dict(
        d_model=args.d_model, d_state=args.d_state,
        n_layers=args.n_layers, n_heads=args.n_heads, dropout=args.dropout,
    )

    if args.mlaad:
        return _run_mlaad(args, device, feature_root, heads, head_cfg)

    if args.protocol:
        return _run_protocol(args, device, feature_root, heads, head_cfg)

    model_dirs = discover_model_dirs(feature_root, args.models)
    if not model_dirs:
        logger.error("No per-layer feature dirs under %s (filter=%s).",
                     feature_root, args.models)
        return 2

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
            _train_heads_on(
                data, model_key, layer, heads=heads, head_cfg=head_cfg, args=args,
                device=device, results_dir=results_dir, summary_rows=summary_rows,
            )

    write_summary(results_dir / "summary.csv", summary_rows)
    logger.info("Done: %d runs in %.1f min -> %s",
                len(summary_rows), (time.time() - t_start) / 60.0, results_dir)
    return 0


def _run_protocol(
    args: argparse.Namespace, device: str, feature_root: Path,
    heads: Sequence[str], head_cfg: Dict,
) -> int:
    """Protocol mode: build train/val/test per :mod:`protocols` and write results
    under ``<results-dir>/<protocol>/<model>[__cnn]/<head>/layer_KK/``."""
    protocol = args.protocol
    groups = discover_feature_groups(feature_root, args.models)
    if not groups:
        logger.error("No feature dirs under %s (filter=%s).", feature_root, args.models)
        return 2

    results_dir = Path(args.results_dir) / protocol
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    summary_rows: List[Dict] = []
    t_start = time.time()

    for (model, pooling, cnn), subset_layers in groups.items():
        avail = protocol_layers(subset_layers, protocol)
        if not avail:
            logger.warning(
                "[%s] skipping %s (%s%s): need subsets %s, have %s.",
                protocol, model, pooling, ", cnn" if cnn else "",
                PROTOCOL_SUBSETS[protocol], sorted(subset_layers),
            )
            continue
        layers = ([l for l in args.layers if l in avail]
                  if args.layers is not None else avail)
        if not layers:
            continue
        model_tag = f"{model}__cnn" if cnn else model
        logger.info("=== [%s] %s (%s%s): layers %s x heads %s ===",
                    protocol, model, pooling, ", cnn" if cnn else "", layers, heads)
        for layer in layers:
            if protocol in PROTOCOL_FOLDS:
                _run_cv_layer(
                    protocol, feature_root, model, pooling, cnn, layer, model_tag,
                    heads=heads, head_cfg=head_cfg, args=args, device=device,
                    results_dir=results_dir, summary_rows=summary_rows,
                    n_folds=args.cv_folds,
                )
                continue
            data = make_protocol_loaders(
                feature_root, protocol, model=model, pooling=pooling, layer=layer,
                cnn=cnn, batch_size=args.batch_size, seed=args.seed,
                standardize=not args.no_standardize, num_workers=args.num_workers,
                device=device,
            )
            _train_heads_on(
                data, model_tag, layer, heads=heads, head_cfg=head_cfg, args=args,
                device=device, results_dir=results_dir, summary_rows=summary_rows,
            )

    if not summary_rows:
        logger.error("No runnable feature groups for protocol '%s' under %s "
                     "(are the required subsets extracted?).", protocol, feature_root)
        return 2

    write_summary(results_dir / "summary.csv", summary_rows)
    logger.info("Done [%s]: %d runs in %.1f min -> %s",
                protocol, len(summary_rows), (time.time() - t_start) / 60.0, results_dir)
    return 0


def _parse_mlaad_dir(name: str) -> Optional[tuple]:
    """``feat_hubert_emotion_00`` -> ``("hubert_emotion", 0)``; ``None`` if the
    directory name isn't a ``feat_<encoder>_<layer>`` feature tree."""
    if not name.startswith("feat_"):
        return None
    encoder, _, suffix = name[len("feat_"):].rpartition("_")
    if not encoder or not suffix.isdigit():
        return None
    return encoder, int(suffix)


def _run_mlaad(
    args: argparse.Namespace, device: str, feature_root: Path,
    heads: Sequence[str], head_cfg: Dict,
) -> int:
    """MLAAD mode: closed-set source-tracing (82 generators) over the ratio split
    in :mod:`protocols_mlaad`. Each ``feat_<encoder>_<layer>/`` dir under
    ``--features-dir`` is one layer; the existing per-head training/eval path is
    reused unchanged. Results land under ``<results-dir>/mlaad/<encoder>/<head>/
    layer_KK/``."""
    if args.mlaad_manifest:
        split = load_manifest(args.mlaad_manifest)
        logger.info("MLAAD split loaded from manifest %s (%d clips).",
                    args.mlaad_manifest, len(split.rel_paths))
    else:
        split = build_ratio_split(args.mlaad_root, seed=args.seed)
        logger.info("MLAAD split built fresh (seed %d) from %s.", args.seed, args.mlaad_root)

    layer_dirs = sorted(d for d in feature_root.glob("feat_*_*") if d.is_dir())
    if args.models:
        layer_dirs = [d for d in layer_dirs if any(m in d.name for m in args.models)]
    if not layer_dirs:
        logger.error("No feat_<encoder>_<layer>/ dirs under %s (filter=%s). "
                     "Run extract_mlaad.py first.", feature_root, args.models)
        return 2

    results_dir = Path(args.results_dir) / "mlaad"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    summary_rows: List[Dict] = []
    t_start = time.time()
    for layer_dir in layer_dirs:
        parsed = _parse_mlaad_dir(layer_dir.name)
        if parsed is None:
            logger.warning("Skipping unrecognized dir name: %s", layer_dir.name)
            continue
        encoder, layer = parsed
        if args.layers is not None and layer not in args.layers:
            continue
        logger.info("=== [mlaad] %s layer %02d x heads %s ===", encoder, layer, heads)
        data = make_mlaad_loaders(
            layer_dir, split, batch_size=args.batch_size,
            standardize=not args.no_standardize, num_workers=args.num_workers,
            device=device,
        )
        _train_heads_on(
            data, encoder, layer, heads=heads, head_cfg=head_cfg, args=args,
            device=device, results_dir=results_dir, summary_rows=summary_rows,
        )

    if not summary_rows:
        logger.error("No runnable MLAAD layer dirs under %s.", feature_root)
        return 2
    write_summary(results_dir / "summary.csv", summary_rows)
    logger.info("Done [mlaad]: %d runs in %.1f min -> %s",
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
    p.add_argument("--protocol", choices=sorted(PROTOCOL_SUBSETS), default=None,
                   help="Custom train/val/test protocol (csr1|csr2|attr2|attr17|cv5). "
                        "Builds splits across subsets via protocols.py and writes "
                        "results under <results-dir>/<protocol>/. Omit for the "
                        "default single-subset stratified split. --test-size is "
                        "ignored in protocol mode (splits are protocol-defined).")
    p.add_argument("--cv-folds", type=int, default=5,
                   help="Number of folds for cross-validation protocols (cv5). "
                        "Ignored by single-split protocols.")
    p.add_argument("--mlaad", action="store_true",
                   help="MLAAD source-tracing mode: closed-set 82-class attribution "
                        "over the protocols_mlaad ratio split (seed via --seed). "
                        "--features-dir must hold the feat_<encoder>_<layer>/ trees "
                        "from extract_mlaad.py; results go to <results-dir>/mlaad/. "
                        "Ignores --protocol/--test-size.")
    p.add_argument("--mlaad-manifest", default=None,
                   help="MLAAD split manifest CSV (protocols_mlaad --write-manifest). "
                        "If omitted, the split is rebuilt from --mlaad-root with --seed.")
    p.add_argument("--mlaad-root", default=str(MLAAD_ROOT_DEFAULT),
                   help="MLAAD v5 root, used to rebuild the split when no manifest given.")
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
