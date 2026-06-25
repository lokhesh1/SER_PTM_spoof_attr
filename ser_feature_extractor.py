#!/usr/bin/env python3
"""SER feature extraction.

Part of the *spoof speech source attribution* project. Given a model loaded by
:mod:`model_loader`, this module turns audio into features:

    * select a specific layer / set of layers (``--layers 6 9 12`` or ``all``,
      negative indices allowed, e.g. ``-1`` = last layer),
    * optional temporal pooling of the requested layers (``--pooling mean|max|
      mean_std``) -> one vector per layer instead of a ``(T, D)`` matrix,
    * optionally save the extracted features to disk (``--save``).

Model *loading* lives in :mod:`model_loader`; this file only handles audio I/O,
layer selection, pooling, saving and the CLI.

Examples
--------
List registered models::

    python ser_feature_extractor.py --list-models

Extract pooled last-layer features from wavlm-base-emotion and save::

    python ser_feature_extractor.py --model wavlm_base_emotion \
        --audio sample1.wav sample2.flac --layers -1 --pooling mean \
        --save --save-dir feats/

Run with no ``--model`` to be prompted interactively for the model key.

Programmatic use::

    from model_loader import load_model
    from ser_feature_extractor import extract_features, ExtractionConfig

    model = load_model("emotion2vec_plus_large", device="cuda")
    result = extract_features(model, "sample.wav",
                              ExtractionConfig(layers=[-1], pooling="mean"))
    vec = result.features[result.final_layer]   # (D,) numpy array
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch

from model_loader import (
    BACKEND_CLASSES,
    MODEL_REGISTRY,
    TARGET_SAMPLE_RATE,
    BaseSERModel,
    configure_logging,
    load_model,
)

logger = logging.getLogger("ser.feature_extractor")

AudioInput = Union[str, Path, np.ndarray, "torch.Tensor"]


# --------------------------------------------------------------------------- #
# Configuration & result containers
# --------------------------------------------------------------------------- #
@dataclass
class ExtractionConfig:
    """Controls a single feature-extraction call.

    Attributes
    ----------
    layers:
        Layer indices to return. ``None`` -> every available layer. Negative
        indices are resolved relative to the number of available layers
        (``-1`` == last). Layer ``0`` is the encoder's input embedding output;
        layers ``1..N`` are transformer block outputs.
    pooling:
        ``None`` keeps the full ``(T, D)`` sequence; otherwise pool over time to
        a single vector per layer. One of ``mean``, ``max``, ``mean_std``.
    save / save_dir / save_format:
        When ``save`` is true, write one file per audio input into ``save_dir``
        using ``save_format`` (``npz`` | ``pt`` | ``npy``).
    """

    layers: Optional[Sequence[int]] = None
    pooling: Optional[str] = None
    save: bool = False
    save_dir: Optional[Union[str, Path]] = None
    save_format: str = "npz"

    def __post_init__(self) -> None:
        if self.pooling not in (None, "mean", "max", "mean_std"):
            raise ValueError(f"Unknown pooling method: {self.pooling!r}")
        if self.save_format not in ("npz", "pt", "npy"):
            raise ValueError(f"Unknown save format: {self.save_format!r}")


@dataclass
class FeatureResult:
    """Output of one extraction call."""

    model_name: str
    source: str  # file path, or "<array>" for in-memory audio
    # layer key -> (T, D) or (D,) if pooled. Keys are ints in the default mode,
    # or strings ("cnn_*"/"tf_*") when the model runs with extract_cnn enabled.
    features: Dict[Union[int, str], np.ndarray]
    pooled: bool
    sample_rate: int
    saved_to: Optional[str] = None

    @property
    def final_layer(self) -> Union[int, str]:
        return max(self.features)


# --------------------------------------------------------------------------- #
# Audio loading
# --------------------------------------------------------------------------- #
def load_audio(audio: AudioInput, target_sr: int = TARGET_SAMPLE_RATE) -> "torch.Tensor":
    """Return a 1-D mono float32 tensor resampled to ``target_sr``."""
    if isinstance(audio, (str, Path)):
        waveform, sr = _read_file(str(audio))
    elif isinstance(audio, np.ndarray):
        waveform, sr = torch.as_tensor(audio, dtype=torch.float32), target_sr
    elif torch.is_tensor(audio):
        waveform, sr = audio.float(), target_sr
    else:
        raise TypeError(f"Unsupported audio input type: {type(audio)!r}")

    if waveform.ndim == 2:  # (channels, samples) -> mono
        waveform = waveform.mean(dim=0)
    waveform = waveform.reshape(-1)

    if sr != target_sr:
        import torchaudio.functional as AF

        waveform = AF.resample(waveform, sr, target_sr)
    return waveform.float()


def _read_file(path: str):
    """Load an audio file, preferring soundfile (libsndfile) with a torchaudio
    fallback.

    soundfile reads FLAC/WAV/OGG natively and needs no FFmpeg, so it covers the
    whole ASVspoof corpus without the torchaudio/torchcodec backend dance.
    torchaudio is kept only for formats libsndfile can't decode (e.g. mp3/m4a).
    """
    try:
        import soundfile as sf

        data, sr = sf.read(path, dtype="float32", always_2d=True)  # (samples, ch)
        return torch.from_numpy(data.T), sr  # -> (channels, samples)
    except Exception as exc:  # pragma: no cover - depends on installed backends
        logger.debug("soundfile.read failed (%s); falling back to torchaudio", exc)
        import torchaudio

        waveform, sr = torchaudio.load(path)  # (channels, samples)
        return waveform, sr


# --------------------------------------------------------------------------- #
# Pooling, layer selection & saving
# --------------------------------------------------------------------------- #
def pool_over_time(feat: "torch.Tensor", method: str) -> "torch.Tensor":
    """Pool a ``(T, D)`` sequence into a single vector."""
    if method == "mean":
        return feat.mean(dim=0)
    if method == "max":
        return feat.max(dim=0).values
    if method == "mean_std":
        return torch.cat([feat.mean(dim=0), feat.std(dim=0)], dim=-1)
    raise ValueError(f"Unknown pooling method: {method!r}")


def select_layers(
    feats: Dict[Union[int, str], "torch.Tensor"], layers: Optional[Sequence[int]]
) -> Dict[Union[int, str], "torch.Tensor"]:
    """Pick the requested layers, resolving negative indices.

    Integer ``--layers`` selection targets the default integer-keyed layers; the
    string-keyed CNN mode (``cnn_*``/``tf_*``) is intended to be consumed whole,
    so pass ``layers=None`` (the default / ``all``) there.
    """
    if layers is None:
        return dict(feats)
    n = len(feats)
    selected: Dict[Union[int, str], "torch.Tensor"] = {}
    for layer in layers:
        idx = layer if layer >= 0 else n + layer
        if idx not in feats:
            logger.warning(
                "Layer %s (resolved to %s) unavailable; model exposes layers 0..%d. Skipping.",
                layer, idx, n - 1,
            )
            continue
        selected[idx] = feats[idx]
    return selected


def save_features(result: FeatureResult, save_dir: Union[str, Path], fmt: str) -> str:
    """Persist ``result.features`` to ``save_dir`` and return the file path."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(result.source).stem if result.source != "<array>" else "array"
    arrays = {f"layer_{idx}": arr for idx, arr in result.features.items()}

    if fmt == "npz":
        out = save_dir / f"{stem}__{result.model_name}.npz"
        np.savez_compressed(out, **arrays)
    elif fmt == "npy":
        out = save_dir / f"{stem}__{result.model_name}.npy"
        # A dict is stored as a 0-d object array; load with allow_pickle=True.
        np.save(out, arrays, allow_pickle=True)
    elif fmt == "pt":
        out = save_dir / f"{stem}__{result.model_name}.pt"
        torch.save(
            {
                "model_name": result.model_name,
                "source": result.source,
                "pooled": result.pooled,
                "sample_rate": result.sample_rate,
                "features": {idx: torch.from_numpy(a) for idx, a in result.features.items()},
            },
            out,
        )
    else:  # pragma: no cover - guarded by ExtractionConfig
        raise ValueError(f"Unknown save format: {fmt!r}")
    logger.info("Saved features -> %s", out)
    return str(out)


# --------------------------------------------------------------------------- #
# Extraction orchestration
# --------------------------------------------------------------------------- #
def extract_features(
    model: BaseSERModel, audio: AudioInput, config: Optional[ExtractionConfig] = None
) -> FeatureResult:
    """Extract layer features from ``audio`` using a loaded ``model``."""
    config = config or ExtractionConfig()
    source = str(audio) if isinstance(audio, (str, Path)) else "<array>"

    waveform = load_audio(audio, TARGET_SAMPLE_RATE)
    all_layers = model.forward_layers(waveform)
    chosen = select_layers(all_layers, config.layers)
    if not chosen:
        raise RuntimeError(
            f"No layers selected for {model.name}; requested {config.layers}, "
            f"available 0..{len(all_layers) - 1}."
        )

    features: Dict[Union[int, str], np.ndarray] = {}
    for idx, feat in chosen.items():
        if config.pooling is not None:
            feat = pool_over_time(feat, config.pooling)
        features[idx] = feat.detach().cpu().numpy()

    result = FeatureResult(
        model_name=model.name,
        source=source,
        features=features,
        pooled=config.pooling is not None,
        sample_rate=TARGET_SAMPLE_RATE,
    )
    if config.save:
        result.saved_to = save_features(
            result, config.save_dir or "features", config.save_format
        )
    return result


def extract_batch(
    model: BaseSERModel,
    audios: Sequence[AudioInput],
    config: Optional[ExtractionConfig] = None,
) -> List[FeatureResult]:
    return [extract_features(model, a, config) for a in audios]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_layers(values: Optional[List[str]]) -> Optional[List[int]]:
    if not values:
        return None
    if len(values) == 1 and values[0].lower() == "all":
        return None  # None -> every layer
    return [int(v) for v in values]


def _print_models() -> None:
    print("Registered SER models:\n")
    for spec in MODEL_REGISTRY.values():
        print(f"  {spec.key:<24} [{spec.backend:<12}] {spec.hf_id}")
        if spec.description:
            print(f"  {'':<24} {spec.description}")
    print()


def _prompt_for_model() -> str:
    _print_models()
    keys = sorted(MODEL_REGISTRY)
    while True:
        choice = input("Enter model key to load: ").strip()
        if choice in MODEL_REGISTRY:
            return choice
        print(f"  '{choice}' is not a registered key. Options: {keys}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Extract SER layer features from audio.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", help="Registered model key (prompted if omitted).")
    p.add_argument("--hf-id", help="Override the Hugging Face repo id.")
    p.add_argument("--backend", choices=sorted(BACKEND_CLASSES),
                   help="Backend for an unregistered --hf-id (default transformers).")
    p.add_argument("--list-models", action="store_true",
                   help="Print registered models and exit.")
    p.add_argument("--audio", nargs="+", help="Audio file(s) to process.")
    p.add_argument("--layers", nargs="+",
                   help="Layer indices (e.g. 6 9 12 or -1) or 'all'. Default: all.")
    p.add_argument("--cnn-layers", action="store_true",
                   help="Extract every CNN feature-encoder layer plus the first "
                        "two transformer blocks (transformers backend only). "
                        "Yields cnn_*/tf_* keyed features; use with 'all' layers.")
    p.add_argument("--pooling", choices=["mean", "max", "mean_std"],
                   help="Temporal pooling; omit to keep the full (T, D) sequence.")
    p.add_argument("--save", action="store_true", help="Save extracted features.")
    p.add_argument("--save-dir", default="features", help="Output directory.")
    p.add_argument("--save-format", choices=["npz", "pt", "npy"], default="npz")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:0 ...")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    configure_logging(verbose=args.verbose)

    if args.list_models:
        _print_models()
        return 0

    model_key = args.model or _prompt_for_model()
    model = load_model(
        model_key, device=args.device, hf_id=args.hf_id, backend=args.backend,
        extract_cnn=args.cnn_layers,
    )

    if not args.audio:
        logger.info("Model '%s' loaded. No --audio given, nothing to extract.", model_key)
        return 0

    config = ExtractionConfig(
        layers=_parse_layers(args.layers),
        pooling=args.pooling,
        save=args.save,
        save_dir=args.save_dir,
        save_format=args.save_format,
    )
    for audio_path in args.audio:
        result = extract_features(model, audio_path, config)
        shapes = {idx: tuple(arr.shape) for idx, arr in sorted(result.features.items())}
        logger.info(
            "%s -> layers %s | pooled=%s%s",
            audio_path, shapes, result.pooled,
            f" | saved={result.saved_to}" if result.saved_to else "",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
