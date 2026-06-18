#!/usr/bin/env python3
"""Modular SER pre-trained-model loader & feature extractor.

Part of the *spoof speech source attribution* project. This module loads a
Speech-Emotion-Recognition (SER) pre-trained model from Hugging Face *on demand*
(only the model requested at run time is instantiated) and exposes it for two
purposes:

    1. Feature extraction  -- the current task.
    2. Fine-tuning         -- the underlying ``nn.Module`` is exposed via
                              ``extractor.model`` together with ``freeze()`` /
                              ``unfreeze()`` helpers, so the very same object can
                              be plugged into a training loop later.

Feature extraction supports:
    * selecting a specific layer / set of layers (``--layers 6 9 12`` or ``all``,
      negative indices allowed, e.g. ``-1`` = last layer),
    * optional temporal pooling of the requested layers (``--pooling mean|max|
      mean_std``) -> one vector per layer instead of a (T, D) matrix,
    * optionally saving the extracted features to disk (``--save``).

Two backends are implemented behind a common interface:
    * ``transformers``  -- WavLM / wav2vec2 / HuBERT etc. Full per-layer access
                           through ``output_hidden_states=True``.
    * ``funasr``        -- emotion2vec+. FunASR's high-level API exposes a single
                           (final) representation; see ``Emotion2vecExtractor``.

Examples
--------
List the models that are registered::

    python ser_feature_extractor.py --list-models

Extract pooled features from the last layer of wavlm-base-emotion and save::

    python ser_feature_extractor.py --model wavlm_base_emotion \
        --audio sample1.wav sample2.flac --layers -1 --pooling mean \
        --save --save-dir feats/

Run with no ``--model`` to be prompted interactively for the model key.

Programmatic use::

    from ser_feature_extractor import load_model, ExtractionConfig
    extractor = load_model("emotion2vec_plus_large", device="cuda")
    result = extractor.extract("sample.wav",
                               ExtractionConfig(layers=[-1], pooling="mean"))
    vec = result.features[result.final_layer]   # (D,) numpy array
"""

from __future__ import annotations

import argparse
import logging
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

try:  # torch / torchaudio are required at run time, not import time, for clarity
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyTorch is required. Install the project requirements: "
        "pip install -r requirements.txt"
    ) from exc


logger = logging.getLogger("ser_feature_extractor")

AudioInput = Union[str, Path, np.ndarray, "torch.Tensor"]
TARGET_SAMPLE_RATE = 16_000  # every registered SER model expects 16 kHz mono


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ModelSpec:
    """Static description of a registered pre-trained model."""

    key: str
    backend: str  # "transformers" | "funasr"
    hf_id: str
    description: str = ""


# Add new models here. ``hf_id`` can always be overridden at load time, so an
# unregistered checkpoint can still be used via ``load_model(name, hf_id=...)``.
MODEL_REGISTRY: Dict[str, ModelSpec] = {
    # --- emotion2vec+ family (FunASR backend) ----------------------------- #
    "emotion2vec_plus_seed": ModelSpec(
        "emotion2vec_plus_seed", "funasr",
        "emotion2vec/emotion2vec_plus_seed",
        "emotion2vec+ seed checkpoint.",
    ),
    "emotion2vec_plus_base": ModelSpec(
        "emotion2vec_plus_base", "funasr",
        "emotion2vec/emotion2vec_plus_base",
        "emotion2vec+ base checkpoint.",
    ),
    "emotion2vec_plus_large": ModelSpec(
        "emotion2vec_plus_large", "funasr",
        "emotion2vec/emotion2vec_plus_large",
        "emotion2vec+ large checkpoint.",
    ),
    # --- transformers backend (WavLM / wav2vec2 / HuBERT) ----------------- #
    "wavlm_base_emotion": ModelSpec(
        "wavlm_base_emotion", "transformers",
        # Override with --hf-id if you target a different emotion checkpoint.
        "microsoft/wavlm-base-plus",
        "WavLM base+ encoder (use --hf-id for an emotion-finetuned variant).",
    ),
    "wav2vec2_emotion": ModelSpec(
        "wav2vec2_emotion", "transformers",
        "superb/wav2vec2-base-superb-er",
        "wav2vec2 base fine-tuned for emotion recognition (SUPERB ER).",
    ),
    "hubert_emotion": ModelSpec(
        "hubert_emotion", "transformers",
        "superb/hubert-large-superb-er",
        "HuBERT large fine-tuned for emotion recognition (SUPERB ER).",
    ),
}


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
    features: Dict[int, np.ndarray]  # layer index -> (T, D) or (D,) if pooled
    pooled: bool
    sample_rate: int
    saved_to: Optional[str] = None

    @property
    def final_layer(self) -> int:
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
    """Load an audio file, preferring torchaudio with a soundfile fallback."""
    try:
        import torchaudio

        waveform, sr = torchaudio.load(path)  # (channels, samples)
        return waveform, sr
    except Exception as exc:  # pragma: no cover - depends on installed backends
        logger.debug("torchaudio.load failed (%s); falling back to soundfile", exc)
        import soundfile as sf

        data, sr = sf.read(path, dtype="float32", always_2d=True)  # (samples, ch)
        return torch.from_numpy(data.T), sr


# --------------------------------------------------------------------------- #
# Pooling & saving helpers
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
    feats: Dict[int, "torch.Tensor"], layers: Optional[Sequence[int]]
) -> Dict[int, "torch.Tensor"]:
    """Pick the requested layers, resolving negative indices."""
    if layers is None:
        return dict(feats)
    n = len(feats)
    selected: Dict[int, "torch.Tensor"] = {}
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
# Base extractor
# --------------------------------------------------------------------------- #
class BaseSERFeatureExtractor(ABC):
    """Common interface for all SER feature extractors."""

    backend: str = "base"

    def __init__(self, name: str, hf_id: str, device: str = "cpu"):
        self.name = name
        self.hf_id = hf_id
        self.device = device
        self.model = None  # underlying nn.Module / FunASR AutoModel
        self._loaded = False

    # -- lifecycle --------------------------------------------------------- #
    def load(self) -> "BaseSERFeatureExtractor":
        if not self._loaded:
            logger.info("Loading %s (%s backend) from '%s' on %s",
                        self.name, self.backend, self.hf_id, self.device)
            self._load()
            self._loaded = True
        return self

    @abstractmethod
    def _load(self) -> None:
        """Instantiate ``self.model`` (and any processor)."""

    @abstractmethod
    def _forward_layers(self, waveform: "torch.Tensor") -> Dict[int, "torch.Tensor"]:
        """Run the model and return ``{layer_index: (T, D) tensor}`` for every
        available layer. Keys must be contiguous starting at 0."""

    # -- feature extraction ------------------------------------------------ #
    def extract(self, audio: AudioInput, config: Optional[ExtractionConfig] = None) -> FeatureResult:
        if not self._loaded:
            self.load()
        config = config or ExtractionConfig()
        source = str(audio) if isinstance(audio, (str, Path)) else "<array>"

        waveform = load_audio(audio, TARGET_SAMPLE_RATE)
        all_layers = self._forward_layers(waveform)
        chosen = select_layers(all_layers, config.layers)
        if not chosen:
            raise RuntimeError(
                f"No layers selected for {self.name}; requested {config.layers}, "
                f"available 0..{len(all_layers) - 1}."
            )

        features: Dict[int, np.ndarray] = {}
        for idx, feat in chosen.items():
            if config.pooling is not None:
                feat = pool_over_time(feat, config.pooling)
            features[idx] = feat.detach().cpu().numpy()

        result = FeatureResult(
            model_name=self.name,
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
        self, audios: Sequence[AudioInput], config: Optional[ExtractionConfig] = None
    ) -> List[FeatureResult]:
        return [self.extract(a, config) for a in audios]

    # -- fine-tuning helpers ----------------------------------------------- #
    def freeze(self) -> None:
        """Freeze all parameters (feature-extraction / probing setups)."""
        if hasattr(self.model, "parameters"):
            for p in self.model.parameters():
                p.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all parameters (full fine-tuning)."""
        if hasattr(self.model, "parameters"):
            for p in self.model.parameters():
                p.requires_grad = True


# --------------------------------------------------------------------------- #
# transformers backend (WavLM / wav2vec2 / HuBERT / ...)
# --------------------------------------------------------------------------- #
class TransformersSERExtractor(BaseSERFeatureExtractor):
    backend = "transformers"

    def _load(self) -> None:
        from transformers import AutoFeatureExtractor, AutoModel

        self.processor = AutoFeatureExtractor.from_pretrained(self.hf_id)
        # AutoModel loads the base encoder even from a *ForSequenceClassification
        # checkpoint, which is what we want for per-layer features.
        self.model = AutoModel.from_pretrained(self.hf_id, output_hidden_states=True)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def _forward_layers(self, waveform: "torch.Tensor") -> Dict[int, "torch.Tensor"]:
        inputs = self.processor(
            waveform.numpy(), sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        # hidden_states: tuple of (1, T, D); index 0 = embeddings, 1..N = layers.
        hidden_states = outputs.hidden_states
        return {i: hs.squeeze(0).cpu() for i, hs in enumerate(hidden_states)}


# --------------------------------------------------------------------------- #
# FunASR backend (emotion2vec+)
# --------------------------------------------------------------------------- #
class Emotion2vecExtractor(BaseSERFeatureExtractor):
    """emotion2vec+ via FunASR.

    NOTE: FunASR's high-level ``generate`` API exposes a single (final)
    representation rather than every transformer block. We therefore return one
    layer keyed ``0`` (frame-level, ``(T, D)``). Requesting other layer indices
    will be skipped with a warning. Per-block extraction would require forward
    hooks on the encoder and is left as a documented extension point
    (``_register_layer_hooks``) for the attribution experiments.
    """

    backend = "funasr"

    def _load(self) -> None:
        from funasr import AutoModel as FunASRAutoModel

        self.model = FunASRAutoModel(model=self.hf_id, hub="hf", disable_update=True)

    @torch.no_grad()
    def _forward_layers(self, waveform: "torch.Tensor") -> Dict[int, "torch.Tensor"]:
        # granularity="frame" -> (T, D); "utterance" would pre-pool. We always
        # request frame-level and let the common code handle pooling uniformly.
        results = self.model.generate(
            waveform.numpy(),
            granularity="frame",
            extract_embedding=True,
        )
        feats = np.asarray(results[0]["feats"])
        if feats.ndim == 1:  # defensive: some versions return (D,)
            feats = feats[None, :]
        return {0: torch.from_numpy(feats).float()}

    def _register_layer_hooks(self):  # pragma: no cover - extension point
        """Placeholder for per-block extraction via forward hooks.

        emotion2vec is a data2vec-style encoder; intermediate block outputs can
        be captured by registering hooks on its transformer layers. Implement
        here if the attribution study needs layer-wise emotion2vec features.
        """
        raise NotImplementedError(
            "Per-layer emotion2vec extraction is not implemented yet."
        )


BACKEND_CLASSES = {
    "transformers": TransformersSERExtractor,
    "funasr": Emotion2vecExtractor,
}


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def load_model(
    name: str,
    device: str = "auto",
    hf_id: Optional[str] = None,
    backend: Optional[str] = None,
) -> BaseSERFeatureExtractor:
    """Instantiate and load the requested model (and only that one).

    Parameters
    ----------
    name:
        A key from :data:`MODEL_REGISTRY`, or an arbitrary identifier when also
        passing ``hf_id`` (and ``backend`` for non-transformers models).
    device:
        ``"auto"`` (default), ``"cpu"``, ``"cuda"``, ``"cuda:0"`` ...
    hf_id:
        Override the Hugging Face repo id for the chosen key, or supply the repo
        id for an unregistered model.
    backend:
        Required when ``name`` is unregistered and not transformers-based.
    """
    device = _resolve_device(device)
    spec = MODEL_REGISTRY.get(name)
    if spec is not None:
        resolved_backend = backend or spec.backend
        resolved_hf_id = hf_id or spec.hf_id
    else:
        if hf_id is None:
            raise KeyError(
                f"Unknown model '{name}'. Choose one of {sorted(MODEL_REGISTRY)} "
                f"or pass hf_id=... to use an unregistered checkpoint."
            )
        resolved_backend = backend or "transformers"
        resolved_hf_id = hf_id
        logger.info("Using unregistered model '%s' (%s).", name, resolved_hf_id)

    if resolved_backend not in BACKEND_CLASSES:
        raise ValueError(
            f"Unknown backend '{resolved_backend}'. Known: {sorted(BACKEND_CLASSES)}"
        )
    extractor = BACKEND_CLASSES[resolved_backend](
        name=name, hf_id=resolved_hf_id, device=device
    )
    return extractor.load()


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


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
        description="Load an SER pre-trained model and extract layer features.",
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
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.list_models:
        _print_models()
        return 0

    model_key = args.model or _prompt_for_model()
    extractor = load_model(
        model_key, device=args.device, hf_id=args.hf_id, backend=args.backend
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
        result = extractor.extract(audio_path, config)
        shapes = {idx: tuple(arr.shape) for idx, arr in sorted(result.features.items())}
        logger.info(
            "%s -> layers %s | pooled=%s%s",
            audio_path, shapes, result.pooled,
            f" | saved={result.saved_to}" if result.saved_to else "",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
