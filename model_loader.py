#!/usr/bin/env python3
"""SER pre-trained-model loading.

Part of the *spoof speech source attribution* project. This module is
responsible for one thing: loading a Speech-Emotion-Recognition (SER)
pre-trained model from Hugging Face *on demand* (only the model requested is
instantiated) and exposing it for both feature extraction and fine-tuning.

A loaded model object provides:
    * ``forward_layers(waveform)`` -> ``{layer_index: (T, D) tensor}`` for every
      available layer (consumed by :mod:`ser_feature_extractor`),
    * ``.model`` / ``.processor`` -- the underlying ``nn.Module`` and processor,
      so the same object can be dropped into a training loop,
    * ``freeze()`` / ``unfreeze()`` -- toggle ``requires_grad`` for probing vs.
      full fine-tuning.

Two backends sit behind a common interface (:class:`BaseSERModel`):
    * ``transformers`` -- WavLM / wav2vec2 / HuBERT etc. Full per-layer access
                          via ``output_hidden_states=True``.
    * ``funasr``       -- emotion2vec+ (loaded from the Hugging Face mirror).

Use :func:`load_model` to obtain a ready model::

    from model_loader import load_model
    model = load_model("wavlm_base_emotion", device="cuda")
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyTorch is required. Install the project requirements: "
        "pip install -r requirements.txt"
    ) from exc


logger = logging.getLogger("ser.model_loader")

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
    # --- emotion2vec+ family (FunASR backend, Hugging Face mirror) --------- #
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
        "jihedjabnoun/wavlm-base-emotion",
        "WavLM base fine-tuned for emotion recognition.",
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
# Base model
# --------------------------------------------------------------------------- #
class BaseSERModel(ABC):
    """Common interface for all loaded SER models."""

    backend: str = "base"

    def __init__(self, name: str, hf_id: str, device: str = "cpu"):
        self.name = name
        self.hf_id = hf_id
        self.device = device
        self.model = None  # underlying nn.Module / FunASR AutoModel
        self.processor = None
        self._loaded = False

    # -- lifecycle --------------------------------------------------------- #
    def load(self) -> "BaseSERModel":
        if not self._loaded:
            logger.info("Loading %s (%s backend) from '%s' on %s",
                        self.name, self.backend, self.hf_id, self.device)
            self._load()
            self._loaded = True
        return self

    @abstractmethod
    def _load(self) -> None:
        """Instantiate ``self.model`` (and ``self.processor`` if applicable)."""

    @abstractmethod
    def forward_layers(self, waveform: "torch.Tensor") -> Dict[int, "torch.Tensor"]:
        """Run the model on a 1-D 16 kHz mono ``waveform`` and return
        ``{layer_index: (T, D) tensor}`` for every available layer. Keys must be
        contiguous starting at 0."""

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
class TransformersSERModel(BaseSERModel):
    backend = "transformers"

    def _load(self) -> None:
        from transformers import AutoFeatureExtractor, AutoModel, Wav2Vec2FeatureExtractor

        try:
            self.processor = AutoFeatureExtractor.from_pretrained(self.hf_id)
        except (OSError, ValueError) as exc:
            # Some *ForSequenceClassification checkpoints ship no preprocessor
            # config; fall back to the standard 16 kHz raw-waveform extractor.
            logger.warning("No feature-extractor config in '%s' (%s); using a "
                           "default 16 kHz Wav2Vec2FeatureExtractor.", self.hf_id, exc)
            self.processor = Wav2Vec2FeatureExtractor(
                sampling_rate=TARGET_SAMPLE_RATE, do_normalize=True,
                return_attention_mask=True,
            )
        # AutoModel loads the base encoder even from a *ForSequenceClassification
        # checkpoint, which is what we want for per-layer features.
        self.model = AutoModel.from_pretrained(self.hf_id, output_hidden_states=True)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def forward_layers(self, waveform: "torch.Tensor") -> Dict[int, "torch.Tensor"]:
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
class Emotion2vecModel(BaseSERModel):
    """emotion2vec+ via FunASR, loaded from the Hugging Face mirror.

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

        # hub="hf" -> download from Hugging Face (emotion2vec/...). Switch to
        # hub="ms" with an "iic/..." id to use the ModelScope mirror instead.
        self.model = FunASRAutoModel(model=self.hf_id, hub="hf", disable_update=True)

    @torch.no_grad()
    def forward_layers(self, waveform: "torch.Tensor") -> Dict[int, "torch.Tensor"]:
        # granularity="frame" -> (T, D); "utterance" would pre-pool. We always
        # request frame-level and let the caller handle pooling uniformly.
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
    "transformers": TransformersSERModel,
    "funasr": Emotion2vecModel,
}


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def load_model(
    name: str,
    device: str = "auto",
    hf_id: Optional[str] = None,
    backend: Optional[str] = None,
) -> BaseSERModel:
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
    device = resolve_device(device)
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
    model = BACKEND_CLASSES[resolved_backend](
        name=name, hf_id=resolved_hf_id, device=device
    )
    return model.load()


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device
