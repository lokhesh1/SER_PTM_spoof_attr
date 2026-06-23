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

import contextlib
import io
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Union

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
# Logging / noise control
# --------------------------------------------------------------------------- #
# Third-party loggers (and bare ``logging.info`` calls -> the "root" logger)
# that flood the console during model download / checkpoint load. We keep their
# WARNING+ records but mute the INFO/DEBUG chatter.
_NOISY_LOGGERS = (
    "root", "funasr", "modelscope", "httpx", "httpcore", "urllib3",
    "filelock", "datasets", "huggingface_hub", "numba", "jieba",
    "matplotlib", "torio", "torchaudio",
)


class _MuteThirdParty(logging.Filter):
    """Drop sub-WARNING records from noisy third-party / root loggers, while
    letting this project's own ``ser.*`` logs through unchanged."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        return record.name.split(".", 1)[0] not in _NOISY_LOGGERS


def configure_logging(verbose: bool = False) -> None:
    """Install a single, filtered stderr handler.

    Our ``ser.*`` loggers report at INFO (or DEBUG with ``verbose``); funasr /
    HF / modelscope import & download chatter is muted below WARNING. Also
    disables the Hugging Face download progress bars. Safe to call once from a
    CLI entry point; replaces any pre-existing root handlers so later
    ``basicConfig`` calls by third-party libs become no-ops.
    """
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Silences the transformers 5.x "LOAD REPORT" block printed when the
    # classifier head is dropped from a *ForSequenceClassification checkpoint.
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    handler.addFilter(_MuteThirdParty())

    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(logging.INFO)
    logging.getLogger("ser").setLevel(logging.DEBUG if verbose else logging.INFO)


@contextlib.contextmanager
def _suppress_stdout():
    """Swallow stdout ``print`` noise (e.g. funasr's "miss key in ckpt"
    warnings) without hiding stderr or exceptions."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


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

    def __init__(self, name: str, hf_id: str, device: str = "cpu",
                 extract_cnn: bool = False):
        self.name = name
        self.hf_id = hf_id
        self.device = device
        # When True, ``forward_layers`` returns the last two CNN feature-encoder
        # layers plus the first two transformer-encoder layers (transformers
        # backend only; ignored by backends without an exposed conv encoder).
        self.extract_cnn = extract_cnn
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
    def forward_layers(
        self, waveform: "torch.Tensor"
    ) -> Dict[Union[int, str], "torch.Tensor"]:
        """Run the model on a 1-D 16 kHz mono ``waveform`` and return
        ``{layer_key: (T, D) tensor}`` for every available layer.

        In the default mode keys are contiguous integers starting at 0 (``0`` is
        the encoder's input embedding, ``1..N`` the transformer block outputs).
        When ``extract_cnn`` is enabled the transformers backend instead returns
        the last two conv feature-encoder layers followed by the first two
        transformer-encoder layers, re-keyed as integers ``0..3`` (``0/1`` =
        ``cnn[-2]/cnn[-1]``, ``2/3`` = transformer input embedding + block 1)."""

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
        self._conv_layers = self._locate_conv_layers()
        if self.extract_cnn and self._conv_layers is None:
            logger.warning(
                "extract_cnn requested but the CNN feature encoder could not be "
                "located on %s (%s); expected model.feature_extractor.conv_layers.",
                self.name, type(self.model).__name__,
            )

    def _locate_conv_layers(self):
        """Return the ``ModuleList`` of conv feature-encoder layers, or ``None``.

        WavLM / wav2vec2 / HuBERT expose them at
        ``model.feature_extractor.conv_layers``; each layer's forward output is a
        ``(B, C, T)`` tensor.
        """
        encoder = getattr(self.model, "feature_extractor", None)
        conv_layers = getattr(encoder, "conv_layers", None)
        if conv_layers is not None and len(conv_layers) > 0:
            return conv_layers
        return None

    @torch.no_grad()
    def forward_layers(
        self, waveform: "torch.Tensor"
    ) -> Dict[Union[int, str], "torch.Tensor"]:
        if self.extract_cnn:
            return self._forward_cnn_layers(waveform)

        inputs = self.processor(
            waveform.numpy(), sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        # hidden_states: tuple of (1, T, D); index 0 = embeddings, 1..N = layers.
        hidden_states = outputs.hidden_states
        return {i: hs.squeeze(0).cpu() for i, hs in enumerate(hidden_states)}

    @torch.no_grad()
    def _forward_cnn_layers(
        self, waveform: "torch.Tensor"
    ) -> Dict[int, "torch.Tensor"]:
        """Return the last two CNN conv-layer outputs plus the first two
        transformer-encoder layers, re-keyed as contiguous integers ``0..3``.

        Conv outputs are captured via forward hooks on
        ``feature_extractor.conv_layers`` (native shape ``(B, C, T)``) and
        transposed to time-major ``(T, C)`` so they share the transformer
        ``(T, D)`` convention and flow through the same downstream pooling.

        Integer keys (so the per-layer files are ``layer_00..layer_03`` and the
        integer-only feature dataloader consumes them unchanged):

            0 -> cnn[-2]  second-to-last conv feature-encoder layer  (512-d)
            1 -> cnn[-1]  last conv feature-encoder layer             (512-d)
            2 -> tf_0     transformer input embedding (hidden_states[0])
            3 -> tf_1     first transformer block output (hidden_states[1])

        The conv layers are 512-d while the transformer layers are the model
        hidden size (768 base / 1024 large); each per-layer file is
        self-contained, so the differing dim across layers is fine.
        """
        if self._conv_layers is None:
            raise RuntimeError(
                f"CNN feature extraction requested but the conv encoder could "
                f"not be located on {self.name} ({type(self.model).__name__}); "
                f"expected model.feature_extractor.conv_layers."
            )

        inputs = self.processor(
            waveform.numpy(), sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        captured: Dict[int, "torch.Tensor"] = {}
        handles = []

        def make_hook(layer_idx: int):
            def hook(_module, _args, output):
                out = output[0] if isinstance(output, (tuple, list)) else output
                captured[layer_idx] = out.detach()
            return hook

        for i, layer in enumerate(self._conv_layers):
            handles.append(layer.register_forward_hook(make_hook(i)))

        try:
            outputs = self.model(**inputs, output_hidden_states=True)
        finally:
            for h in handles:
                h.remove()

        feats: Dict[int, "torch.Tensor"] = {}
        out_idx = 0
        # Last two conv feature-encoder layers: (B, C, T) -> (T, C).
        for i in sorted(captured)[-2:]:
            conv = captured[i].squeeze(0)  # (C, T)
            feats[out_idx] = conv.transpose(0, 1).contiguous().cpu()
            out_idx += 1
        # First two transformer-encoder layers: input embedding + block 1.
        hidden_states = outputs.hidden_states
        for j in range(min(2, len(hidden_states))):
            feats[out_idx] = hidden_states[j].squeeze(0).cpu()  # (T, D)
            out_idx += 1
        return feats


# --------------------------------------------------------------------------- #
# FunASR backend (emotion2vec+)
# --------------------------------------------------------------------------- #
class Emotion2vecModel(BaseSERModel):
    """emotion2vec+ via FunASR, loaded from the Hugging Face mirror.

    FunASR's high-level ``generate`` API only returns a single (final)
    representation, but the underlying data2vec-style encoder computes a hidden
    state per transformer block. We surface all of them by registering forward
    hooks on ``model.model.blocks`` and tapping them during a normal
    ``generate(...)`` call -- this reuses FunASR's own preprocessing instead of
    reimplementing it.

    Layer indexing matches the transformers backend: ``0`` is the encoder's
    input embedding (the tensor fed to the first block) and ``1..N`` are the
    outputs of the N transformer blocks.

    If the encoder structure can't be located (FunASR version differences), we
    fall back to the single final ``feats`` representation keyed ``0``.
    """

    backend = "funasr"

    def _load(self) -> None:
        from funasr import AutoModel as FunASRAutoModel

        # hub="hf" -> download from Hugging Face (emotion2vec/...). Switch to
        # hub="ms" with an "iic/..." id to use the ModelScope mirror instead.
        # _suppress_stdout swallows funasr's "miss key in ckpt" print spam (the
        # data2vec decoder weights are absent by design -- encoder-only model).
        with _suppress_stdout():
            self.model = FunASRAutoModel(model=self.hf_id, hub="hf", disable_update=True)
        self._blocks = self._locate_blocks()
        if self._blocks is None:
            logger.warning(
                "Could not locate emotion2vec transformer blocks; per-layer "
                "extraction is unavailable and only the final representation "
                "(layer 0) will be returned."
            )

    def _locate_blocks(self):
        """Return the ``ModuleList`` of transformer blocks, or ``None``."""
        inner = getattr(self.model, "model", None)
        blocks = getattr(inner, "blocks", None)
        if blocks is not None and len(blocks) > 0:
            return blocks
        return None

    @torch.no_grad()
    def forward_layers(self, waveform: "torch.Tensor") -> Dict[int, "torch.Tensor"]:
        captured: Dict[int, "torch.Tensor"] = {}
        handles = []

        if self._blocks is not None:
            # Pre-hook on block 0 captures the input embeddings -> layer 0.
            def pre_hook(_module, args):
                captured[0] = args[0].detach().clone()

            handles.append(self._blocks[0].register_forward_pre_hook(pre_hook))

            # Forward hook on each block captures its output -> layers 1..N.
            def make_hook(layer_idx: int):
                def hook(_module, _args, output):
                    out = output[0] if isinstance(output, (tuple, list)) else output
                    captured[layer_idx] = out.detach().clone()
                return hook

            for i, block in enumerate(self._blocks):
                handles.append(block.register_forward_hook(make_hook(i + 1)))

        try:
            # granularity="frame" -> frame-level; hooks fire during this call.
            # disable_pbar silences funasr's per-utterance tqdm bar (one bar per
            # file would be 25k bars over the full train set).
            results = self.model.generate(
                waveform.numpy(),
                granularity="frame",
                extract_embedding=True,
                disable_pbar=True,
            )
        finally:
            for h in handles:
                h.remove()

        if captured:
            return {idx: self._to_time_dim(t) for idx, t in sorted(captured.items())}

        # Fallback: high-level final feature only.
        feats = np.asarray(results[0]["feats"])
        if feats.ndim == 1:  # defensive: some versions return (D,)
            feats = feats[None, :]
        return {0: torch.from_numpy(feats).float()}

    @staticmethod
    def _to_time_dim(t: "torch.Tensor") -> "torch.Tensor":
        """Collapse a block tensor to ``(T, D)``, dropping the singleton batch."""
        t = t.detach().cpu().float()
        if t.ndim == 3:  # (B, T, C) or (T, B, C) with batch size 1
            if t.shape[0] == 1:
                t = t[0]
            elif t.shape[1] == 1:
                t = t[:, 0]
            else:  # unexpected; flatten leading dims onto time
                t = t.reshape(-1, t.shape[-1])
        return t


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
    extract_cnn: bool = False,
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
    extract_cnn:
        When ``True``, ``forward_layers`` returns the last two CNN feature-encoder
        layers plus the first two transformer-encoder layers (re-keyed 0..3)
        instead of all transformer layers. Supported by the ``transformers``
        backend only.
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
    if extract_cnn and resolved_backend != "transformers":
        logger.warning(
            "extract_cnn is only supported for the transformers backend; "
            "ignoring it for '%s' (%s backend).", name, resolved_backend
        )
    model = BACKEND_CLASSES[resolved_backend](
        name=name, hf_id=resolved_hf_id, device=device, extract_cnn=extract_cnn
    )
    return model.load()


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device
