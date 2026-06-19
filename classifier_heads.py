#!/usr/bin/env python3
"""Classification heads for layer-wise spoof-source attribution probing.

Part of the *spoof speech source attribution* project. Each head maps one
pooled layer feature vector ``(B, input_dim)`` to ``(B, num_classes)`` logits.
Per the project's current setup the input is a single 768-d mean-pooled vector,
which the heads treat as a length-``input_dim`` 1-channel sequence -- i.e. the
conv / state-space / attention operators run *along the feature axis*.

Three heads (selectable by key in :data:`HEADS`):

    * ``cnn_pool``       -- Conv1d stack + global statistics pooling.
    * ``s4``             -- stacked S4D (diagonal structured state space), a
                            self-contained implementation (no external dep,
                            kernel built with ``torch.fft``).
    * ``cnn_attention``  -- strided Conv1d down-sampling + multi-head self
                            attention + attentive pooling.

Build one with :func:`build_head`.
"""

from __future__ import annotations

import math
from typing import Dict, Type

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Head 1: CNN + global statistics pooling
# --------------------------------------------------------------------------- #
class CNNPoolHead(nn.Module):
    """Conv1d feature extractor followed by mean+std pooling over the sequence."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        channels=(64, 128, 128),
        kernel_size: int = 5,
        dropout: float = 0.1,
        **_,
    ):
        super().__init__()
        layers = []
        c_in = 1
        for c in channels:
            layers += [
                nn.Conv1d(c_in, c, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(c),
                nn.GELU(),
                nn.MaxPool1d(2),
            ]
            c_in = c
        self.conv = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(c_in * 2, num_classes)  # *2 for mean||std pooling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.unsqueeze(1)             # (B, 1, L)
        u = self.conv(u)              # (B, C, L')
        pooled = torch.cat([u.mean(dim=-1), u.std(dim=-1)], dim=-1)  # (B, 2C)
        return self.head(self.dropout(pooled))


# --------------------------------------------------------------------------- #
# Head 2: S4D (diagonal structured state space)
# --------------------------------------------------------------------------- #
class S4DKernel(nn.Module):
    """Minimal diagonal SSM kernel (S4D). Computes a length-``L`` convolution
    kernel per channel from a complex diagonal state matrix ``A``. Follows the
    standalone S4D formulation (conjugate-symmetric, ``N/2`` complex states)."""

    def __init__(self, d_model: int, n_state: int = 64,
                 dt_min: float = 1e-3, dt_max: float = 1e-1):
        super().__init__()
        h, n = d_model, n_state // 2
        log_dt = torch.rand(h) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)
        # C complex, stored as (..., 2) real/imag pairs.
        self.C = nn.Parameter(torch.view_as_real(torch.randn(h, n, dtype=torch.cfloat)))
        # S4D-Lin init: A = -1/2 + i*pi*n.
        self.log_A_real = nn.Parameter(torch.log(0.5 * torch.ones(h, n)))
        self.A_imag = nn.Parameter(math.pi * torch.arange(n).unsqueeze(0).repeat(h, 1).float())

    def forward(self, L: int) -> torch.Tensor:
        dt = torch.exp(self.log_dt)                       # (H,)
        C = torch.view_as_complex(self.C)                 # (H, N)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (H, N)
        dtA = A * dt.unsqueeze(-1)                         # (H, N)
        steps = torch.arange(L, device=A.device)          # (L,)
        K = dtA.unsqueeze(-1) * steps                      # (H, N, L)
        C = C * (torch.exp(dtA) - 1.0) / A                 # discretised C (ZOH, B=1)
        K = 2.0 * torch.einsum("hn,hnl->hl", C, torch.exp(K)).real  # (H, L)
        return K


class S4D(nn.Module):
    """One S4D layer: diagonal SSM convolution + skip + gated output."""

    def __init__(self, d_model: int, n_state: int = 64, dropout: float = 0.0, **kw):
        super().__init__()
        self.D = nn.Parameter(torch.randn(d_model))
        self.kernel = S4DKernel(d_model, n_state=n_state, **kw)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.output_linear = nn.Sequential(
            nn.Conv1d(d_model, 2 * d_model, 1), nn.GLU(dim=-2)
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:  # u: (B, H, L)
        L = u.size(-1)
        k = self.kernel(L)                                # (H, L)
        k_f = torch.fft.rfft(k, n=2 * L)                  # (H, L+1)
        u_f = torch.fft.rfft(u, n=2 * L)                  # (B, H, L+1)
        y = torch.fft.irfft(u_f * k_f, n=2 * L)[..., :L]  # (B, H, L)
        y = y + u * self.D.unsqueeze(-1)
        y = self.dropout(self.activation(y))
        return self.output_linear(y)                      # (B, H, L)


class S4Head(nn.Module):
    """Stacked S4D blocks (pre-norm residual) + mean pooling + linear classifier."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        d_model: int = 128,
        d_state: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
        **_,
    ):
        super().__init__()
        self.encoder = nn.Linear(1, d_model)
        self.s4_layers = nn.ModuleList(
            [S4D(d_model, n_state=d_state, dropout=dropout) for _ in range(n_layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.encoder(x.unsqueeze(-1))         # (B, L, d_model)
        for s4, norm in zip(self.s4_layers, self.norms):
            z = norm(u).transpose(-1, -2)         # (B, d_model, L)
            z = s4(z).transpose(-1, -2)           # (B, L, d_model)
            u = u + self.dropout(z)               # residual
        u = self.out_norm(u).mean(dim=1)          # pool over sequence
        return self.head(u)


# --------------------------------------------------------------------------- #
# Head 3: CNN down-sample + self-attention + attentive pooling
# --------------------------------------------------------------------------- #
class CNNAttentionHead(nn.Module):
    """Strided Conv1d down-sampling, multi-head self-attention, attentive pool."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        d_model: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1,
        **_,
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, d_model, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(d_model), nn.GELU(),
        )
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.pool_w = nn.Linear(d_model, 1)       # attentive pooling weights
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.conv(x.unsqueeze(1))             # (B, d_model, L')
        u = u.transpose(1, 2)                     # (B, L', d_model)
        a, _ = self.attn(u, u, u)
        u = self.norm(u + self.dropout(a))        # residual + norm
        w = torch.softmax(self.pool_w(u), dim=1)  # (B, L', 1)
        pooled = (u * w).sum(dim=1)               # (B, d_model)
        return self.head(pooled)


# --------------------------------------------------------------------------- #
# Registry / factory
# --------------------------------------------------------------------------- #
HEADS: Dict[str, Type[nn.Module]] = {
    "cnn_pool": CNNPoolHead,
    "s4": S4Head,
    "cnn_attention": CNNAttentionHead,
}


def build_head(name: str, input_dim: int, num_classes: int, **kwargs) -> nn.Module:
    """Instantiate a head by key. Extra ``kwargs`` are forwarded; each head
    ignores those it doesn't use, so a single config dict drives all three."""
    if name not in HEADS:
        raise KeyError(f"Unknown head '{name}'. Choose from {sorted(HEADS)}.")
    return HEADS[name](input_dim=input_dim, num_classes=num_classes, **kwargs)
