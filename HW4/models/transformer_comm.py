"""Transformer encoder/decoder for the interactive AWGN feedback channel (Part 2).

System (see Figure 1 of the assignment)
---------------------------------------
* A transmitter holds a message ``m in {1..8}^4`` (4 symbols).
* For ``T`` rounds the TX encoder emits 4 coded symbols ``x^(t) in R^4`` under an
  average-power constraint ``E||x^(t)||^2 <= 1``.
* Forward channel is AWGN: ``y^(t) = x^(t) + eps``, ``eps ~ N(0, sigma^2 I)``.
* Feedback is a noiseless relay of the received symbols (Hint 1): ``f^(t)=y^(t)``.
* After all ``T`` rounds the RX decoder maps the collected ``{y^(t)}`` to symbol
  estimates ``m_hat`` (Hint 2: the decoder runs only once, at the end).

Both encoder and decoder are transformer-based with an MLP before and after the
transformer module (Hint 3).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def normalize_power(x: torch.Tensor) -> torch.Tensor:
    """Scale a round's symbols so that ``E[||x||^2] = 1`` across the batch.

    The expectation is approximated as the batch mean of the per-sample squared
    norm, matching the average-power constraint ``E||x^(t)||^2 <= 1``.

    Args:
        x: Coded symbols of shape ``(batch, num_symbols)``.

    Returns:
        Power-normalised tensor of the same shape.
    """
    power = x.pow(2).sum(dim=1).mean()       # E[ ||x||^2 ]
    return x / torch.sqrt(power + 1e-9)


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding added to token embeddings."""

    def __init__(self, d_model: int, max_len: int = 16) -> None:
        """Precompute the encoding table.

        Args:
            d_model: Embedding dimension.
            max_len: Maximum number of tokens (here, symbol positions).
        """
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to ``x`` of shape ``(B, L, d_model)``."""
        return x + self.pe[:, : x.size(1)]


class TransformerBlock(nn.Module):
    r"""Standard post-norm transformer block from the assignment.

    .. math::
        H^{(l)} = \mathrm{LayerNorm}(H^{(l-1)} + \mathrm{MultiHead}(H^{(l-1)})) \\
        H^{(l)} = \mathrm{LayerNorm}(H^{(l)} + \mathrm{FFN}(H^{(l)}))
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        """Initialise attention + feed-forward sub-layers.

        Args:
            d_model: Model/embedding dimension.
            n_heads: Number of attention heads.
            d_ff: Hidden dimension of the feed-forward network.
            dropout: Dropout probability.
        """
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Apply self-attention then FFN, each with a post-norm residual."""
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        h = self.norm1(h + attn_out)
        h = self.norm2(h + self.ffn(h))
        return h


class TXEncoder(nn.Module):
    """Transmitter: produces 4 coded symbols per round from history + feedback.

    Each of the 4 symbol positions is a transformer token. A token's raw feature
    vector has a fixed layout (so the input size stays constant across rounds):
    the one-hot original symbol, the round-indexed history of previously
    transmitted symbols, and the round-indexed history of received feedback.
    """

    def __init__(
        self,
        alphabet_size: int = 8,
        rounds: int = 4,
        d_model: int = 64,
        n_heads: int = 4,
        d_ff: int = 128,
        n_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        """Initialise the TX encoder.

        Args:
            alphabet_size: Symbol alphabet size (8 here).
            rounds: Number of communication rounds T.
            d_model: Transformer model dimension.
            n_heads: Attention heads.
            d_ff: Feed-forward hidden size.
            n_layers: Number of transformer blocks.
            dropout: Dropout probability.
        """
        super().__init__()
        self.rounds = rounds
        self.alphabet_size = alphabet_size
        # raw feature = one-hot symbol (A) + tx history (T) + feedback history (T)
        raw_dim = alphabet_size + 2 * rounds
        self.pre_mlp = nn.Sequential(
            nn.Linear(raw_dim, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        )
        self.pos = PositionalEncoding(d_model, max_len=alphabet_size)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.post_mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1)
        )

    def forward(
        self,
        symbols_onehot: torch.Tensor,
        tx_history: torch.Tensor,
        fb_history: torch.Tensor,
    ) -> torch.Tensor:
        """Generate one round of coded symbols.

        Args:
            symbols_onehot: ``(B, 4, alphabet_size)`` one-hot original symbols.
            tx_history: ``(B, 4, T)`` previously transmitted symbols (0 = unused).
            fb_history: ``(B, 4, T)`` feedback received so far (0 = unused).

        Returns:
            Power-normalised coded symbols ``x^(t)`` of shape ``(B, 4)``.
        """
        raw = torch.cat([symbols_onehot, tx_history, fb_history], dim=-1)
        h = self.pre_mlp(raw)                 # (B, 4, d_model)
        h = self.pos(h)
        for blk in self.blocks:
            h = blk(h)
        x = self.post_mlp(h).squeeze(-1)      # (B, 4)
        return normalize_power(x)


class RXDecoder(nn.Module):
    """Receiver: decodes the 4 symbols from all received rounds (runs once)."""

    def __init__(
        self,
        alphabet_size: int = 8,
        rounds: int = 4,
        d_model: int = 64,
        n_heads: int = 4,
        d_ff: int = 128,
        n_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        """Initialise the RX decoder.

        Args mirror :class:`TXEncoder`; the per-token input is the length-``T``
        vector of received values for that symbol position.
        """
        super().__init__()
        self.pre_mlp = nn.Sequential(
            nn.Linear(rounds, d_model), nn.ReLU(), nn.Linear(d_model, d_model)
        )
        self.pos = PositionalEncoding(d_model, max_len=alphabet_size)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.post_mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, alphabet_size)
        )

    def forward(self, received: torch.Tensor) -> torch.Tensor:
        """Decode received symbols into per-position class logits.

        Args:
            received: ``(B, 4, T)`` received values across all rounds.

        Returns:
            Logits of shape ``(B, 4, alphabet_size)``.
        """
        h = self.pre_mlp(received)             # (B, 4, d_model)
        h = self.pos(h)
        for blk in self.blocks:
            h = blk(h)
        return self.post_mlp(h)                # (B, 4, alphabet_size)
