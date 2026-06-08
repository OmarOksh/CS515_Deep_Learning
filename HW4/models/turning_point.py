"""TurningPointDetector: bidirectional recurrent buy/pass classifier (Part d).

The detector emits a *buy* signal when, over the lookback window, the model
predicts that the maximum-price ratio ``p_max^{t+d} / p_t`` will exceed the
threshold ``gamma`` for at least one horizon ``d = 1..D``. Training uses binary
cross-entropy on a single logit, so the forward pass returns raw logits and the
loss is :class:`torch.nn.BCEWithLogitsLoss`.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TurningPointDetector(nn.Module):
    """Bidirectional LSTM/GRU encoder followed by a single-logit head.

    Bidirectionality lets the model condition on the entire lookback window in
    both directions, which helps localise local minima/maxima (turning points)
    that a causal model could only see from one side.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.2,
        cell: str = "lstm",
    ) -> None:
        """Initialise the detector.

        Args:
            input_size: Number of input features F.
            hidden_size: Per-direction hidden dimension H.
            num_layers: Number of stacked recurrent layers.
            dropout: Dropout probability before the read-out.
            cell: ``"lstm"`` or ``"gru"`` recurrent cell type.
        """
        super().__init__()
        rnn_cls = nn.LSTM if cell.lower() == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        # Two directions -> 2 * hidden_size features into the classifier head.
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(batch, T, F)``.

        Returns:
            Raw logits of shape ``(batch,)`` (apply sigmoid for probabilities).
        """
        out, _ = self.rnn(x)                    # (B, T, 2H)
        last = self.dropout(out[:, -1, :])      # (B, 2H)
        return self.fc(last).squeeze(-1)        # (B,)
