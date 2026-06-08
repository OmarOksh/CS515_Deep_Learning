"""StockGRU: stacked GRU forecaster, the gated-recurrent analogue of StockLSTM."""
from __future__ import annotations

import torch
import torch.nn as nn


class StockGRU(nn.Module):
    r"""Stacked GRU layers + dropout + linear read-out.

    Implements the GRU recurrence

    .. math::
        z_t = \sigma(W_z[h_{t-1}, x_t] + b_z), \quad
        r_t = \sigma(W_r[h_{t-1}, x_t] + b_r), \\
        \tilde h_t = \tanh(W_h[r_t \odot h_{t-1}, x_t] + b_h), \quad
        h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde h_t,

    through :class:`torch.nn.GRU`. The GRU merges the LSTM's forget/input gates
    into a single update gate and drops the separate cell state, giving fewer
    parameters and often faster, more stable training on short series.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float = 0.2,
        use_conv: bool = False,
        conv_channels: int = 16,
    ) -> None:
        """Initialise the StockGRU.

        Args:
            input_size: Number of input features F per time step.
            hidden_size: GRU hidden dimension H.
            num_layers: Number of stacked GRU layers.
            output_size: Number of outputs D.
            dropout: Dropout probability.
            use_conv: If ``True``, prepend a Conv1d feature extractor.
            conv_channels: Output channels of the optional Conv1d.
        """
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv1d(input_size, conv_channels, kernel_size=3, padding=1)
            gru_in = conv_channels
        else:
            gru_in = input_size

        self.gru = nn.GRU(
            input_size=gru_in,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(batch, T, F)``.

        Returns:
            Predictions of shape ``(batch, output_size)``.
        """
        if self.use_conv:
            x = x.transpose(1, 2)
            x = torch.relu(self.conv(x))
            x = x.transpose(1, 2)

        out, _ = self.gru(x)                    # (B, T, H)
        last = self.dropout(out[:, -1, :])      # (B, H)
        return self.fc(last)                    # (B, output_size)
