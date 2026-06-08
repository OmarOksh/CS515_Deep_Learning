"""StockLSTM: stacked LSTM forecaster for d-day return prediction (Part b)."""
from __future__ import annotations

import torch
import torch.nn as nn


class StockLSTM(nn.Module):
    r"""One or more stacked LSTM layers + dropout + a linear read-out.

    Implements the recurrence

    .. math::
        f_t = \sigma(W_f[h_{t-1}, x_t] + b_f), \quad
        i_t = \sigma(W_i[h_{t-1}, x_t] + b_i), \\
        \tilde c_t = \tanh(W_c[h_{t-1}, x_t] + b_c), \quad
        c_t = f_t \odot c_{t-1} + i_t \odot \tilde c_t, \\
        o_t = \sigma(W_o[h_{t-1}, x_t] + b_o), \quad
        h_t = o_t \odot \tanh(c_t),

    via :class:`torch.nn.LSTM`. The hidden state of the final time step is passed
    through dropout and a fully-connected layer to produce ``output_size``
    predictions (one per forecast horizon ``d = 1..D``).

    An optional 1-D convolution over the time axis can act as the auxiliary
    feature extractor mentioned in the assignment footnote (e.g. a learned
    moving-average filter) before the recurrent stack.
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
        """Initialise the StockLSTM.

        Args:
            input_size: Number of input features F per time step.
            hidden_size: LSTM hidden dimension H.
            num_layers: Number of stacked LSTM layers.
            output_size: Number of outputs D (consecutive d-day returns).
            dropout: Dropout probability (between LSTM layers and before read-out).
            use_conv: If ``True``, prepend a Conv1d feature extractor.
            conv_channels: Output channels of the optional Conv1d.
        """
        super().__init__()
        self.use_conv = use_conv
        if use_conv:
            self.conv = nn.Conv1d(input_size, conv_channels, kernel_size=3, padding=1)
            lstm_in = conv_channels
        else:
            lstm_in = input_size

        self.lstm = nn.LSTM(
            input_size=lstm_in,
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
            x = x.transpose(1, 2)               # (B, F, T)
            x = torch.relu(self.conv(x))        # (B, C, T)
            x = x.transpose(1, 2)               # (B, T, C)

        out, _ = self.lstm(x)                   # out: (B, T, H)
        last = self.dropout(out[:, -1, :])      # final-step hidden state (B, H)
        return self.fc(last)                    # (B, output_size)
