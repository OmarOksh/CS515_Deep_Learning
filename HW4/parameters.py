"""Command-line interface and typed configuration objects.

The homework formatting guidelines ask for two things that this module provides:

* arguments are parsed with :mod:`argparse`;
* the parsed values are packed into small, **individual dataclasses**
  (:class:`DataConfig`, :class:`ModelConfig`, :class:`TrainConfig`) grouped by a
  top-level :class:`Config`, so the rest of the code passes around a typed
  object instead of a loose dictionary.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List


@dataclass
class DataConfig:
    """Controls *what* we model and how sliding windows are produced."""

    tickers: List[str]
    start_date: str
    end_date: str
    train_end: str          # last date (inclusive) of the training split
    val_end: str            # last date (inclusive) of the validation split
    lookback: int           # T: input window length
    horizon: int            # D: number of consecutive d-day returns predicted
    task: str               # "returns" | "rolling" | "turning"
    rolling_window: int     # l: rolling-average window (Part c)
    gamma: float            # buy/pass price-ratio threshold (Part d)
    add_moving_average: bool
    ma_window: int
    use_synthetic: bool     # generate GBM data when yfinance is unavailable

    @property
    def base_features(self) -> List[str]:
        """Raw OHLC feature names pulled from Yahoo Finance."""
        return ["Open", "High", "Low", "Close"]

    @property
    def num_features(self) -> int:
        """Number of input features F (OHLC, plus an optional MA channel)."""
        return len(self.base_features) + (1 if self.add_moving_average else 0)


@dataclass
class ModelConfig:
    """Architecture hyper-parameters for the recurrent forecaster."""

    model_type: str         # "lstm" | "gru"
    input_size: int         # = DataConfig.num_features
    hidden_size: int
    num_layers: int
    dropout: float
    output_size: int        # D for regression tasks, 1 for the turning-point task
    bidirectional: bool     # True is required for the Part d detector
    use_conv: bool          # optional 1D-conv auxiliary feature extractor
    conv_channels: int


@dataclass
class TrainConfig:
    """Optimisation / bookkeeping hyper-parameters."""

    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    optimizer: str          # "adam" | "adamw"
    seed: int
    device: str             # "auto" | "cpu" | "cuda" | "mps"
    save_path: str
    log_interval: int


@dataclass
class Config:
    """Top-level container grouping the three sub-configs."""

    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    mode: str               # "train" | "test" | "both"


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser with sensible homework defaults."""
    p = argparse.ArgumentParser(
        description="CS515 HW4 - LSTM/GRU stock return forecasting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- run control -----------------------------------------------------
    p.add_argument("--mode", choices=["train", "test", "both"], default="both")
    p.add_argument("--task", choices=["returns", "rolling", "turning"],
                   default="returns",
                   help="returns=Part b, rolling=Part c, turning=Part d")

    # --- data ------------------------------------------------------------
    p.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "GOOGL"],
                   help="S&P 500 tickers to download via yfinance.")
    p.add_argument("--start-date", default="2020-01-01")
    p.add_argument("--end-date", default="2025-12-31")
    p.add_argument("--train-end", default="2024-07-31")
    p.add_argument("--val-end", default="2024-12-31")
    p.add_argument("--lookback", type=int, default=20, help="Window length T.")
    p.add_argument("--horizon", type=int, default=5, help="Max horizon D.")
    p.add_argument("--rolling-window", type=int, default=3, help="Window l (Part c).")
    p.add_argument("--gamma", type=float, default=1.1,
                   help="Buy threshold on the max-price ratio (Part d).")
    p.add_argument("--add-moving-average", action=argparse.BooleanOptionalAction,
                   default=True, help="Append a moving-average feature channel.")
    p.add_argument("--ma-window", type=int, default=5)
    p.add_argument("--use-synthetic", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="Use synthetic GBM prices (fallback when offline).")

    # --- model -----------------------------------------------------------
    p.add_argument("--model", dest="model_type", choices=["lstm", "gru"],
                   default="lstm")
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--use-conv", action=argparse.BooleanOptionalAction,
                   default=False, help="Enable the 1D-conv front-end.")
    p.add_argument("--conv-channels", type=int, default=16)

    # --- training --------------------------------------------------------
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", dest="learning_rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--optimizer", choices=["adam", "adamw"], default="adam")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--save-path", default="checkpoints/best_model.pth")
    p.add_argument("--log-interval", type=int, default=50)
    return p


def get_config() -> Config:
    """Parse ``sys.argv`` and assemble the typed :class:`Config`.

    Derived fields (``input_size``, ``output_size``, ``bidirectional``) are
    inferred from the chosen task so the user cannot set them inconsistently.

    Returns:
        A fully populated :class:`Config`.
    """
    args = _build_parser().parse_args()

    data = DataConfig(
        tickers=args.tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        train_end=args.train_end,
        val_end=args.val_end,
        lookback=args.lookback,
        horizon=args.horizon,
        task=args.task,
        rolling_window=args.rolling_window,
        gamma=args.gamma,
        add_moving_average=args.add_moving_average,
        ma_window=args.ma_window,
        use_synthetic=args.use_synthetic,
    )

    # The turning-point task is binary classification (Part d) and the
    # assignment mandates a bidirectional recurrent network for it.
    is_turning = args.task == "turning"
    model = ModelConfig(
        model_type=args.model_type,
        input_size=data.num_features,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        output_size=1 if is_turning else args.horizon,
        bidirectional=is_turning,
        use_conv=args.use_conv,
        conv_channels=args.conv_channels,
    )

    train = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        seed=args.seed,
        device=args.device,
        save_path=args.save_path,
        log_interval=args.log_interval,
    )

    return Config(data=data, model=model, train=train, mode=args.mode)
