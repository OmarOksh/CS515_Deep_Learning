"""Main entry point for Part 1 (financial forecasting).

Examples
--------
Part b (exact d-day returns, LSTM)::

    python main.py --task returns --model lstm

Part c (weighted rolling-average returns, l=3)::

    python main.py --task rolling --rolling-window 3 --model gru

Part d (turning-point buy/pass signal, bidirectional)::

    python main.py --task turning --gamma 1.1 --model lstm

Add ``--use-synthetic`` to run without internet access to Yahoo Finance.
"""
from __future__ import annotations

import torch.nn as nn

from parameters import Config, get_config
from data.dataset import build_dataloaders
from models import StockGRU, StockLSTM, TurningPointDetector
from train import run_training
from test import run_test
from utils import resolve_device, set_seed


def build_model(cfg: Config) -> nn.Module:
    """Instantiate the model dictated by the task and ``--model`` flag.

    Args:
        cfg: Full run configuration.

    Returns:
        An un-trained model on the CPU (caller moves it to the device).
    """
    m = cfg.model
    if cfg.data.task == "turning":
        # Part d mandates a bidirectional recurrent network.
        return TurningPointDetector(
            input_size=m.input_size,
            hidden_size=m.hidden_size,
            num_layers=m.num_layers,
            dropout=m.dropout,
            cell=m.model_type,
        )

    common = dict(
        input_size=m.input_size,
        hidden_size=m.hidden_size,
        num_layers=m.num_layers,
        output_size=m.output_size,
        dropout=m.dropout,
        use_conv=m.use_conv,
        conv_channels=m.conv_channels,
    )
    return StockLSTM(**common) if m.model_type == "lstm" else StockGRU(**common)


def main() -> None:
    """Parse config, build everything and run train / test according to mode."""
    cfg = get_config()
    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    print(f"Task: {cfg.data.task} | Model: {cfg.model.model_type} | Device: {device}")

    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    model = build_model(cfg).to(device)
    print(model)

    if cfg.mode in ("train", "both"):
        run_training(model, cfg, (train_loader, val_loader), device)
    if cfg.mode in ("test", "both"):
        run_test(model, cfg, test_loader, device)


if __name__ == "__main__":
    main()
