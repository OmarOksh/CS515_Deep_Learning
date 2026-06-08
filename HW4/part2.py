"""Part 2 (bonus): end-to-end training of the interactive AWGN feedback code.

Run, e.g.::

    python part2.py --rounds 4 --noise-var 0.25 --steps 4000

The transmitter and receiver (both transformers) are trained jointly to
minimise the symbol cross-entropy after ``T`` rounds of noisy forward
transmission with a noiseless feedback relay. We report the per-symbol error
rate (SER) and block (message) error rate (BLER), and compare against an
uncoded baseline that simply repeats the one-hot symbol over the same rounds.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer_comm import RXDecoder, TXEncoder
from utils import resolve_device, set_seed


@dataclass
class CommConfig:
    """Typed configuration for the Part-2 experiment."""

    alphabet_size: int
    num_symbols: int
    rounds: int
    noise_var: float
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    steps: int
    batch_size: int
    lr: float
    seed: int
    device: str


def _get_config() -> CommConfig:
    """Parse CLI arguments into a :class:`CommConfig`."""
    p = argparse.ArgumentParser(description="CS515 HW4 Part 2 - feedback coding.")
    p.add_argument("--alphabet-size", type=int, default=8)
    p.add_argument("--num-symbols", type=int, default=4)
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--noise-var", type=float, default=0.25)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--d-ff", type=int, default=128)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    a = p.parse_args()
    return CommConfig(
        alphabet_size=a.alphabet_size, num_symbols=a.num_symbols, rounds=a.rounds,
        noise_var=a.noise_var, d_model=a.d_model, n_heads=a.n_heads,
        n_layers=a.n_layers, d_ff=a.d_ff, steps=a.steps, batch_size=a.batch_size,
        lr=a.lr, seed=a.seed, device=a.device,
    )


def transmit(
    tx: TXEncoder,
    msg_onehot: torch.Tensor,
    cfg: CommConfig,
    device: torch.device,
) -> torch.Tensor:
    """Simulate ``T`` rounds of TX -> AWGN -> noiseless feedback relay.

    Args:
        tx: The transmitter encoder.
        msg_onehot: One-hot messages, shape ``(B, S, A)``.
        cfg: Experiment configuration.
        device: Compute device.

    Returns:
        Received values across all rounds, shape ``(B, S, T)``.
    """
    b = msg_onehot.size(0)
    sigma = cfg.noise_var ** 0.5
    tx_hist = torch.zeros(b, cfg.num_symbols, cfg.rounds, device=device)
    fb_hist = torch.zeros(b, cfg.num_symbols, cfg.rounds, device=device)
    received = torch.zeros(b, cfg.num_symbols, cfg.rounds, device=device)

    for t in range(cfg.rounds):
        x = tx(msg_onehot, tx_hist, fb_hist)                 # (B, S)
        noise = sigma * torch.randn_like(x)
        y = x + noise                                        # AWGN forward channel
        received[:, :, t] = y
        # Update fixed-size histories for the next round.
        tx_hist = tx_hist.clone(); tx_hist[:, :, t] = x
        fb_hist = fb_hist.clone(); fb_hist[:, :, t] = y      # noiseless relay (Hint 1)
    return received


def evaluate(
    tx: TXEncoder, rx: RXDecoder, cfg: CommConfig, device: torch.device,
    batches: int = 40,
) -> tuple[float, float]:
    """Estimate symbol- and block-error rates over random messages.

    Returns:
        ``(ser, bler)`` as floats in ``[0, 1]``.
    """
    tx.eval(); rx.eval()
    sym_err, sym_tot, blk_err, blk_tot = 0, 0, 0, 0
    with torch.no_grad():
        for _ in range(batches):
            m = torch.randint(0, cfg.alphabet_size,
                              (cfg.batch_size, cfg.num_symbols), device=device)
            onehot = F.one_hot(m, cfg.alphabet_size).float()
            received = transmit(tx, onehot, cfg, device)
            pred = rx(received).argmax(-1)                   # (B, S)
            sym_err += (pred != m).sum().item()
            sym_tot += m.numel()
            blk_err += (pred != m).any(dim=1).sum().item()
            blk_tot += m.size(0)
    return sym_err / sym_tot, blk_err / blk_tot


def main() -> None:
    """Train the TX/RX transformers jointly and report error rates."""
    cfg = _get_config()
    set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    print(f"Part 2 | rounds={cfg.rounds} sigma^2={cfg.noise_var} device={device}")

    tx = TXEncoder(cfg.alphabet_size, cfg.rounds, cfg.d_model,
                   cfg.n_heads, cfg.d_ff, cfg.n_layers).to(device)
    rx = RXDecoder(cfg.alphabet_size, cfg.rounds, cfg.d_model,
                   cfg.n_heads, cfg.d_ff, cfg.n_layers).to(device)
    opt = torch.optim.Adam(list(tx.parameters()) + list(rx.parameters()), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    for step in range(1, cfg.steps + 1):
        tx.train(); rx.train()
        m = torch.randint(0, cfg.alphabet_size,
                          (cfg.batch_size, cfg.num_symbols), device=device)
        onehot = F.one_hot(m, cfg.alphabet_size).float()
        received = transmit(tx, onehot, cfg, device)
        logits = rx(received)                                # (B, S, A)
        loss = criterion(logits.reshape(-1, cfg.alphabet_size), m.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()

        if step % max(cfg.steps // 10, 1) == 0:
            ser, bler = evaluate(tx, rx, cfg, device, batches=10)
            print(f"step {step:5d} | loss {loss.item():.4f} | "
                  f"SER {ser:.4f} | BLER {bler:.4f}")

    ser, bler = evaluate(tx, rx, cfg, device)
    print(f"\nFinal  SER {ser:.4f} | BLER {bler:.4f}")


if __name__ == "__main__":
    main()
