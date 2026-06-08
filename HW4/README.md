# CS515 Deep Learning — Homework 4

Sequence modeling for **financial forecasting** (Part 1, LSTM/GRU) and an
end-to-end **transformer feedback communication code** (Part 2, bonus).

A full write-up of the methodology and results is in
[`report/CS515-Omar-ALAKSH-38192-HW4.pdf`](report/CS515-Omar-ALAKSH-38192-HW4.pdf).

## Repository layout

```
HW4/
├── main.py                  # Part 1 entry point (returns / rolling / turning)
├── part2.py                 # Part 2 entry point (feedback comm system)
├── parameters.py            # argparse + typed dataclass configs
├── train.py                 # shared training loop (MSE / BCE)
├── test.py                  # evaluation + metric reporting
├── utils.py                 # seeding, scaler, metrics, power constraint
├── run_all.sh               # reproduce every result with one command
├── data/
│   └── dataset.py           # yfinance download / synthetic, windows, splits
├── models/
│   ├── lstm.py              # StockLSTM            (Part b)
│   ├── gru.py               # StockGRU            (Part b)
│   ├── turning_point.py     # TurningPointDetector (Part d, bidirectional)
│   └── transformer_comm.py  # TXEncoder / RXDecoder (Part 2)
├── report/                  # LaTeX source, figures and compiled PDF
├── results_20260607_121721.log   # full reference run log
├── requirements.txt
└── README.md
```

The code follows the requested conventions: `argparse`, the
`main/train/test/parameters` + `models/` structure, **type hints**, **docstrings**
on every class and function, and **dataclasses** for argument passing
(`DataConfig`, `ModelConfig`, `TrainConfig`, `Config`, `CommConfig`).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Part 1 — Financial forecasting

Daily OHLC data for the chosen tickers is downloaded with `yfinance` over
2020-01-01 … 2025-12-31 and split **chronologically**: train ≤ 2024-07-31,
validation 2024-08 … 2024-12, test 2025-01 … 2025-12. If Yahoo Finance is
unreachable, pass `--use-synthetic` to fall back to geometric-Brownian-motion
prices so the full pipeline still runs.

```bash
# (b) exact d-day returns, d = 1..5
python main.py --task returns --model lstm
python main.py --task returns --model gru

# (c) weighted rolling-average returns, window l = 3
python main.py --task rolling --rolling-window 3 --model gru

# (d) turning-point buy/pass detector, bidirectional, gamma = 1.1
python main.py --task turning --gamma 1.1 --model lstm
```

Useful flags: `--lookback 20`, `--horizon 5`, `--hidden-size`, `--num-layers`,
`--use-conv` (1-D conv front-end / learned moving average), `--optimizer adamw`,
`--epochs`, `--device cpu`, `--use-synthetic`.

### Design choices
- **Features (F):** Open, High, Low, Close, plus an optional moving-average
  channel (`--add-moving-average`, on by default). A 1-D-conv auxiliary-feature
  front end is available via `--use-conv`.
- **Normalisation:** a per-feature z-score `StandardScaler` is fit **only on the
  training segment** and applied to val/test, so no future statistics leak. The
  return targets are computed from **raw** prices (returns are already scale-free).
- **Windows:** built **inside** each chronological segment, which guarantees a
  training window never peeks at validation/test prices through its `t+d` targets.
- **Loss / optimiser:** MSE for (b)/(c), `BCEWithLogitsLoss` for (d); Adam/AdamW,
  gradient clipping at norm 5, `ReduceLROnPlateau` on the validation loss.

## Results (real `yfinance` data: AAPL, MSFT, GOOGL)

Reproduced by `run_all.sh`; full transcript in `results_20260607_121721.log`.
Dataset windows: train 3372, val 246, test 675.

### (b) Multi-horizon returns — test MSE (×10⁻³)

| Horizon | LSTM  | GRU   |
|--------:|:-----:|:-----:|
| d=1     | 0.362 | 0.362 |
| d=2     | 0.710 | 0.711 |
| d=3     | 1.138 | 1.139 |
| d=4     | 1.490 | 1.490 |
| d=5     | 1.792 | 1.789 |
| **all** | **1.099** | **1.098** |

Error grows monotonically with the horizon, as expected; LSTM and GRU are
essentially tied on this weak signal.

### (c) Weighted rolling-average target — does it stabilise training?

**Yes.** With the same model and seed, the rolling target (l=3) gives a lower
test MSE at **every** horizon, and a **44.1 % lower overall MSE** than the raw
return target:

| Target            | overall MSE (×10⁻³) | overall MAE (×10⁻²) |
|-------------------|:-------------------:|:-------------------:|
| Raw return (b)    | 1.099               | 2.273               |
| Rolling avg. (c)  | **0.614**           | **1.615**           |

The weighted average suppresses single-day noise, producing a smoother,
lower-variance objective; the trade-off is that the model now predicts a
denoised trend rather than the exact day-`t+d` return.

### (d) Turning-point detection (γ = 1.1, positive rate 0.046, threshold 0.5)

| Model | Acc.  | Prec. | Rec.  | F1    | pos. wt. |
|-------|:-----:|:-----:|:-----:|:-----:|:--------:|
| LSTM  | 0.944 | 0.231 | 0.097 | 0.136 | 28.84    |
| GRU   | 0.954 | 0.500 | 0.065 | 0.114 | 28.84    |

Two practical notes:
1. **Threshold interpretation.** Read literally as a return ratio, `gamma = 1.1`
   means a **+110 %** move in ≤5 days, which essentially never occurs. We apply
   the threshold to the **price ratio** `p_max/p_t > gamma` (i.e. a **+10 %**
   target). `--gamma` is configurable if your grader intends the literal form.
2. **Class imbalance.** Buy events are rare, so unweighted BCE collapses to
   "always pass." We pass `pos_weight = #neg/#pos ≈ 28.84` to
   `BCEWithLogitsLoss`. With the threshold fixed at 0.5 the detector is
   high-precision / low-recall; lowering the threshold trades precision for
   recall.

## Part 2 (bonus) — transformer feedback code

```bash
python part2.py --rounds 4 --noise-var 0.25 --steps 6000
```

A transmitter and receiver (both transformers, MLP→transformer→MLP per Hint 3)
are trained end-to-end. Each round the TX emits 4 power-constrained coded symbols
(`E||x^(t)||² = 1`); the forward channel is AWGN (`σ² = 0.25`); feedback is a
noiseless relay of the received symbols (Hint 1). The decoder runs once at the
end (Hint 2) and outputs 8-way logits per symbol, trained with cross-entropy.

### Error rate vs. interaction rounds (σ² = 0.25)

| Rounds T | SER     | BLER    |
|---------:|:-------:|:-------:|
| 1        | 0.709   | 0.994   |
| 2        | 0.595   | 0.975   |
| **4**    | **0.314** | **0.778** |
| 8        | 0.0004  | 0.0018  |

SER falls monotonically as rounds increase, confirming the feedback/multi-round
mechanism works. **The saturation at the assignment-fixed T=4 is a fundamental
capacity limit, not a training failure:** per-symbol power is 1/4 = 0.25 against
σ² = 0.25, i.e. 0 dB SNR, so each AWGN use carries ½·log₂(1+SNR) = 0.5 bit. Over
4 symbols × T rounds that is 2T bits, while the message is 4 symbols × log₂8 = 12
bits. Reliable transmission needs 2T ≥ 12, i.e. **T ≥ 6**. Hence T=4 (8 bits)
floors at SER ≈ 0.31, while T=8 (16 bits) drives SER to ~10⁻⁴. (Feedback does not
raise the capacity of a memoryless AWGN channel, so this bound is unconditional.)

## Reproduce everything

```bash
chmod +x run_all.sh
./run_all.sh                # all of Part 1 (LSTM+GRU) + Part 2 + rounds sweep
```

## Troubleshooting

### `RuntimeError: Numpy is not available` (Intel Mac)

The last PyTorch with Intel-macOS wheels is **2.2.2**, compiled against NumPy
1.x. If NumPy 2 gets installed, torch crashes on `torch.from_numpy`. Fix:

```bash
pip install "numpy<2" "pandas<3"
```

The pinned `requirements.txt` already enforces this.

### Force CPU (avoid flaky MPS on Intel Macs)

`--device auto` may select MPS, which is unreliable on Intel Macs with torch
2.2.2. For this small workload CPU is equally fast:

```bash
python main.py --task returns --model lstm --device cpu
python part2.py --rounds 4 --device cpu
```
