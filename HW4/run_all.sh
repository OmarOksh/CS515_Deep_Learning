#!/usr/bin/env bash
#
# run_all.sh — reproduce every HW4 result with one command.
#
# Runs Part 1 (b: returns, c: rolling, d: turning) for both LSTM and GRU,
# then the Part 2 transformer at the homework-fixed T=4 plus a rounds sweep
# that demonstrates the feedback mechanism working (SER falls as rounds rise).
#
# All console output is also written to results_<timestamp>.log so you have a
# clean, attachable record of the run.
#
# Usage:
#   chmod +x run_all.sh        # once, to make it executable
#   ./run_all.sh               # run everything
#
# Notes:
#   * --device cpu is forced: the models are tiny and CPU avoids the flaky
#     MPS backend on Intel Macs with torch 2.2.2.
#   * Steps/epochs use the project defaults. Edit STEPS below to trade
#     runtime for Part 2 convergence.

set -euo pipefail

# Always run from the directory this script lives in.
cd "$(dirname "$0")"

DEVICE="cpu"
STEPS=6000                       # Part 2 training steps at the fixed T=4
SWEEP_STEPS=2000                 # shorter runs for the rounds-sweep comparison
LOG="results_$(date +%Y%m%d_%H%M%S).log"

# Send everything (stdout + stderr) to the terminal AND the log file.
exec > >(tee "$LOG") 2>&1

section () {
  echo ""
  echo "============================================================"
  echo ">>> $1"
  echo "============================================================"
}

echo "HW4 full run — $(date)"
echo "device=$DEVICE  part2_steps=$STEPS  sweep_steps=$SWEEP_STEPS"
echo "logging to: $LOG"

# ----------------------------------------------------------------------
# Part 1b — d-day-ahead return ratios
# ----------------------------------------------------------------------
section "Part 1b | returns | LSTM"
python3 main.py --task returns --model lstm --device "$DEVICE"

section "Part 1b | returns | GRU"
python3 main.py --task returns --model gru  --device "$DEVICE"

# ----------------------------------------------------------------------
# Part 1c — weighted rolling-average target (window l=3)
# ----------------------------------------------------------------------
section "Part 1c | rolling (window=3) | LSTM"
python3 main.py --task rolling --rolling-window 3 --model lstm --device "$DEVICE"

section "Part 1c | rolling (window=3) | GRU"
python3 main.py --task rolling --rolling-window 3 --model gru  --device "$DEVICE"

# ----------------------------------------------------------------------
# Part 1d — turning-point buy/pass detector (bidirectional, gamma=1.1)
# ----------------------------------------------------------------------
section "Part 1d | turning (gamma=1.1) | LSTM"
python3 main.py --task turning --gamma 1.1 --model lstm --device "$DEVICE"

section "Part 1d | turning (gamma=1.1) | GRU"
python3 main.py --task turning --gamma 1.1 --model gru  --device "$DEVICE"

# ----------------------------------------------------------------------
# Part 2 — transformer feedback comms at the homework-fixed T=4
# ----------------------------------------------------------------------
section "Part 2 | rounds=4 sigma^2=0.25 (capacity-limited floor)"
python3 part2.py --rounds 4 --noise-var 0.25 --steps "$STEPS" --device "$DEVICE"

# ----------------------------------------------------------------------
# Part 2 — rounds sweep: proves the feedback/multi-round mechanism works.
# SER should fall steeply as rounds increase (more channel uses => more
# capacity), approaching ~0 by rounds=8.
# ----------------------------------------------------------------------
for R in 1 2 8; do
  section "Part 2 sweep | rounds=$R sigma^2=0.25"
  python3 part2.py --rounds "$R" --noise-var 0.25 --steps "$SWEEP_STEPS" --device "$DEVICE"
done

section "ALL DONE"
echo "Finished — $(date)"
echo "Full transcript saved to: $LOG"
