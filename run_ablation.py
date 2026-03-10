"""
Ablation study runner.

Executes all experiment configurations sequentially, logging output
to individual files under the logs/ directory. Covers:
    - Baseline
    - Architecture (depth & width)
    - Activation functions
    - Dropout
    - L2 regularization
    - L1 regularization            (NEW)
    - Batch normalization ablation  (NEW)
    - LR scheduler comparison       (NEW)
    - Early stopping                (NEW)
"""

import os
import subprocess
import time

# Create log directory
os.makedirs("logs", exist_ok=True)

# Base command shared by all experiments
base_cmd = [
    "python", "main.py",
    "--dataset", "mnist",
    "--model", "mlp",
    "--epochs", "15",
]

# ============================================================
# Experiment definitions
# Key = experiment name (used for log filename)
# Value = list of extra CLI arguments
# ============================================================
experiments = {
    # ---- 1. Baseline ----
    "baseline": [
        "--hidden_sizes", "512", "256",
        "--activation", "relu",
        "--dropout", "0.3",
        "--weight_decay", "1e-4",
    ],

    # ---- 2. Architecture (Depth & Width) ----
    "arch_shallow": ["--hidden_sizes", "512", "--dropout", "0.3"],
    "arch_deep": ["--hidden_sizes", "512", "256", "128", "64", "--dropout", "0.3"],
    "arch_narrow": ["--hidden_sizes", "64", "32", "--dropout", "0.3"],
    "arch_wide": ["--hidden_sizes", "1024", "512", "--dropout", "0.3"],

    # ---- 3. Activation Functions ----
    "act_gelu": ["--activation", "gelu"],

    # ---- 4. Dropout Module ----
    "drop_0.0": ["--dropout", "0.0"],
    "drop_0.7": ["--dropout", "0.7"],

    # ---- 5. L2 Regularization (Weight Decay) ----
    "l2_0.0": ["--weight_decay", "0.0"],
    "l2_1e-2": ["--weight_decay", "1e-2"],

    # ---- 6. L1 Regularization (NEW) ----
    "l1_1e-5": ["--l1_lambda", "1e-5"],
    "l1_1e-4": ["--l1_lambda", "1e-4"],
    "l1_1e-3": ["--l1_lambda", "1e-3"],

    # ---- 7. Batch Normalization Ablation (NEW) ----
    "bn_disabled": ["--no_bn"],
    "bn_after_act": ["--bn_position", "after"],

    # ---- 8. LR Scheduler Comparison (NEW) ----
    "sched_cosine": ["--scheduler", "cosine"],
    "sched_plateau": ["--scheduler", "plateau"],
    # baseline already uses --scheduler step (default)

    # ---- 9. Early Stopping (NEW) ----
    "early_stop_p3": ["--early_stop_patience", "3", "--epochs", "30"],
    "early_stop_p5": ["--early_stop_patience", "5", "--epochs", "30"],
}

print("=" * 60)
print("Starting Ablation Studies")
print("=" * 60)
start_time = time.time()

for exp_name, args in experiments.items():
    print(f"\n---> Running experiment: {exp_name}")

    cmd = base_cmd + args
    log_file_path = f"logs/{exp_name}.txt"

    with open(log_file_path, "w") as log_file:
        log_file.write(f"Command: {' '.join(cmd)}\n")
        log_file.write("=" * 50 + "\n")
        subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True)

    print(f"     Finished. Logs -> {log_file_path}")

total_time = (time.time() - start_time) / 60
print(f"\n{'=' * 60}")
print(f"All experiments completed in {total_time:.2f} minutes!")
print(f"{'=' * 60}")
