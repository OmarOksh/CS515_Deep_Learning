"""Entry point for HW3: Robustness and Adversarial Samples.

This module delegates to robustness.py which contains all experiment logic.
Run directly with:

    python main.py --device cuda --epochs 10 --batch_size 128

Or run robustness.py directly for the same effect.
"""

from __future__ import annotations

import random
import ssl

import numpy as np
import torch

ssl._create_default_https_context = ssl._create_unverified_context


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    set_seed(42)
    # Import and run the main experiment pipeline
    from robustness import main
    main()
