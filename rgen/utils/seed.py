from __future__ import annotations
import os
import random
import numpy as np
import torch

def set_seed(seed: int = 0, deterministic: bool = True) -> None:
    """
    Set all relevant RNG seeds. If `deterministic` is True,
    enforce deterministic CuDNN kernels (may reduce speed).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
