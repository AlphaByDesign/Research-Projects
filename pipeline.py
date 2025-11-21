import random
import numpy as np
import torch

# --- CONFIGURATION ---
MY_SEED = 42

def set_all_seeds(seed):
    """Sets seed for reproducibility across all necessary libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # Set GPU seeds if you plan to use a GPU
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_all_seeds(MY_SEED)