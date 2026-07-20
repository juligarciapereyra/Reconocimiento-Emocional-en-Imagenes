import random
import numpy as np
import torch

def set_seed(seed: int):
    """
    Fija una semilla global para garantizar reproducibilidad.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # Para reproducibilidad en operaciones de convolución
    torch.backends.cudnn.benchmark = False