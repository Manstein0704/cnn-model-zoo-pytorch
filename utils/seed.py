import torch
import random
import numpy as np


def set_seed(seed:int):
    random.seed(seed)
    np.random(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.cuda.manual_seed(seed)
    

