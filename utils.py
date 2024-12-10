import os
import random
import numpy as np
import torch
import dgl

def set_seed(seed=2024):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def graph_collate_func(x):
    d, c, y, p = zip(*x)
    d = dgl.batch(d)
    return d, c, y, p


def pearson_cc(z_ij, y_ij):
    z_bar_i = torch.mean(z_ij, dim=1, keepdim=True)
    y_bar_i = torch.mean(y_ij, dim=1, keepdim=True)
    numerator = torch.sum((z_ij - z_bar_i) * (y_ij - y_bar_i), dim=1)
    denominator = torch.sqrt(torch.sum((z_ij - z_bar_i)**2, dim=1)) * torch.sqrt(torch.sum((y_ij - y_bar_i)**2, dim=1))
    pcc = torch.mean(numerator / denominator)
    return pcc
