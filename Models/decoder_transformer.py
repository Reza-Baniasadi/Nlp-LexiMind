import torch
import torch.nn as nn
import math


class PositionalEncodingSimple(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()