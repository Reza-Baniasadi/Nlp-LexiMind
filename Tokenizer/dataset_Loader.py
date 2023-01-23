import torch
from torch.utils.data import Dataset
import csv



class TextPairDataset(Dataset):
    def __init__(self, file_path, tokenizer, src_col=0, tgt_col=1, max_len=512):
        self.samples = []
