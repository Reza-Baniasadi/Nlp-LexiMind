import torch
from torch.utils.data import Dataset
import csv


class TextPairDataset(Dataset):
    def __init__(self, file_path, tokenizer, src_col=0, tgt_col=1, max_len=512):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) <= max(src_col, tgt_col):
                    continue
                src = row[src_col].strip()
                tgt = row[tgt_col].strip()
                self.samples.append((src, tgt))

                
    def __len__(self):
        return len(self.samples)
