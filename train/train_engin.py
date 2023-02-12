import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import math
from ..utils.checkpoint_util import save_checkpoint, load_checkpoint
from ..utils.metrics_util import compute_perplexity



class TrainingEngine:
    def __init__(self, model, optimizer, scheduler=None, device="cpu", pad_id=0, ckpt_dir="checkpoints"):
        self.model = model.to(device)
        self.optim = optimizer
        self.sched = scheduler
        self.device = device
        self.pad_id = pad_id
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
        self.ckpt_dir = ckpt_dir
        os.makedirs(ckpt_dir, exist_ok=True)

        def train_one_epoch(self, train_dataset, batch_size=32, grad_clip=None, collate_fn=None):
            self.model.train()
            loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
            pbar = tqdm(loader, desc="train")
            total_loss = 0.0
            steps = 0
            for batch in pbar:
                src = batch["src"].to(self.device)
                tgt = batch["tgt"].to(self.device)