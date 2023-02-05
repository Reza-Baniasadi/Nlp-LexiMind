import torch
import torch.nn as nn
import math


class PositionalEncodingSimple(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]
    

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=6, dim_feedforward=2048, max_len=512, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncodingSimple(d_model, max_len=max_len)
        decoder_layer = nn.TransformerDecoderLayer(d_model, n_heads, dim_feedforward, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size, bias=False)
        self._d_model = d_model


    def _generate_square_subsequent_mask(self, sz, device):
        mask = torch.triu(torch.ones((sz, sz), device=device) * float('-inf'), diagonal=1)
        return mask
    

    def forward(self, input_ids, memory=None):
        device = input_ids.device
        emb = self.tok_emb(input_ids) * math.sqrt(self._d_model)
        emb = self.pos_enc(emb)
        seq_len = input_ids.size(1)
        tgt_mask = self._generate_square_subsequent_mask(seq_len, device)