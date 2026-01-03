import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    对应 GitHub: models/embedding/positional_encoding.py
    """
    def __init__(self, d_model, max_len, device):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # 我们不训练 PE，它是固定的

        pos = torch.arange(0, max_len, device=device).float().unsqueeze(dim=1)
        _2i = torch.arange(0, d_model, step=2, device=device).float()

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        batch_size, seq_len = x.size()
        return self.encoding[:seq_len, :]