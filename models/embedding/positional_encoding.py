import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    对应 GitHub: models/embedding/positional_encoding.py
    [优化] 删掉了没用的 device 参数，register_buffer 会自动处理设备移动
    """
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        
        encoding = torch.zeros(max_len, d_model)
        encoding.requires_grad = False  # 显式告诉 PyTorch 不需要梯度

        pos = torch.arange(0, max_len).float().unsqueeze(dim=1)
        _2i = torch.arange(0, d_model, step=2).float()

        encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

        # 核心：注册为 buffer，随模型保存和移动
        self.register_buffer('encoding', encoding)

    def forward(self, x):
        # x shape: [batch_size, seq_len]
        batch_size, seq_len = x.size()
        
        # 返回 shape: [seq_len, d_model]
        # 在加法时，PyTorch 会自动广播成 [batch_size, seq_len, d_model]
        return self.encoding[:seq_len, :]
