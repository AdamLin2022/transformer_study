import torch
import numpy as np
import matplotlib.pyplot as plt

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:x.size(1), :]

d_model = 128
max_len = 100
pe_layer = PositionalEncoding(d_model, max_len)

# 造一个假数据输入 [Batch=1, Len=50, Dim=128]
x = torch.zeros(1, 50, 128)
output_pe = pe_layer(x)


# 实验一：可视化频率 (验证不同维度的正弦波频率不同)

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1) # 第一张图
y_high_freq = output_pe[:, 0].numpy()  
y_low_freq = output_pe[:, 64].numpy()  

plt.plot(y_high_freq, label='Dim 0 (High Freq)', alpha=0.7)
plt.plot(y_low_freq, label='Dim 64 (Low Freq)', linewidth=3)
plt.title("Experiment 1: Frequency Comparison (High vs Low dimensions)")
plt.legend()
plt.grid(True)


# 实验二：可视化内积 (验证位置越近越相似)

plt.subplot(2, 1, 2) 

pos_5 = output_pe[5] 
scores = torch.matmul(output_pe, pos_5)

plt.plot(scores.numpy())
plt.axvline(x=5, color='r', linestyle='--', label='Self (Pos 5)')
plt.title("Experiment 2: Dot Product Similarity (Focus on Pos 5)")
plt.xlabel("Position Index")
plt.ylabel("Similarity Score")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('result.png')