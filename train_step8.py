import torch
import torch.nn as nn
import torch.optim as optim
import conf
# [唯一变化] 导入 v8
from my_transformer_v8 import Transformer 

def run_step8():
    print(f"Testing Step 8 (Modular Embeddings)...")
    # ... (其余代码和 train_step7.py 完全一样，你可以直接复制过来)
    # 只要能跑通，就说明模块拆分成功。
    
    # 简单的验证代码：
    V_SIZE = 1000
    model = Transformer(V_SIZE, V_SIZE, conf.d_model, conf.n_heads, conf.max_len, conf.ffn_hidden, conf.n_layers, conf.drop_prob, conf.device).to(conf.device)
    src = torch.randint(1, V_SIZE, (2, 10)).to(conf.device)
    trg = torch.randint(1, V_SIZE, (2, 10)).to(conf.device)
    out = model(src, trg)
    print("✅ Step 8 模块化重构成功，输出形状:", out.shape)

if __name__ == '__main__':
    run_step8()