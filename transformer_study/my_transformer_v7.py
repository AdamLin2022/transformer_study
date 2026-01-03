import torch
import torch.nn as nn
import torch.optim as optim
from my_transformer_v4 import Transformer
import conf

def run_config_training():
    print(f"Training on: {conf.device}")

    V_SIZE = 1000 
    
    model = Transformer(
        enc_voc_size=V_SIZE, 
        dec_voc_size=V_SIZE,
        d_model=conf.d_model,
        n_head=conf.n_heads,
        max_len=conf.max_len,
        ffn_hidden=conf.ffn_hidden,
        n_layers=conf.n_layers,
        drop_prob=conf.drop_prob,
        device=conf.device
    ).to(conf.device)

    optimizer = optim.Adam(model.parameters(), lr=conf.init_lr, weight_decay=conf.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    print(f"\n[模型配置已加载]")
    print(f"层数: {conf.n_layers}, 维度: {conf.d_model}, 头数: {conf.n_heads}")
    print(f"Batch Size: {conf.batch_size}, Epochs: {conf.epoch}")
    
    model.train()
    
    fixed_src = torch.randint(1, V_SIZE, (conf.batch_size, 10)).to(conf.device)
    fixed_trg = fixed_src.clone()

    for i in range(100):
        trg_input = torch.cat([torch.zeros(conf.batch_size, 1, device=conf.device, dtype=torch.long), fixed_trg[:, :-1]], dim=1)
        
        output = model(fixed_src, trg_input)
        
        output_flat = output.view(-1, V_SIZE)
        trg_flat = fixed_trg.view(-1)
        loss = criterion(output_flat, trg_flat)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Step [{i}] Loss: {loss.item():.6f}")

if __name__ == '__main__':
    run_config_training()