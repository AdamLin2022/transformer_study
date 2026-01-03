import torch
import torch.nn as nn
import torch.optim as optim
import time
import math
from my_transformer_v8 import Transformer
import data
import conf

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

model = Transformer(
    enc_voc_size=len(data.vocab_src),
    dec_voc_size=len(data.vocab_tgt),
    d_model=conf.d_model,
    n_head=conf.n_heads,
    max_len=conf.max_len,
    ffn_hidden=conf.ffn_hidden,
    n_layers=conf.n_layers,
    drop_prob=conf.drop_prob,
    device=conf.device
).to(conf.device)

print(f'The model has {count_parameters(model):,} trainable parameters')


import os
if os.path.exists('transformer_model.pt'):
    print("检测到存档 transformer_model.pt，正在加载权重...")
    model.load_state_dict(torch.load('transformer_model.pt'))
    print("✅ 成功加载！将从之前的进度继续训练 (Resume Training)。")
else:
    print("未检测到存档，将重新开始训练 (Start from scratch)。")
    model.apply(initialize_weights) 


optimizer = optim.Adam(model.parameters(), lr=conf.init_lr, weight_decay=conf.weight_decay)

# [修改] 删除了 verbose=True，防止报错
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=conf.factor, patience=conf.patience)

criterion = nn.CrossEntropyLoss(ignore_index=data.PAD_IDX)

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    
    for i, (src, trg) in enumerate(iterator):
        src = src.to(conf.device)
        trg = trg.to(conf.device)

        optimizer.zero_grad()

        trg_input = trg[:, :-1]
        output = model(src, trg_input)
        
        output_dim = output.shape[-1]
        trg_label = trg[:, 1:].contiguous().view(-1)
        output_flat = output.contiguous().view(-1, output_dim)
        
        loss = criterion(output_flat, trg_label)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        epoch_loss += loss.item()
        
        if i % 10 == 0:
            print(f"Step: {i} | Loss: {loss.item():.4f}")
        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for i, (src, trg) in enumerate(iterator):
            src = src.to(conf.device)
            trg = trg.to(conf.device)

            trg_input = trg[:, :-1]
            output = model(src, trg_input)
            
            output_dim = output.shape[-1]
            trg_label = trg[:, 1:].contiguous().view(-1)
            output_flat = output.contiguous().view(-1, output_dim)
            
            loss = criterion(output_flat, trg_label)
            epoch_loss += loss.item()
            
    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def run():
    # 稍微调大一点 batch_size 或者让它跑起来
    train_iter = data.make_iter(conf.batch_size, mode='train')
    valid_iter = data.make_iter(conf.batch_size, mode='valid')

    best_valid_loss = float('inf')

    print(f"Start Training on {conf.device}...")
    
    for epoch in range(conf.epoch):
        start_time = time.time()
        
        train_loss = train(model, train_iter, optimizer, criterion, conf.clip)
        valid_loss = evaluate(model, valid_iter, criterion)
        
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        scheduler.step(valid_loss)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'transformer_model.pt')
            print("Model Saved!")
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f}')

if __name__ == '__main__':
    run()