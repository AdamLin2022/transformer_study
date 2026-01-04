import torch
import torch.nn as nn
import torch.optim as optim
import time
import math
from torch.optim.lr_scheduler import LambdaLR
from my_transformer_v8 import Transformer
import data
import conf
# LabelSmoothingLoss 已移除，改用标准 CrossEntropyLoss

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

# [新增] Transformer 标准 Noam 学习率调度器
def get_scheduler(optimizer, warmup_steps, d_model):
    def lr_lambda(step):
        step = max(step, 1)  # 避免除零
        return (d_model ** -0.5) * min(step ** -0.5, step * warmup_steps ** -1.5)
    return LambdaLR(optimizer, lr_lambda)

model = Transformer(
    enc_voc_size=len(data.vocab_src),
    dec_voc_size=len(data.vocab_tgt),
    d_model=conf.d_model,
    n_head=conf.n_heads,
    max_len=conf.max_len,
    ffn_hidden=conf.ffn_hidden,
    n_layers=conf.n_layers,
    drop_prob=conf.drop_prob,
    device=conf.device,
    src_pad_idx=data.PAD_IDX,
    trg_pad_idx=data.PAD_IDX
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


# [修复] 使用正确的 Noam 调度器
# 关键：lr=1.0，因为 lr_lambda 返回的是绝对学习率值
optimizer = optim.Adam(model.parameters(), lr=1.0,
                       betas=(0.9, 0.98), eps=conf.adam_eps,
                       weight_decay=conf.weight_decay)

scheduler = get_scheduler(optimizer, conf.warmup, conf.d_model)

# [回滚] 使用标准 CrossEntropyLoss，避免 Label Smoothing 导致的模式崩塌
criterion = nn.CrossEntropyLoss(ignore_index=data.PAD_IDX)

# 全局 step 计数器（用于学习率调度）
global_step = 0

def train(model, iterator, optimizer, criterion, clip, scheduler):
    global global_step
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
        scheduler.step()  # [修复] 每个 batch 后更新学习率
        global_step += 1
        
        epoch_loss += loss.item()
        
        if i % 10 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Step: {i} | Loss: {loss.item():.4f} | LR: {current_lr:.6f}")
        
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
        
        # [修复] 传入 scheduler 给 train 函数
        train_loss = train(model, train_iter, optimizer, criterion, conf.clip, scheduler)
        valid_loss = evaluate(model, valid_iter, criterion)
        
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        # [移除] 不再使用 ReduceLROnPlateau 的 scheduler.step(valid_loss)
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f}')
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'transformer_model.pt')
            print(f'\tVal. Loss improved, model saved!')
        else:
            print(f'\tVal. Loss did not improve (Best: {best_valid_loss:.3f})')

if __name__ == '__main__':
    run()
