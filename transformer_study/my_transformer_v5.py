import torch
import torch.nn as nn
import torch.optim as optim
import random
from my_transformer_v4 import Transformer

def run_training():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    V_SIZE = 100 
    BATCH_SIZE = 32
    SEQ_LEN = 10
    LR = 0.0005
    EPOCHS = 3000

    model = Transformer(
        enc_voc_size=V_SIZE, dec_voc_size=V_SIZE,
        d_model=128, n_head=4, max_len=50, 
        ffn_hidden=256, n_layers=2, drop_prob=0.1,
        device=device
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    model.train()

    print("\n[开始训练 - Copy Task]")
    print("任务: 输入 [A, B, C] -> 输出 [A, B, C]")
    
    for epoch in range(EPOCHS):
        data = torch.randint(1, V_SIZE, (BATCH_SIZE, SEQ_LEN)).to(device)
        
        src = data
        trg = data

        trg_input = torch.cat([torch.zeros(BATCH_SIZE, 1, device=device, dtype=torch.long), trg[:, :-1]], dim=1)
        
        output = model(src, trg_input)
        
        output_flat = output.view(-1, V_SIZE)
        trg_flat = trg.view(-1)
        
        loss = criterion(output_flat, trg_flat)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch [{epoch}/{EPOCHS}] Loss: {loss.item():.4f}")

    print("\n[训练结束]")
    
    model.eval() 
    with torch.no_grad():
        test_seq = torch.randint(1, V_SIZE, (1, SEQ_LEN)).to(device)
        test_input = torch.cat([torch.zeros(1, 1, device=device, dtype=torch.long), test_seq[:, :-1]], dim=1)
        
        output = model(test_seq, test_input)
        pred = output.argmax(dim=-1) 
        
        print("\n[测试结果]")
        print(f"输入: {test_seq[0].cpu().numpy()}")
        print(f"预测: {pred[0].cpu().numpy()}")
        
        if torch.equal(test_seq, pred):
            print("✅ 完美复制！模型学会了！")
        else:
            print("⚠️ 还有点误差，可能需要多训练一会儿。")

if __name__ == '__main__':
    run_training()