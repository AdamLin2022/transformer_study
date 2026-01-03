import torch
import torch.nn as nn
import torch.optim as optim
from my_transformer_v4 import Transformer

def run_fixed_training():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    V_SIZE = 100 
    BATCH_SIZE = 64
    SEQ_LEN = 10
    LR = 0.0005
    EPOCHS = 1000

    model = Transformer(
        enc_voc_size=V_SIZE, dec_voc_size=V_SIZE,
        d_model=128, n_head=4, max_len=50,
        ffn_hidden=256, n_layers=2, drop_prob=0.0, 
        device=device
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    print("\n[开始过拟合测试 - 固定数据集]")
    
    fixed_src = torch.randint(1, V_SIZE, (BATCH_SIZE, SEQ_LEN)).to(device)
    fixed_trg = fixed_src.clone()

    model.train()

    for epoch in range(EPOCHS):
        trg_input = torch.cat([torch.zeros(BATCH_SIZE, 1, device=device, dtype=torch.long), fixed_trg[:, :-1]], dim=1)
        
        output = model(fixed_src, trg_input)
        
        output_flat = output.view(-1, V_SIZE)
        trg_flat = fixed_trg.view(-1)
        loss = criterion(output_flat, trg_flat)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch [{epoch}/{EPOCHS}] Loss: {loss.item():.6f}")

    print("\n[训练结束]")
    
    model.eval()
    with torch.no_grad():
        test_src = fixed_src[0:1]
        test_trg_input = trg_input[0:1]
        
        output = model(test_src, test_trg_input)
        pred = output.argmax(dim=-1)
        
        print("\n[最终检查]")
        print(f"输入: {test_src[0].cpu().numpy()}")
        print(f"预测: {pred[0].cpu().numpy()}")
        
        if torch.equal(test_src, pred):
            print("✅✅✅ 完美过拟合！Loss 极低，说明模型结构完全正确！")
        else:
            print("❌ 还是没背下来，可能代码真有Bug。")

if __name__ == '__main__':
    run_fixed_training()
    