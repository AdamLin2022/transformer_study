import torch
import torch.nn as nn

class MockEncoder(nn.Module):
    def __init__(self, d_model, enc_voc_size, max_len, **kwargs):
        super().__init__()
        self.embedding = nn.Embedding(enc_voc_size, d_model)

    def forward(self, x, mask):
        return self.embedding(x)

class MockDecoder(nn.Module):
    def __init__(self, d_model, dec_voc_size, max_len, **kwargs):
        super().__init__()
        self.embedding = nn.Embedding(dec_voc_size, d_model)
        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        x = self.embedding(trg)
        output = self.linear(x)
        return output

class Transformer(nn.Module):
    def __init__(self, src_pad_idx, trg_pad_idx, enc_voc_size, dec_voc_size, d_model, n_head, max_len,
                 ffn_hidden, n_layers, drop_prob, device):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
        self.encoder = MockEncoder(d_model=d_model, enc_voc_size=enc_voc_size, max_len=max_len)
        self.decoder = MockDecoder(d_model=d_model, dec_voc_size=dec_voc_size, max_len=max_len)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_mask)
        
        return output

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {device}")
    
    B, Src_Len, Trg_Len = 2, 10, 12
    D_model = 512
    V_src, V_dec = 1000, 1000
    
    model = Transformer(
        src_pad_idx=0, trg_pad_idx=0, 
        enc_voc_size=V_src, dec_voc_size=V_dec, 
        d_model=D_model, n_head=8, max_len=100,
        ffn_hidden=2048, n_layers=2, drop_prob=0.1,
        device=device
    ).to(device)
    
    src = torch.randint(1, V_src, (B, Src_Len)).to(device)
    trg = torch.randint(1, V_dec, (B, Trg_Len)).to(device)
    
    output = model(src, trg)
    
    print(f"\n[Step 1 验证]")
    print(f"输入形状: src={src.shape}, trg={trg.shape}")
    print(f"输出形状: {output.shape}")
    
    expected_shape = torch.Size([B, Trg_Len, V_dec])
    if output.shape == expected_shape:
        print("✅ 测试通过！Transformer 骨架搭建完成。")
        print("下一步：去实现真正的 Encoder 和 Decoder，替换掉 Mock 类。")
    else:
        print(f"❌ 形状错误: 期望 {expected_shape}, 实际 {output.shape}")