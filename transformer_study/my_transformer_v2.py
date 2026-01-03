import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class MockEncoderLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.identity_map = nn.Linear(d_model, d_model)

    def forward(self, x, src_mask):
        return self.identity_map(x)

class Encoder(nn.Module):
    def __init__(self, d_model, n_head, max_len, ffn_hidden, enc_voc_size, drop_prob, n_layers, device):
        super().__init__()
        
        self.emb = nn.Embedding(enc_voc_size, d_model)
        self.pe = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(drop_prob)
        
        self.layers = nn.ModuleList([
            MockEncoderLayer(d_model=d_model) for _ in range(n_layers)
        ])

    def forward(self, x, src_mask):
        x = self.emb(x) * math.sqrt(x.size(-1))
        x = self.pe(x)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, src_mask)
            
        return x

class MockDecoder(nn.Module):
    def __init__(self, d_model, dec_voc_size, **kwargs):
        super().__init__()
        self.linear = nn.Linear(d_model, dec_voc_size)
    def forward(self, trg, enc_src, trg_mask, src_mask):
        batch, len_trg = trg.shape
        return self.linear(enc_src[:, :len_trg, :])

class Transformer(nn.Module):
    def __init__(self, enc_voc_size, dec_voc_size, d_model, n_head, max_len, ffn_hidden, n_layers, drop_prob, device):
        super().__init__()
        self.device = device
        
        self.encoder = Encoder(d_model, n_head, max_len, ffn_hidden, enc_voc_size, drop_prob, n_layers, device)
        self.decoder = MockDecoder(d_model, dec_voc_size)
        
        self.src_pad_idx = 0
        self.trg_pad_idx = 0

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        return None

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {device}")
    
    B, Len = 2, 10
    D_model = 512
    N_Layers = 6
    
    model = Transformer(
        enc_voc_size=1000, dec_voc_size=1000,
        d_model=D_model, n_head=8, max_len=100,
        ffn_hidden=2048, n_layers=N_Layers, drop_prob=0.1,
        device=device
    ).to(device)
    
    src = torch.randint(1, 1000, (B, Len)).to(device)
    trg = torch.randint(1, 1000, (B, Len)).to(device)
    
    output = model(src, trg)
    
    print(f"\n[Step 2 验证: Encoder 堆叠]")
    print(f"Encoder 层数: {len(model.encoder.layers)}")
    print(f"输入形状: {src.shape}")
    print(f"输出形状: {output.shape}")
    
    pe_device = model.encoder.pe.pe.device
    print(f"Positional Encoding Device: {pe_device}")
    
    if output.shape == torch.Size([B, Len, 1000]):
        print("✅ Encoder 骨架测试通过！数据成功流过了 Embedding -> PE -> 6层 Layer。")
    else:
        print("❌ 测试失败。")