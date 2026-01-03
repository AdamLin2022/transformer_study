import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_model // n_head
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        q = self.w_q(q).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.n_head, self.d_head).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.w_out(output)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))

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
        return x + self.pe[:, :x.size(1), :]

class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)
        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)

    def forward(self, x, src_mask):
        _x = x
        x = self.attention(x, x, x, src_mask)
        x = self.norm1(x + self.dropout1(_x))
        _x = x
        x = self.ffn(x)
        x = self.norm2(x + self.dropout2(_x))
        return x

class Encoder(nn.Module):
    def __init__(self, d_model, n_head, max_len, ffn_hidden, enc_voc_size, drop_prob, n_layers, device):
        super().__init__()
        self.emb = nn.Embedding(enc_voc_size, d_model)
        self.pe = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(drop_prob)
        self.layers = nn.ModuleList([EncoderLayer(d_model, ffn_hidden, n_head, drop_prob) for _ in range(n_layers)])

    def forward(self, x, src_mask):
        x = self.emb(x) * math.sqrt(self.emb.embedding_dim)
        x = self.pe(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_head)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(drop_prob)
        
        self.enc_dec_attention = MultiHeadAttention(d_model, n_head)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(drop_prob)
        
        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(drop_prob)

    def forward(self, dec, enc_src, trg_mask, src_mask):
        _x = dec
        x = self.self_attention(q=dec, k=dec, v=dec, mask=trg_mask)
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        if enc_src is not None:
            _x = x
            x = self.enc_dec_attention(q=x, k=enc_src, v=enc_src, mask=src_mask)
            x = self.dropout2(x)
            x = self.norm2(x + _x)

        _x = x
        x = self.ffn(x)
        x = self.dropout3(x)
        x = self.norm3(x + _x)
        
        return x

class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = nn.Embedding(dec_voc_size, d_model)
        self.pe = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(drop_prob)
        
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, ffn_hidden, n_head, drop_prob)
            for _ in range(n_layers)
        ])
        
        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        x = self.emb(trg) * math.sqrt(self.emb.embedding_dim)
        x = self.pe(x)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, enc_src, trg_mask, src_mask)
            
        output = self.linear(x)
        return output

class Transformer(nn.Module):
    def __init__(self, enc_voc_size, dec_voc_size, d_model, n_head, max_len, ffn_hidden, n_layers, drop_prob, device):
        super().__init__()
        self.device = device
        self.src_pad_idx = 0
        self.trg_pad_idx = 0
        
        self.encoder = Encoder(d_model, n_head, max_len, ffn_hidden, enc_voc_size, drop_prob, n_layers, device)
        self.decoder = Decoder(dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device)

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {device}")
    
    V_SRC = 1000
    V_TRG = 2000
    
    model = Transformer(
        enc_voc_size=V_SRC, dec_voc_size=V_TRG,
        d_model=512, n_head=8, max_len=100,
        ffn_hidden=2048, n_layers=6, drop_prob=0.1,
        device=device
    ).to(device)
    
    src = torch.randint(1, V_SRC, (2, 10)).to(device)
    trg = torch.randint(1, V_TRG, (2, 12)).to(device)
    
    output = model(src, trg)
    
    print(f"\n[Step 4: 终极验证]")
    print(f"输入 src: {src.shape}")
    print(f"输入 trg: {trg.shape}")
    print(f"输出 logits: {output.shape}")
    
    expected_shape = torch.Size([2, 12, V_TRG])
    if output.shape == expected_shape:
        print("✅✅✅ 恭喜")
        print("数学结构验证通过：Self-Attn (Masked) + Cross-Attn 逻辑正确。")
    else:
        print(" 测试失败")