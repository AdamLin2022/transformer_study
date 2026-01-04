"""
BLEU Score Calculator for the Transformer Model
使用验证集计算 BLEU 分数，客观评估翻译质量
"""
import torch
import data
import conf
from my_transformer_v8 import Transformer
from inference import greedy_decode  # 使用简单的 greedy_decode 进行调试
from collections import Counter
import math

# ==================================================================================
# 简单的 BLEU 实现（不依赖 torchtext）
# ==================================================================================
def compute_ngrams(tokens, n):
    """计算 n-gram"""
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def bleu_score(pred_sentences, ref_sentences, max_n=4):
    """
    计算 BLEU 分数
    pred_sentences: list of list of tokens (预测结果)
    ref_sentences: list of list of list of tokens (参考，每个样本可有多个参考)
    """
    precisions = []
    
    for n in range(1, max_n + 1):
        total_matches = 0
        total_count = 0
        
        for pred, refs in zip(pred_sentences, ref_sentences):
            pred_ngrams = compute_ngrams(pred, n)
            if not pred_ngrams:
                continue
                
            # 合并所有参考的 n-gram
            ref_ngram_counts = Counter()
            for ref in refs:
                ref_ngrams = compute_ngrams(ref, n)
                ref_counts = Counter(ref_ngrams)
                for ngram, count in ref_counts.items():
                    ref_ngram_counts[ngram] = max(ref_ngram_counts[ngram], count)
            
            # 计算匹配数
            pred_counts = Counter(pred_ngrams)
            matches = 0
            for ngram, count in pred_counts.items():
                matches += min(count, ref_ngram_counts.get(ngram, 0))
            
            total_matches += matches
            total_count += len(pred_ngrams)
        
        if total_count > 0:
            precisions.append(total_matches / total_count)
        else:
            precisions.append(0)
    
    # 计算 brevity penalty
    pred_len = sum(len(p) for p in pred_sentences)
    ref_len = sum(min(len(r) for r in refs) for refs in ref_sentences)
    
    if pred_len > ref_len:
        bp = 1
    elif pred_len == 0:
        bp = 0
    else:
        bp = math.exp(1 - ref_len / pred_len)
    
    # 计算几何平均
    if min(precisions) > 0:
        log_precision = sum(math.log(p) for p in precisions) / len(precisions)
        bleu = bp * math.exp(log_precision)
    else:
        bleu = 0
    
    return bleu

# ==================================================================================
# 模型加载
# ==================================================================================
print("正在加载模型...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Transformer(
    enc_voc_size=len(data.vocab_src),
    dec_voc_size=len(data.vocab_tgt),
    d_model=conf.d_model,
    n_head=conf.n_heads,
    max_len=conf.max_len,
    ffn_hidden=conf.ffn_hidden,
    n_layers=conf.n_layers,
    drop_prob=conf.drop_prob,
    device=device,
    src_pad_idx=data.PAD_IDX,
    trg_pad_idx=data.PAD_IDX
).to(device)

if torch.cuda.is_available():
    model.load_state_dict(torch.load('transformer_model.pt'))
else:
    model.load_state_dict(torch.load('transformer_model.pt', map_location=torch.device('cpu')))
model.eval()
print("模型加载完成！")

# ==================================================================================
# BLEU 计算
# ==================================================================================
print("正在计算 BLEU 分数，请稍候...")

# 使用配置文件中的 batch_size（改进点2）
valid_iter = data.make_iter(batch_size=conf.batch_size, mode='valid')

pred_sentences = []  # 模型预测的翻译
ref_sentences = []   # 真实的参考翻译

max_samples = 200  # 为了速度，只测试前 200 个样本
sample_count = 0

for batch_idx, batch in enumerate(valid_iter):
    # 兼容 torchtext BucketIterator (对象属性) 和 PyTorch DataLoader (元组)（改进点2）
    if hasattr(batch, 'src') and hasattr(batch, 'trg'):
        src_batch = batch.src
        trg_batch = batch.trg
    elif isinstance(batch, (tuple, list)) and len(batch) == 2:
        src_batch, trg_batch = batch
    else:
        raise ValueError(f"Unknown batch format: {type(batch)}")
    
    # 确保在正确的设备上
    src_batch = src_batch.to(device)
    trg_batch = trg_batch.to(device)
    
    # [调试] 打印前 2 个 Batch 的 src_tensor 信息
    if batch_idx < 2:
        print(f"\n[DEBUG Batch {batch_idx}]")
        print(f"  src_batch.shape: {src_batch.shape}")
        print(f"  src_batch[0] (first 20 tokens): {src_batch[0][:20].tolist()}")
        print(f"  src_batch[0] (last 10 tokens): {src_batch[0][-10:].tolist()}")
        print(f"  PAD_IDX = {data.PAD_IDX}, SOS_IDX = {data.SOS_IDX}, EOS_IDX = {data.EOS_IDX}")
    
    batch_size = src_batch.shape[0]
    
    for i in range(batch_size):
        if sample_count >= max_samples:
            break
        
        # [调试] 还原并打印源句子（仅在第一个样本和前几个样本打印）
        src_indexes = src_batch[i].tolist()
        src_tokens = []
        for idx in src_indexes:
            if idx == data.EOS_IDX:
                break
            if idx not in [data.PAD_IDX, data.SOS_IDX, data.EOS_IDX]:
                src_tokens.append(data.vocab_src.itos[idx])
        
        src_sentence = " ".join(src_tokens)
        
        # [调试] 打印源句子（前 5 个样本）
        if sample_count < 5:
            print(f"\n[DEBUG Sample {sample_count}]")
            print(f"  Source sentence: {src_sentence}")
            print(f"  Source indexes (first 15): {src_indexes[:15]}")
        
        # 1. 获取真实目标句子（英语参考译文）
        trg_indexes = trg_batch[i].tolist()
        ref_tokens = []
        for idx in trg_indexes:
            if idx == data.EOS_IDX:
                break
            if idx not in [data.PAD_IDX, data.SOS_IDX, data.EOS_IDX]:
                ref_tokens.append(data.vocab_tgt.itos[idx])
        
        # 2. 模型预测（使用简单的 Greedy Decode 进行调试）
        # 取出单条样本并增加 batch 维度: [seq_len] -> [1, seq_len]
        src_tensor = src_batch[i].unsqueeze(0)
        
        # [调试] 打印 src_tensor 信息（前 5 个样本）
        if sample_count < 5:
            print(f"  src_tensor.shape: {src_tensor.shape}")
            print(f"  src_tensor[0] (first 15): {src_tensor[0][:15].tolist()}")
        
        # [调试] 强制禁用 src_mask（传入 None）
        src_mask = None  # 暂时禁用 mask 进行调试
        # src_mask = model.make_src_mask(src_tensor)  # 原始代码（已注释）
        
        if sample_count < 5:
            print(f"  Using src_mask = None (masking disabled)")
        
        # 使用简单的 greedy_decode，确保 Encoder-Decoder 连接正确
        best_indexes = greedy_decode(
            model, src_tensor, src_mask, 
            max_len=50, start_symbol=data.SOS_IDX
        )
        
        # 将索引转回单词（跳过 SOS_IDX, 在 EOS_IDX 处截断）
        pred_tokens = []
        for idx in best_indexes:
            if idx == data.SOS_IDX:
                continue
            if idx == data.EOS_IDX:
                break
            pred_tokens.append(data.vocab_tgt.itos[idx])
        
        # [调试] 打印预测结果（前 5 个样本）
        if sample_count < 5:
            print(f"  Predicted: {' '.join(pred_tokens)}")
            print(f"  Reference: {' '.join(ref_tokens)}")
        
        # 3. 存储结果
        pred_sentences.append(pred_tokens)
        ref_sentences.append([ref_tokens])  # BLEU 要求 ref 是 list of lists
        
        sample_count += 1
        if sample_count % 50 == 0:
            print(f"  已处理 {sample_count}/{max_samples} 个样本...")
    
    if sample_count >= max_samples:
        break

# 计算 BLEU
bleu = bleu_score(pred_sentences, ref_sentences)

print("\n" + "=" * 50)
print(f"BLEU Score: {bleu * 100:.2f}")
print("=" * 50)

# 判断标准
if bleu > 0.20:
    print("✅ 模型表现良好！BLEU > 20 说明翻译质量不错。")
elif bleu > 0.10:
    print("⚠️ 模型表现一般，还有提升空间。")
elif bleu > 0.05:
    print("⚠️ 模型在学习中，但翻译质量较低。")
else:
    print("❌ 模型可能有问题，需要检查训练过程。")

# 展示几个翻译示例
print("\n" + "=" * 50)
print("翻译示例:")
print("=" * 50)
for i in range(min(5, len(pred_sentences))):
    print(f"\n[样本 {i+1}]")
    print(f"  参考: {' '.join(ref_sentences[i][0])}")
    print(f"  预测: {' '.join(pred_sentences[i])}")
