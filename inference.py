import torch
import torch.nn.functional as F
import math
import data
import conf
from my_transformer_v8 import Transformer

# ==================================================================================
# 模型加载
# ==================================================================================

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

print("Loading model weights...")
if torch.cuda.is_available():
    model.load_state_dict(torch.load('transformer_model.pt'))
else:
    model.load_state_dict(torch.load('transformer_model.pt', map_location=torch.device('cpu')))
model.eval()
print("Model loaded!")

# ==================================================================================
# 简单的 Greedy Decode 实现（用于调试）
# ==================================================================================

def greedy_decode(model, src, src_mask, max_len=50, start_symbol=2):
    """
    极其简单的 Greedy Decode 函数，用于调试
    确保 Decoder 正确使用 Encoder 的输出
    :param model: Transformer 模型
    :param src: 源语言张量 [1, src_len]
    :param src_mask: 源语言掩码 [1, 1, 1, src_len] 或 None（用于调试）
    :param max_len: 最大生成长度
    :param start_symbol: 起始符号索引（默认 2 = SOS_IDX）
    :return: 生成的索引列表（包含 SOS 和 EOS）
    """
    model.eval()
    
    with torch.no_grad():
        # [调试] 如果 src_mask 为 None，创建一个全 True 的 mask（禁用 masking）
        if src_mask is None:
            src_len = src.shape[1]
            src_mask = torch.ones(1, 1, 1, src_len, dtype=torch.bool, device=src.device)
            print(f"[DEBUG] src_mask is None, created all-True mask with shape {src_mask.shape}")
        
        # 1. Encoder 前向传播（只跑一次）
        memory = model.encoder(src, src_mask)  # [1, src_len, d_model]
        
        # 2. 初始化 Decoder 输入（只有 SOS）
        ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)  # [1, 1]
        
        # 3. 逐步生成
        for i in range(max_len - 1):
            # 构建 trg_mask（每次都要重新构建，因为序列长度在增长）
            trg_mask = model.make_trg_mask(ys)  # [1, seq_len, seq_len]
            
            # Decoder 前向传播
            # 关键：确保 memory (enc_src) 和 src_mask 正确传入
            out = model.decoder(ys, memory, trg_mask, src_mask)  # [1, seq_len, vocab_size]
            
            # 取最后一个时间步的预测
            prob = out[:, -1, :]  # [1, vocab_size]
            
            # Greedy：选择概率最高的词
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()
            
            # 将新词添加到序列中
            ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
            
            # 如果生成 EOS，提前终止
            if next_word == data.EOS_IDX:
                break
        
        return ys[0].tolist()  # 返回列表 [SOS, word1, word2, ..., EOS]

# ==================================================================================
# Beam Search 实现
# ==================================================================================

def beam_search_decode(model, src_tensor, src_mask, max_len=50, beam_size=5, alpha=0.7):
    """
    执行 Beam Search 解码
    :param model: Transformer 模型
    :param src_tensor: 源语言张量 [1, src_len]
    :param src_mask: 源语言掩码
    :param max_len: 最大生成长度
    :param beam_size: 集束大小 (Beam Width)
    :param alpha: 长度惩罚因子 (Length Penalty Factor), 通常 0.6 ~ 0.8
    :return: 最佳序列的索引列表
    """
    
    # 1. 编码器前向传播 (Encoder Forward)
    # 只需要计算一次 Encoder，因为源句子是不变的
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    # 2. 初始化 Beam
    # 每个候选路径格式: (indices_list, log_prob_score)
    # 初始状态只有 [SOS]，分数为 0
    beams = [([data.SOS_IDX], 0.0)]
    
    # 存储所有已经遇到 <eos> 的完成序列
    completed_sequences = []

    # 3. 逐步生成
    for _ in range(max_len):
        new_beams = []
        
        # 遍历当前 Beam 中的所有候选路径
        for trg_indexes, score in beams:
            # 如果当前路径已经结束（最后一个词是 EOS），则不再扩展
            # (理论上已结束的序列在上一轮会被移入 completed_sequences，这里是双重保险)
            if trg_indexes[-1] == data.EOS_IDX:
                completed_sequences.append((trg_indexes, score))
                continue

            # 准备 Decoder 输入
            trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(conf.device)
            trg_mask = model.make_trg_mask(trg_tensor)

            with torch.no_grad():
                # Decoder 输出 shape: [1, curr_seq_len, vocab_size]
                output = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
                
                # 我们只关心当前这一步（最后一个时间步）的预测
                # prediction shape: [1, vocab_size]
                prediction = output[:, -1, :]
                
                # 计算 Log Softmax 得到对数概率
                log_probs = F.log_softmax(prediction, dim=-1)

            # 选出概率最高的 top-k 个词 (top-k beam_size)
            # topk_log_probs: [1, beam_size]
            # topk_indices:   [1, beam_size]
            topk_log_probs, topk_indices = log_probs.topk(beam_size, dim=-1)

            # 扩展当前路径
            for i in range(beam_size):
                token_idx = topk_indices[0][i].item()
                log_prob = topk_log_probs[0][i].item()
                
                new_score = score + log_prob
                new_indexes = trg_indexes + [token_idx]
                
                # 如果生成了 EOS，视为完成序列，放入 completed 列表
                if token_idx == data.EOS_IDX:
                    completed_sequences.append((new_indexes, new_score))
                else:
                    new_beams.append((new_indexes, new_score))
        
        # 4. 剪枝 (Pruning)
        # 如果没有新的未完成序列生成（说明所有路径都结束了），则退出循环
        if not new_beams:
            break
            
        # 对新生成的候选路径按分数从高到低排序
        new_beams.sort(key=lambda x: x[1], reverse=True)
        
        # 仅保留前 beam_size 个最好的未完成路径
        beams = new_beams[:beam_size]
        
        # 优化：如果当前最好的完成序列的分数，已经比最好的未完成序列还要高很多（考虑到长度惩罚），
        # 其实可以提前退出。但在简单实现中，我们跑完 max_len 或者直到 beams 为空。

    # 5. 选择最终最佳序列 (Length Penalty)
    # 如果没有生成任何完成的序列（极罕见），就从当前的 beams 里选最好的
    if not completed_sequences:
        completed_sequences = beams

    best_seq = None
    best_score = -float('inf')

    for seq, score in completed_sequences:
        # 长度惩罚公式: Score = log_prob / length^alpha
        # 长度计算：通常不包含 SOS，但包含 EOS。这里 seq 包含 SOS 和 EOS。
        # 有效长度 = len(seq) - 1
        valid_length = len(seq) - 1
        if valid_length < 1: valid_length = 1
            
        lp = math.pow(valid_length, alpha)
        final_score = score / lp
        
        if final_score > best_score:
            best_score = final_score
            best_seq = seq
            
    return best_seq

# ==================================================================================
# 主翻译函数
# ==================================================================================

def translate_sentence(sentence, model, max_len=50, beam_size=5):
    """
    翻译单个句子，默认使用 Beam Search
    """
    # 1. 分词与数值化
    tokens = data.tokenize_de(sentence)
    src_indexes = [data.SOS_IDX] + [data.vocab_src[token] for token in tokens] + [data.EOS_IDX]
    
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(conf.device)
    src_mask = model.make_src_mask(src_tensor)
    
    # 2. 执行 Beam Search
    # 如果想用回 Greedy Search，可以把 beam_size 设为 1
    best_indexes = beam_search_decode(model, src_tensor, src_mask, max_len, beam_size=beam_size)
    
    # 3. 将索引转回单词
    trg_tokens = [data.vocab_tgt.itos[i] for i in best_indexes]
    
    # 4. 去除 <sos> 和 <eos>
    result = []
    for token in trg_tokens:
        if token == '<sos>': continue
        if token == '<eos>': break
        result.append(token)
        
    return result

if __name__ == '__main__':
    sentences = [
        "Ein Mann geht auf der Straße.",
        "Eine Frau sitzt auf der Bank.",
        "Zwei Katzen spielen im Garten.",  # 重点测试这一句
        "Guten Morgen!"
    ]

    print(f"\n====== Translation Test (Beam Size = 5) ======")
    for sent in sentences:
        trans = translate_sentence(sent, model, beam_size=5)
        print(f"German:  {sent}")
        print(f"English: {' '.join(trans)}")
        print("-" * 30)
