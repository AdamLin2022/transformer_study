
import torch.nn as nn

class TokenEmbedding(nn.Embedding):
    """
    对应 GitHub: models/embedding/token_embedding.py
    """
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)