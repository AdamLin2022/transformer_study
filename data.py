import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from datasets import load_dataset
import spacy
from collections import Counter
import sys

try:
    de_tokenizer = spacy.load('de_core_news_sm')
    en_tokenizer = spacy.load('en_core_web_sm')
except OSError:
    print("Please run: python3 -m spacy download de_core_news_sm")
    print("Please run: python3 -m spacy download en_core_web_sm")
    sys.exit()

def tokenize_de(text):
    return [token.text for token in de_tokenizer.tokenizer(text)]

def tokenize_en(text):
    return [token.text for token in en_tokenizer.tokenizer(text)]

class Vocab:
    def __init__(self, tokens, min_freq=2):
        self.stoi = {'<unk>': 0, '<pad>': 1, '<sos>': 2, '<eos>': 3}
        self.itos = {0: '<unk>', 1: '<pad>', 2: '<sos>', 3: '<eos>'}
        
        counter = Counter(tokens)
        
        idx = 4
        for token, freq in counter.items():
            if freq >= min_freq:
                self.stoi[token] = idx
                self.itos[idx] = token
                idx += 1
                
    def __len__(self):
        return len(self.stoi)
        
    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi['<unk>'])
    
    def lookup_indices(self, tokens):
        return [self[token] for token in tokens]

dataset = load_dataset("bentrevett/multi30k")

def build_vocab_from_dataset(dataset, tokenizer, lang_key):
    all_tokens = []
    for item in dataset:
        all_tokens.extend(tokenizer(item[lang_key]))
    return Vocab(all_tokens, min_freq=2)

vocab_src = build_vocab_from_dataset(dataset['train'], tokenize_de, 'de')
vocab_tgt = build_vocab_from_dataset(dataset['train'], tokenize_en, 'en')

UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3

def text_transform(tokenizer, vocab, text):
    tokens = tokenizer(text)
    return [SOS_IDX] + vocab.lookup_indices(tokens) + [EOS_IDX]

def collate_fn(batch):
    src_batch, trg_batch = [], []
    
    for item in batch:
        src_text = item['de'].rstrip("\n")
        trg_text = item['en'].rstrip("\n")
        
        src_tokens = text_transform(tokenize_de, vocab_src, src_text)
        trg_tokens = text_transform(tokenize_en, vocab_tgt, trg_text)
        
        src_batch.append(torch.tensor(src_tokens, dtype=torch.long))
        trg_batch.append(torch.tensor(trg_tokens, dtype=torch.long))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
    trg_batch = pad_sequence(trg_batch, padding_value=PAD_IDX, batch_first=True)
    
    return src_batch, trg_batch

def make_iter(batch_size, mode='train'):
    hf_mode = 'validation' if mode == 'valid' else mode
    data_iter = dataset[hf_mode]
    dataloader = DataLoader(data_iter, batch_size=batch_size, collate_fn=collate_fn, shuffle=(mode=='train'))
    return dataloader

if __name__ == '__main__':
    loader = make_iter(batch_size=2, mode='train')
    src, trg = next(iter(loader))
    print(f"Src Shape: {src.shape}")
    print(f"Trg Shape: {trg.shape}")
    print(f"Src Example: {src[0]}")
    print("Data Pipeline Ready")