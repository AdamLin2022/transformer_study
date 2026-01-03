import torch
import data
import conf
from my_transformer_v8 import Transformer

model = Transformer(
    enc_voc_size=len(data.vocab_src),
    dec_voc_size=len(data.vocab_tgt),
    d_model=conf.d_model,
    n_head=conf.n_heads,
    max_len=conf.max_len,
    ffn_hidden=conf.ffn_hidden,
    n_layers=conf.n_layers,
    drop_prob=conf.drop_prob,
    device=conf.device
).to(conf.device)

print("Loading model weights...")
model.load_state_dict(torch.load('transformer_model.pt', map_location=conf.device))
model.eval()
print("Model loaded!")

def translate_sentence(sentence, model, max_len=50):
    tokens = data.tokenize_de(sentence)
    src_indexes = [data.SOS_IDX] + [data.vocab_src[token] for token in tokens] + [data.EOS_IDX]
    
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(conf.device)
    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [data.SOS_IDX]

    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(conf.device)
        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():
            output = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        
        pred_token = output.argmax(2)[:,-1].item()
        trg_indexes.append(pred_token)

        if pred_token == data.EOS_IDX:
            break

    trg_tokens = [data.vocab_tgt.itos[i] for i in trg_indexes]
    
    return trg_tokens[1:-1]

if __name__ == '__main__':
    sentences = [
        "Ein Mann geht auf der Straße.",
        "Eine Frau sitzt auf der Bank.",
        "Zwei Katzen spielen im Garten.",
        "Guten Morgen!"
    ]

    print("\n====== Translation Test ======")
    for sent in sentences:
        trans = translate_sentence(sent, model)
        print(f"German:  {sent}")
        print(f"English: {' '.join(trans)}")
        print("-" * 30)