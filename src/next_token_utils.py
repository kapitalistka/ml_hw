import torch 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_text(seed_text, model, word_to_idx, idx_to_word, max_length=10):
    model.eval()
    tokens = seed_text.split()
    seq = [word_to_idx.get(token, word_to_idx['<UNK>']) for token in tokens]

    with torch.no_grad():
        for _ in range(max_length - len(tokens)):
            seq_tensor = torch.tensor([seq]).to(device)
            output = model(seq_tensor)  # [1, vocab_size]
            pred_idx = output.argmax(dim=-1).item()

            if pred_idx == 0:  # <PAD>
                break
            seq.append(pred_idx)
            if idx_to_word.get(pred_idx) in ['.', '!', '?']:
                break

    words = [idx_to_word.get(idx, '<UNK>') for idx in seq]
    return ' '.join(words)

def tensor_to_text(tensor, idx_to_word, pad_idx=0, unk_token="<UNK>", truncate_at_pad=True):

    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    if tensor.ndim == 1:
        tensor = tensor.reshape(1, -1)
    
    texts = []
    for seq in tensor:
        words = []
        for idx in seq:
            idx = int(idx)
            if idx == pad_idx and truncate_at_pad:
                break
            word = idx_to_word.get(idx, unk_token)
            words.append(word)
        texts.append(" ".join(words))
    return texts


def generate_next_word(context, model, word_to_idx, idx_to_word):
    model.eval()
    tokens = context.split()
    seq = [word_to_idx.get(token, word_to_idx['<UNK>']) for token in tokens]
    seq_tensor = torch.tensor([seq], dtype=torch.long).to(device)

    with torch.no_grad():
        output = model(seq_tensor)  # [1, vocab_size]
        pred_idx = output.argmax(dim=-1).item()

    return idx_to_word.get(pred_idx, "<UNK>")

def generate_text(context, model, word_to_idx, idx_to_word, max_length=10):
    model.eval()
    tokens = context.split()
    seq = [word_to_idx.get(token, word_to_idx['<UNK>']) for token in tokens]

    with torch.no_grad():
        for _ in range(max_length - len(tokens)):
            seq_tensor = torch.tensor([seq]).to(device)
            output = model(seq_tensor)  # [1, vocab_size]
            pred_idx = output.argmax(dim=-1).item()

            if pred_idx == 0:  # <PAD>
                break
            seq.append(pred_idx)
            if idx_to_word.get(pred_idx) in ['.', '!', '?']:
                break

    words = [idx_to_word.get(idx, '<UNK>') for idx in seq]
    return ' '.join(words)
