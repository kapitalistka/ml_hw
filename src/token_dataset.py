from torch.utils.data import Dataset
import torch

class TokenPredictionDataset(Dataset):
    def __init__(self, texts, word_to_idx):
        self.input_sequences = []
        self.target_tokens = []

        for text in texts:
            tokens = text.split()
            seq = [word_to_idx.get(token, word_to_idx['<UNK>']) for token in tokens]
            for i in range(1, len(seq)):
                self.input_sequences.append(seq[:i])
                self.target_tokens.append(seq[i]) 

    def __len__(self):
        return len(self.input_sequences)

    def __getitem__(self, idx):
        input_seq = self.input_sequences[idx]
        target_token = self.target_tokens[idx]
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_token, dtype=torch.long)

class ValDataset(Dataset):
    
    def __init__(self, texts, word_to_idx, max_len=None):
        self.sequences = []
        self.targets = []
        self.max_len = max_len
        unk_idx = word_to_idx.get('<UNK>', 1)
        #pad_idx = word_to_idx.get('<PAD>', 0)

        for text in texts:
            tokens = text.strip().split()
            if len(tokens) < 2:
                continue
            indices = [word_to_idx.get(token, unk_idx) for token in tokens]
            # пары: (контекст, следующее слово)
            for i in range(1, len(indices)):
                context = indices[:i]
                target = indices[i]
                if max_len:
                    context = context[-max_len:]  # Ограничиваем длину???
                self.sequences.append(context)
                self.targets.append(target)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.sequences[idx], dtype=torch.long),
            torch.tensor(self.targets[idx], dtype=torch.long)
        )


def collate_fn(batch):
    input_seqs, targets = zip(*batch)
    padded_inputs = torch.nn.utils.rnn.pad_sequence(input_seqs, batch_first=True, padding_value=0)
    targets = torch.stack(targets)
    return padded_inputs, targets