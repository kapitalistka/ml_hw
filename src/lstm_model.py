import torch.nn as nn
import torch.nn.functional as F


class LSTMNextToken(nn.Module):
    def __init__(self, vocab_size, embed_dim=50, hidden_dim=100, num_layers=1, dropout=0.1):
        super(LSTMNextToken, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x) 
        lstm_out, _ = self.lstm(embedded)
        last_hidden = self.dropout(lstm_out[:, -1, :])  #??
        logits = self.fc(last_hidden)  
        return logits