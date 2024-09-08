import torch.nn as nn
import torch
from pathlib import Path
import json

data_path = Path("./composer-classifier")

piece_path = data_path/"data"

vocab_path = data_path/"model"/"vocab.json"
with open(vocab_path, "r") as file:
    vocab = json.load(file)

def tokentoidx(sequence):
    for i in range(len(sequence)):
        if sequence[i][0] in vocab:
            sequence[i] = vocab[sequence[i][0]]
        else:
            sequence[i] = 0 #unk
    return sequence

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, output_size, n_layers):
        super(RNN, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.embedding_size = embedding_size

        self.embedding = nn.Embedding(vocab_size,embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, n_layers, batch_first=True)

        self.linear = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.tensor([tokentoidx(x)])
        x = self.embedding(x)
        out, hn = self.lstm(x)
        out = self.linear(out[:,-1,:])
        return out