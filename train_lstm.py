import torch
from torch.utils.data import DataLoader
from pathlib import Path
from pieceDataset import pieceDataset

import torch.nn as nn
import torch.nn.functional as F

import time
import math

#helper methods
def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

def tokentoidx(sequence):
    for i in range(len(sequence)):
        if sequence[i] in vocab:
            sequence[i] = vocab[sequence[i]]
        else:
            sequence[i] = 0 #unk
    return sequence

#data 1286 train, 352 test
data_path = Path("./composer-classifier")
piece_path = data_path/"data"


train_path = piece_path/"train"
test_path = piece_path/"test"

train_data = pieceDataset(train_path)
test_data = pieceDataset(test_path)

train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)

all_categories = train_data.classes
category_lines = train_data.class_to_idx

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

start = time.time()

vocab = {"UNK": 0}
total = len(train_dataloader)
for num, (sequence, labels, name) in enumerate(train_dataloader):
    for token in sequence:
        if token not in vocab:
            vocab[token] = len(vocab)
    print(f"building vocab {num+1}/{total} ({timeSince(start)})")


#model & training
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
    
    #forward is where we connect all the layers basically
    def forward(self, x):
        x = torch.tensor([tokentoidx(x)])
        x = self.embedding(x)
        out, hn = self.lstm(x)
        out = self.linear(out[:,-1,:])
        return out
    

hidden_size = 128
output_size = 5
n_layers = 1
embed_size = 64
vocab_size = len(vocab)

print(vocab_size)

model = RNN(vocab_size, embed_size, hidden_size, output_size, n_layers)

epochs = 1
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#weights = torch.tensor([1638/1025, 1638/181, 1638/60, 1638/136, 1638/236])  
criterion = nn.CrossEntropyLoss()

trainlen = len(train_dataloader)
testlen = len(test_dataloader)

for epoch in range(epochs):
    model.train()
    for num, (sequence, labels, name) in enumerate(train_dataloader):
        #forward prop
        outputs = model(sequence)
        #calculate loss
        loss = criterion(outputs, labels)

        #clear gradients
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        guess, guess_i = categoryFromOutput(outputs)
        answer = all_categories[labels]

        print(f"{epoch+1} {num+1}/{trainlen} {num/trainlen*100:.1f}% ({timeSince(start)}) {loss:.4f} {name} Guess={guess} Correct={answer}")

model_path = data_path/"model"/"model_weights_lstm.pth"
torch.save(model, model_path)

#testing
model.eval()
correct = 0
total = 0
for num, (sequence, labels, name) in enumerate(test_dataloader):
    outputs = model(sequence)
    guess, guess_i = categoryFromOutput(outputs)
    answer = all_categories[labels]
    if labels == guess_i:
        correct+=1
    total+=1
    print(f"{correct}/{total}, Guess={guess}, Correct={answer}")