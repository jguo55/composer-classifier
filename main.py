import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import os

import torch.nn as nn
import torch.nn.functional as F

def find_classes(directory: str):
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")
        
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

class pieceDataset(Dataset):
    def __init__(self, targ_dir, transform=None):
        self.paths = list(Path(targ_dir).glob("*/*.csv"))
        self.transform = transform
        self.classes, self.class_to_idx = find_classes(targ_dir)

    def __len__(self):
        return len(self.paths)
    
    def load_piece(self, index):
        piece_path = self.paths[index]
        df = pd.read_csv(piece_path)
        tensor = torch.zeros(len(df), 16, 3)
        for i in range(len(df)):
            l = df.iloc[i]
            channel = int(l['channel'])
            tensor[i][channel][0] = l['note']
            tensor[i][channel][1] = l['step']
            tensor[i][channel][2] = l['duration']
        return tensor
    
    def __getitem__(self, index):
        piece = self.load_piece(index)
        class_name = self.paths[index].parent.name
        class_idx = self.class_to_idx[class_name]
        piece_name = os.path.basename(self.paths[index])

        return piece, class_idx, piece_name

# __getitem__(index)[0] is the tensor, __getitem__(index)[1] is the label

data_path = Path("./composer-classifier")
piece_path = data_path/"data"
train_dir = piece_path/"train"
test_dir = piece_path/"test"

train_data = pieceDataset(train_dir, transform=None)
test_data = pieceDataset(test_dir, transform=None)


all_categories = train_data.classes
category_lines = train_data.class_to_idx

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

#print(train_data.__getitem__(0)[0][1])

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        hidden = F.tanh(self.i2h(input) + self.h2h(hidden))
        output = self.h2o(hidden)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

n_hidden = 128
rnn = RNN(48, n_hidden, 5)

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn

criterion = nn.NLLLoss()

def train(category_tensor, line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(torch.flatten(line_tensor[i]), hidden) #hello?

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()

import time
import math

print_every = 1
plot_every = 1000
n_iters = 250

# Keep track of losses for plotting
current_loss = 0
all_losses = []

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

start = time.time()


for iter in range(len(train_data)):
    #iter[1], iter[0]
    tup = train_data.__getitem__(iter)
    category = tup[1]
    category_tensor = torch.tensor([category])
    line_tensor = tup[0]
    line = tup[2]
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    # Print ``iter`` number, loss, name and guess
    if iter % print_every == 0:
        guess, guess_i = categoryFromOutput(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0
