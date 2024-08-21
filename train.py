import torch
from torch.utils.data import DataLoader
from pathlib import Path
from pieceDataset import pieceDataset

import torch.nn as nn
import torch.nn.functional as F

import time
import math


#1638 files total, use 1310 for training and 328 for  testing
#train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [0.8, 0.2])

data_path = Path("./composer-classifier")
piece_path = data_path/"data"

full_data = pieceDataset(piece_path)

train_data, test_data = torch.utils.data.random_split(full_data, [0.8, 0.2])

train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)

all_categories = full_data.classes
category_lines = full_data.class_to_idx

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

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

    for i in range(line_tensor[0].size()[0]):
        output, hidden = rnn(torch.flatten(line_tensor[0][i]), hidden) 

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()

n_iters = len(train_data)

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

for iter in enumerate(train_dataloader):
    category = iter[1][1]
    category_tensor = torch.tensor([category])
    line_tensor = iter[1][0]
    line = iter[1][2]
    output, loss = train(category_tensor, line_tensor)
    current_loss += loss

    guess, guess_i = categoryFromOutput(output)
    correct = '✓' if guess == all_categories[category] else '✗ (%s)' % all_categories[category]
    print('%d %d%% (%s) %.4f %s / %s %s' % (iter[0]+1, iter[0] / n_iters * 100, timeSince(start), loss, line, guess, correct))

model_path = data_path/"model"/"model_weights.pth"
torch.save(rnn, model_path)

def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor[0].size()[0]):
        output, hidden = rnn(torch.flatten(line_tensor[0][i]), hidden) 
    return output

rnn = torch.load(model_path)

acc = 0
total = 0
for iter in enumerate(test_dataloader):
    total+=1
    category = iter[1][1]
    category_tensor = torch.tensor([category])
    line_tensor = iter[1][0]
    line = iter[1][2]
    output = evaluate(line_tensor)

    guess, guess_i = categoryFromOutput(output)
        # Print ``iter`` number, loss, name and guess
    if guess == all_categories[category]:
        acc+=1

    print(f"{acc}/{total} correct, guess: {guess}, answer: {all_categories[category]}")
