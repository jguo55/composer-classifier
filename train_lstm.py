import torch
from torch.utils.data import DataLoader
from pathlib import Path
from pieceDataset import pieceDataset
import torch.nn as nn
import time
import math
import json
from matplotlib import pyplot as plt
from classifier import RNN
import copy

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

#data 1310 train, 328 test
data_path = Path("./composer-classifier")
piece_path = data_path/"data"


train_path = piece_path/"train"
test_path = piece_path/"test"

full_data = pieceDataset(train_path)
test_data = pieceDataset(test_path)

train_data, val_data = torch.utils.data.random_split(full_data, [0.8, 0.2])

train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=1, shuffle=True)

all_categories = full_data.classes

start = time.time()

#MAKE SURE TO REBUILD THE VOCAB IF YOU CHANGE THE TRAINING DATA
vocab_path = data_path/"model"/"vocab.json"
with open(vocab_path, "r") as file:
    vocab = json.load(file)

#model parameters
hidden_size = 64
output_size = 5
n_layers = 1
embed_size = 64
vocab_size = len(vocab)
patience = 2

model = RNN(vocab_size, embed_size, hidden_size, output_size, n_layers)

epochs = 100
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

trainlen = len(train_dataloader)
testlen = len(test_dataloader)
vallen = len(val_dataloader)

losses = []
val_losses = []
current_loss = 0

best_weights = None

for epoch in range(epochs):
    model.train()
    for num, (sequence, labels, name) in enumerate(train_dataloader):
        outputs = model(sequence)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        guess, guess_i = categoryFromOutput(outputs)
        answer = all_categories[labels]
        current_loss+=loss.item()

        print(f"{epoch+1} {num+1}/{trainlen} ({timeSince(start)}) {loss:.4f} {name} guess: {guess}, ans: {answer}")
    losses.append(current_loss/trainlen)
    current_loss = 0

    model.eval()
    for num, (sequence, labels, name) in enumerate(val_dataloader):
        outputs = model(sequence)
        loss = criterion(outputs, labels) #compute loss but don't use it to train
        guess, guess_i = categoryFromOutput(outputs)
        answer = all_categories[labels]
        current_loss+=loss.item()

        print(f"{epoch+1} {num+1}/{vallen} ({timeSince(start)}) {loss:.4f} {name} guess: {guess}, ans: {answer}")
    val_losses.append(current_loss / vallen)
    print(losses)
    print(val_losses)
    current_loss = 0

    if len(val_losses) > patience:
        if val_losses[-1]  > val_losses[-patience-1]:
            break #train until validation loss starts to increase
    
    best_weights = copy.deepcopy(model.state_dict())

plt.plot(val_losses)
plt.plot(losses)
plt.show()

model.load_state_dict(best_weights) #load from the best instance
model_path = data_path/"model"/"model_weights_lstm_val.pth"
torch.save(model, model_path)


#model = torch.load(model_path)
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
    print(f"{correct}/{total}, guess: {guess}, ans: {answer}")