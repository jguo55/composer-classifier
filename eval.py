from pathlib import Path
import torch
import time
import math
from pieceDataset import pieceDataset
from torch.utils.data import DataLoader
import json
from classifier import RNN

#helper methods
def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

data_path = Path("./composer-classifier")
model_path = data_path/"model"/"model_weights_lstm_256.pth"

piece_path = data_path/"data"


train_path = piece_path/"train"
test_path = piece_path/"test"

train_data = pieceDataset(train_path)
test_data = pieceDataset(test_path)

all_categories = train_data.classes
category_lines = train_data.class_to_idx

train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)
    
model = torch.load(model_path)

vocab_path = data_path/"model"/"vocab.json"
with open(vocab_path, "r") as file:
    vocab = json.load(file)

correct = 0
total = 0
def test(dataloader, c, t):
    model.eval()
    correct = c
    total = t
    for num, (sequence, labels, name) in enumerate(dataloader):
        if name[0][:4] != 'aug_': #skip augmented data
            outputs = model(sequence)
            guess, guess_i = categoryFromOutput(outputs)
            answer = all_categories[labels]
            if labels == guess_i:
                correct+=1
            total+=1
            print(f"{correct}/{total}, Guess={guess}, Correct={answer}")
    return correct, total

correct, total = test(test_dataloader, correct, total)
correct, total = test(train_dataloader, correct, total)