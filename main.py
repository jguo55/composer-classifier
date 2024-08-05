import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import os

import torch.nn as nn
import torch.nn.functional as F

def note():
    def __init__(self, note, velocity, begin, channel):
        self.note = note
        self.velocity = velocity
        self.begin = begin
        self.channel = channel

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
        #tokendict = {'note_on': 'n', 'note_off': 'n', 'set_tempo': 't', 'time_signature': 's'}

    def __len__(self):
        return len(self.paths)
    
    def load_piece(self, index):
        piece_path = self.paths[index]
        df = pd.read_csv(piece_path)
        arr = []
        for i in range(len(df)):
            l = df.iloc[i]
            if l['type'] == 'note_off' or l['type'] == 'note_on':
                j = i+1
                while j < len(df):
                    l2 = df.iloc[j]
                    if l['note'] == l2['note'] and l['channel'] == l2['channel'] and l2['velocity'] == 0:
                        arr.append([l['note'],l['velocity'],l2['tick']-l['tick'], l['channel']])
                        break
                    j+=1
        print(arr)
        tens = torch.tensor(arr)
        return F.one_hot(tens.argmax(dim=1), num_classes=16)
        return tens.index_select(dim=1, index=3)

    
    def __getitem__(self, index):
        piece = self.load_piece(index)
        class_name = self.paths[index].parent.name
        class_idx = self.class_to_idx[class_name]

        return piece, class_idx


data_path = Path("./composer-classifier")
piece_path = data_path/"data"
train_dir = piece_path/"train"
test_dir = piece_path/"test"

train_data = pieceDataset(train_dir, transform=None)
test_data = pieceDataset(test_dir, transform=None)

trainloader = DataLoader(dataset="train_data", batch_size=32, num_workers=0,shuffle=True)
testloader = DataLoader(dataset="train_data", batch_size=32, num_workers=0,shuffle=True)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

print(train_data.__getitem__(700)[0])
