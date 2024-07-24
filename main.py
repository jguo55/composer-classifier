import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from pathlib import Path
import os

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
        return pd.read_csv("piece_path")
    
    def __getitem__(self, index):
        piece = self.load_piece(index)
        class_name = self.paths[index].parent.name
        class_idx = self.class_to_idx[class_name]


data_path = Path("./composer-influence-ai")
piece_path = data_path/"data"

train_dir = piece_path/"train"

train_data = pieceDataset(train_dir, transform=None)
print(len(train_data))