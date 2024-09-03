from torch.utils.data import Dataset
import os
from pathlib import Path
import torch
import pandas as pd

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

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
        tokens = []
        for i in range(len(df)):
            l = df.iloc[i]
            tokens.append(f"n_{l['note']}")
            tokens.append(f"s_{l['step']}")
            tokens.append(f"d_{l['duration']}")
        return tokens
        '''
        tensor = torch.zeros(len(df), 16,3)
        tensor.to(device)
        
        for i in range(len(df)):
            l = df.iloc[i]
            channel = int(l['channel'])
            tensor[i][channel*3] = (l['note']-44)
            tensor[i][channel*3+1] = (l['step']-512)
            tensor[i][channel*3+2] = (l['duration']-1024)
        return (tensor)
        '''
        
        '''
        for i in range(len(df)):
            l = df.iloc[i]
            tensor[i][0] = (l['note']-44)/88
            tensor[i][1] = (l['step']-512)/1024
            tensor[i][2] = (l['duration']-1024)/2048
        return (tensor)
        '''
        
    
    def __getitem__(self, index):
        piece = self.load_piece(index)
        class_name = self.paths[index].parent.name
        class_idx = self.class_to_idx[class_name]
        piece_name = os.path.basename(self.paths[index])

        return piece, class_idx, piece_name
