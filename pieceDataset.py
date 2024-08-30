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
    
    #needs to be normalized, the whole dataset, idk how i would do that
    def load_piece(self, index):
        piece_path = self.paths[index]
        df = pd.read_csv(piece_path)
        tensor = torch.zeros(len(df), 3)
        tensor.to(device)
        
        '''
        for i in range(len(df)):
            l = df.iloc[i]
            channel = int(l['channel'])
            tensor[i][channel][0] = l['note']
            tensor[i][channel][1] = l['step']
            tensor[i][channel][2] = l['duration']
        return torch.nn.functional.normalize(tensor)
        '''
        
        for i in range(len(df)):
            l = df.iloc[i]
            tensor[i][0] = l['note']
            tensor[i][1] = l['step']
            tensor[i][2] = l['duration']
        return (tensor)
        
    
    def __getitem__(self, index):
        piece = self.load_piece(index)
        class_name = self.paths[index].parent.name
        class_idx = self.class_to_idx[class_name]
        piece_name = os.path.basename(self.paths[index])

        return piece, class_idx, piece_name
