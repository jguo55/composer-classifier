import json
import pathlib as Path
from pieceDataset import pieceDataset
from torch.utils.data import DataLoader


data_path = Path("./composer-classifier")
piece_path = data_path/"data"


train_path = piece_path/"train"
test_path = piece_path/"test"

train_data = pieceDataset(train_path)
test_data = pieceDataset(test_path)

train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)

vocab = {"UNK": 0}
total = len(train_dataloader)
for num, (sequence, labels, name) in enumerate(train_dataloader):
    for token in sequence:
        token = token[0] #WHY IS IT A TUPLE
        if token not in vocab:
            vocab[token] = len(vocab)
    print(f"building vocab {num+1}/{total})")

#save vocab to json
vocab_path = data_path/"model"/"vocab.json"
with open(vocab_path, 'w') as outfile:
    json.dump(vocab, outfile)

