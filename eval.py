from pathlib import Path
import torch
import time
import math
from pieceDataset import pieceDataset

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

data_path = Path("./composer-classifier")
model_path = data_path/"model"/"model_weights_lstm.pth"
model = torch.load(model_path)

correct = 0
total = 0
def test(dataloader, c, t):
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