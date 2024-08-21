#THIS IS INCOMPLETE. do not use
from pathlib import Path
import torch


all_categories = train_data.classes
category_lines = train_data.class_to_idx

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

def evaluate(line_tensor):
    hidden = rnn.initHidden()

    for i in range(line_tensor[0].size()[0]):
        output, hidden = rnn(torch.flatten(line_tensor[0][i]), hidden) 
    return output

data_path = Path("./composer-classifier")
model_path = data_path/"model"/"model_weights.pth"
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