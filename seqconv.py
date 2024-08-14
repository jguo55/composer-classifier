#converts csv midi files to a note sequence csv
#makes each note has a duration, rather than a start/end time
from pathlib import Path
import os
import pandas as pd

data_path = Path("./composer-classifier")
piece_path = data_path/"data"
train_dir = piece_path/"train"
test_dir = piece_path/"test"

paths = list(Path(train_dir).glob("*/*.csv"))

df = pd.read_csv(paths[0])
notes = []
steps = []
durations = []
channels = []
velocities = []

#find first note for accurate step
for i in range(len(df)):
    l = df.iloc[i]
    if l['type'] == 'note_on':
        prev_start = l['tick']
        break

for i in range(len(df)):
    l = df.iloc[i]
    if l['velocity'] != 0:
        j = i+1
        while j < len(df):
            l2 = df.iloc[j]
            if l['note'] == l2['note'] and l['channel'] == l2['channel'] and l2['velocity'] == 0:
                notes.append(l['note'])
                steps.append(l['tick']-prev_start)
                durations.append(l2['tick']-l['tick'])
                channels.append(l['channel'])
                velocities.append(l['velocity'])
                prev_start = l['tick']
                break
            j+=1
dict = {'note': notes, 'step': steps, 'duration': durations, 'channel': channels, 'velocity': velocities}
output = pd.DataFrame(dict)

print(paths[0])
print(output)

filename = "test.csv"
outpath = data_path/"sequences"/filename
output.to_csv(outpath)