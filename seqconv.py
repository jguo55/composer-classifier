#converts csv midi files to a note sequence csv
#makes each note has a duration, rather than a start/end time
#start from the back and append, then reverse all the lists
#also split the data into train/test sets 80-20
from pathlib import Path
import os
import pandas as pd
import random

data_path = Path("./composer-classifier")

raw_path = data_path/"raws"
classes = sorted(entry.name for entry in os.scandir(raw_path) if entry.is_dir())

for i in classes:
    testpath = data_path/"data"/"test"/i
    trainpath = data_path/"data"/"train"/i
    os.makedirs(testpath)
    os.makedirs(trainpath)

piece_path = data_path/"raws"

paths = list(Path(piece_path).glob("*/*.csv"))

#rounds each note value to the nearest 15,20,30,40, multiple of 30 to decrease vocab size
#most files use these values to represent sixteenth, triplets, etc. not perfect but should work
def roundnote(n):
    if n < 18:
        return 15
    elif n < 25:
        return 20
    elif n < 35:
        return 30
    elif n < 50:
        return 40
    else:
        return 30*round(n/30)

for k in range(len(paths)):
    filename = os.path.basename(paths[k])
    dirname = os.path.basename(os.path.dirname(paths[k]))
    path = str(paths[k])
    df = pd.read_csv(paths[k])
    notes = []
    steps = []
    durations = []
    channels = []
    velocities = []
    notepair = {}
    nopairlines = []

    prev_end = df.iloc[len(df)-1]['tick']
    for i in range(len(df)-1, -1, -1):
        l = df.iloc[i]
        if i % 1000 == 0:
            print(f"{k}/{len(paths)} -- " + path + f" -- {i}/{len(df)} lines")

        notetoken = f"n_{l['note']} c_{l['channel']}"
        if l['velocity'] == 0 or l['type'] == 'note_off':
            notepair[notetoken] = [l['tick'],l['velocity'],prev_end-l['tick']] #[tick, velocity, step]
            prev_end=l['tick']
        elif l['velocity'] > 0 and l['type'] == 'note_on':
            if notetoken in notepair:
                notes.append(l['note'])
                steps.append(notepair[notetoken][2])
                durations.append(roundnote(notepair[notetoken][0]-l['tick']))
                channels.append(l['channel'])
                velocities.append(notepair[notetoken][1])
                notepair.pop(notetoken)
            else:
                nopairlines.append(f"{i}.{notetoken}")

    notes.reverse()
    steps.reverse()
    durations.reverse()
    channels.reverse()
    velocities.reverse()
    dict = {'note': notes, 'step': steps, 'duration': durations, 'channel': channels, 'velocity': velocities}
    output = pd.DataFrame(dict)

    split = "test" if random.random() > 0.8 else "train"

    outpath = data_path/"data"/split/dirname/filename
    output.to_csv(outpath)
