#converts csv midi files to a note sequence csv
#makes each note has a duration, rather than a start/end time
#start from the back and append, then reverse all the lists
#also split the data into train/test sets 80-20
from pathlib import Path
import os
import pandas as pd
import random

data_path = Path("./composer-classifier")

piece_path = data_path/"raws"

paths = list(Path(piece_path).glob("*/*.csv"))

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
        if i % 100 == 0:
            print(f"{k}/{len(paths)} -- " + path + f" -- {i}/{len(df)} lines")

        notetoken = f"n_{l['note']} c_{l['channel']}"
        if l['velocity'] == 0 or l['type'] == 'note_off':
            notepair[notetoken] = [l['tick'],l['velocity'],prev_end-l['tick']] #[tick, velocity, step]
            prev_end=l['tick']
        elif l['velocity'] > 0 and l['type'] == 'note_on':
            if notetoken in notepair:
                notes.append(l['note'])
                steps.append(notepair[notetoken][2])
                durations.append(notepair[notetoken][0]-l['tick'])
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
