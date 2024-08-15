#converts csv midi files to a note sequence csv
#makes each note has a duration, rather than a start/end time
from pathlib import Path
import os
import pandas as pd

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

    #find first note for accurate step
    for i in range(len(df)):
        l = df.iloc[i]
        if l['type'] == 'note_on':
            prev_start = l['tick']
            break

    for i in range(len(df)):
        l = df.iloc[i]
        if i % 100 == 0:
            print(f"{k}/{len(paths)} -- " + path + f" -- {i}/{len(df)} lines")

        notetoken = f"{l['note']}+{l['channel']}"
        if l['velocity'] != 0 and l['type'] == 'note_on':
            notepair[notetoken] = [l['tick'],l['velocity'],l['tick']-prev_start] #[tick, velocity, step]
            prev_start = l["tick"]
        elif l['velocity'] == 0 or l['type'] == 'note_off':
            if notetoken in notepair:
                notes.append(l['note'])
                steps.append(notepair[notetoken][2])
                durations.append(l['tick']-notepair[notetoken][0])
                channels.append(l['channel'])
                velocities.append(notepair[notetoken][1])
                notepair.pop(notetoken)
            else:
                nopairlines.append(f"{i}.{notetoken}")

    #print("unpaired lines: "+ str(nopairlines))
    #print("unpaired notes: "+ str(notepair))
    dict = {'note': notes, 'step': steps, 'duration': durations, 'channel': channels, 'velocity': velocities}
    output = pd.DataFrame(dict)

    outpath = data_path/"data"/"train"/dirname/filename
    output.to_csv(outpath)
