#use this script to augment the data by pitching up/down, or stretching the duration of the pieces
from pathlib import Path
import os
import pandas as pd

data_path = Path("./composer-classifier")

piece_path = data_path/"data"

paths = list(Path(piece_path).glob("*/*.csv"))

stretch = 0.9
pitch = 0 #+/- pitch

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

    for i in range(len(df)):
        l = df.iloc[i]
        if i % 100 == 0:
            print(f"{k}/{len(paths)} -- " + path + f" -- {i}/{len(df)} lines")
        notes.append(l['note']+pitch)
        steps.append(l['step']*stretch)
        durations.append(l['duration']*stretch)
        channels.append(l['channel'])
        velocities.append(l['velocity'])
    
    filename = f"s{stretch}p{pitch}-{filename}"

    dict = {'note': notes, 'step': steps, 'duration': durations, 'channel': channels, 'velocity': velocities}
    output = pd.DataFrame(dict)

    outpath = data_path/"augment"/dirname/filename
    output.to_csv(outpath)
