#use this script to augment the data by pitching up/down, or stretching the duration of the pieces
from pathlib import Path
import os
import pandas as pd

data_path = Path("./composer-classifier")

piece_path = data_path/"data"/"train"

paths = list(Path(piece_path).glob("*/*.csv"))


#change augmentation vars here
stretch = 1
pitch = 4 #+/- pitch


for k in range(len(paths)):
    filename = os.path.basename(paths[k])
    if filename[:4] == 'aug_':
        print(f"skip (augment) {filename}")
    else:
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
        
        filename = f"aug_s{stretch}p{pitch}-{filename}_"

        dict = {'note': notes, 'step': steps, 'duration': durations, 'channel': channels, 'velocity': velocities}
        output = pd.DataFrame(dict)

        outpath = piece_path/dirname/filename
        output.to_csv(outpath)
