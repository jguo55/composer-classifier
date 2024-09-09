classifies a sequence of notes into which composer was most likely to compose them.

dataset from: [https://www.kaggle.com/datasets/vincentloos/classical-music-midi-as-csv](url)

composers: Bach, Beethoveen, Brahms, Mozart, Chopin

composers chosen from availability of data as well as influence/distinctiveness.

TO TRAIN ON NEW DATA
1. place files into raws, with the classification being the directory
2. run seqconv.py
3. (OPTIONAL) run augment.py
4. run buildvocab.py
5. run train_lstm.py

midi file csv-> note sequence with duration csv
note sequence csv -> tokens
tokens -> classification

might add converting from raw midi to midi files in the future