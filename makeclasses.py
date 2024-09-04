from pathlib import Path
import os

data_path = Path("./composer-classifier/data")

classes = ["Bach", "Beethoven", "Brahms", "Chopin", "Mozart"]

for i in classes:
    testpath = data_path/"test"/i
    trainpath = data_path/"train"/i
    os.makedirs(testpath)
    os.makedirs(trainpath)