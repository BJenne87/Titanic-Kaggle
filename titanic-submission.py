import numpy as np
import pandas as pd

train_data = pd.read_csv("/home/PapaIV/Desktop/Python/train.csv")

# open a filtered set of survivors
survivors = pd.read_csv("/home/PapaIV/Desktop/Python/survivors.csv")
print(train_data.info())
