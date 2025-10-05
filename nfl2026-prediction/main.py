import pandas as pd
import numpy as np
import glob 
import os

print(os.getcwd())


# load weekly input data

input_dir = glob.glob(os.path.join('dataset/train', "input_2023_w*.csv"))

input = [pd.read_csv(file) for file in input_dir]