import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from torch import nn

# loading the dataset

train_data  = pd.read_csv('datasets/train.csv')
test_data  = pd.read_csv('datasets/test.csv')

# Dropping ambiguous features

train_data.drop(['MiscFeature', 'MiscVal', 'Id'], inplace=True, axis=1)
test_data.drop(['MiscFeature', 'MiscVal', 'Id'], inplace=True, axis=1)

# Categorizing the features

categorical_features = train_data.select_dtypes(include=['object']).columns
numerical_features = train_data.select_dtypes(include=['number']).columns

# Checking and handling missing values

