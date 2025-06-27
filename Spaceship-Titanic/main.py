import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn

### Data Cleaning and Preparation

train_dataset = pd.read_csv('datasets/train.csv')
test_dataset = pd.read_csv('datasets/test.csv')

## Data Cleaning

# Dropping useless column - PassengerId and Name
train_dataset.drop(['PassengerId', 'Name'], axis=1, inplace=True)
test_dataset.drop(['PassengerId', 'Name'], axis=1, inplace=True)
print(train_dataset.columns)

# Checking for missing and duplicate values
print(train_dataset.isnull().sum())
print(test_dataset.isnull().sum())
print(train_dataset.duplicated().sum())
print(test_dataset.duplicated().sum())

# Drop missing and duplicated values
train_dataset.dropna(inplace = True)
test_dataset.dropna(inplace = True)
train_dataset.drop_duplicates(inplace = True)
test_dataset.drop_duplicates(inplace = True)

# Reorganize the index
train_dataset.reset_index()
test_dataset.reset_index()

print(len(train_dataset))
print(len(test_dataset))

# for column in train_dataset.columns:
#     print(train_dataset[column].value_counts().sort_index())

## Data Normalization


Continuous_columns = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

# Splitting columns Cabin into three different columns - dec/num/side and dropping column Cabin

train_dataset[['Cabin_d', 'Cabin_n', 'Cabin_s']] = train_dataset['Cabin'].str.split('/', expand = True)
test_dataset[['Cabin_d', 'Cabin_n', 'Cabin_s']] = test_dataset['Cabin'].str.split('/', expand = True)

train_dataset.drop('Cabin', axis = 1, inplace = True)
test_dataset.drop('Cabin', axis = 1, inplace = True)

# binning categorical data
Categorical_columns = ['HomePlanet', 'CryoSleep', 'Cabin_d', 'Cabin_n', 'Cabin_s', 'Destination', 'VIP', 'Transported']
new_categorical_column = [col for col in Categorical_columns if col != 'Cabin_n']

binned_train_dataset = pd.get_dummies(train_dataset[new_categorical_column])
binned_test_dataset = pd.get_dummies(test_dataset[[col for col in new_categorical_column if col != 'Transported']])

# Concatenate actual data with the binned data
train_dataset = pd.concat([train_dataset.drop(columns=new_categorical_column), binned_train_dataset], axis=1)
test_dataset = pd.concat([test_dataset.drop(columns=[col for col in new_categorical_column if col != 'Transported']), binned_test_dataset], axis=1)

# Transferring clean dataset into new .csv
# train_dataset.to_csv('cleaned_train_dataset.csv', index = False)
# test_dataset.to_csv('cleaned_test_dataset.csv', index = False)


