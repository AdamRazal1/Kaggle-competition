import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from torch import nn

pd.set_option('display.max_rows', None)

# loading the dataset

train_data  = pd.read_csv('datasets/train.csv')
test_data  = pd.read_csv('datasets/test.csv')

# Dropping ambiguous features

train_data.drop(['MiscFeature', 'MiscVal', 'Id', 'PoolQC', 'Fence', 'Alley'], inplace=True, axis=1)
test_data.drop(['MiscFeature', 'MiscVal', 'Id', 'PoolQC', 'Fence', 'Alley'], inplace=True, axis=1)

# Categorizing the features

categorical_features = train_data.select_dtypes(include=['object']).columns.to_list()
numerical_features = train_data.select_dtypes(include=['number']).columns.to_list()
combine_features = categorical_features + numerical_features

# Checking the number of unique values per feature

for col in categorical_features:
    print(f"\nValue counts for '{col}':")
    print(train_data[col].value_counts())


# Checking and handling missing values

categorical_imputer = SimpleImputer(strategy = 'most_frequent')
numerical_imputer = SimpleImputer(strategy = 'mean')

train_data[categorical_features] = categorical_imputer.fit_transform(train_data[categorical_features])

train_data[numerical_features] = numerical_imputer.fit_transform(train_data[numerical_features])

test_data[categorical_features] = categorical_imputer.fit_transform(test_data[categorical_features])

test_data[[col for col in numerical_features if col != 'SalePrice']] = numerical_imputer.fit_transform(test_data[[col for col in numerical_features if col != 'SalePrice']])

print(test_data.isnull().sum())
train_data.isnull().sum()

# encoding categorical features

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_categorical = encoder.fit_transform(train_data[categorical_features])
encoded_categorical = pd.DataFrame(encoded_categorical, columns=encoder.get_feature_names_out(categorical_features))
print(encoded_categorical.shape)

# Scaling numerical features

scaler = StandardScaler()
scaled_numerical = scaler.fit_transform(train_data[numerical_features])
scaled_numerical = pd.DataFrame(scaled_numerical, columns=numerical_features)
print(scaled_numerical.shape)

# combining encoded and scaled features

combined_data = pd.concat([encoded_categorical, scaled_numerical], axis=1)

# Storing cleaned data into a csv file

# combined_data.to_csv('cleaned_train_data.csv')

# Feature Selection via - ANOVA F-test

X = combined_data[[col for col in combine_features if col != 'SalesPrice']]
y = combined_data['SalesPrice']

selector = SelectKBest(score_func=f_classif, k=10)
seleted = selector.fit_transform(X, y)


