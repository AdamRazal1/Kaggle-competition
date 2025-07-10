import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from torch import nn
from torch import optim

pd.set_option('display.max_rows', None)

# loading the dataset

train_data  = pd.read_csv('datasets/train.csv')
test_data  = pd.read_csv('datasets/test.csv')

# Dropping ambiguous features

train_data.drop(['MiscFeature', 'MiscVal', 'Id', 'PoolQC', 'Fence', 'Alley'], inplace=True, axis=1)
test_data.drop(['MiscFeature', 'MiscVal', 'Id', 'PoolQC', 'Fence', 'Alley'], inplace=True, axis=1)

# Categorizing the features

categorical_features = train_data.select_dtypes(include=['object']).columns.to_list()
numerical_features = train_data.select_dtypes(include=['int64', 'float64']).columns.to_list()
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

new_categorical_features = encoded_categorical.columns.to_list()
new_combined_features = new_categorical_features + numerical_features

# Scaling numerical features

scaler = StandardScaler()
scaled_numerical = scaler.fit_transform(train_data[[col for col in numerical_features if col != 'SalePrice']])
scaled_numerical = pd.DataFrame(scaled_numerical, columns=[col for col in numerical_features if col != 'SalePrice'])
scaled_numerical['SalePrice'] = train_data['SalePrice']

# combining encoded and scaled features

combined_data = pd.concat([encoded_categorical, scaled_numerical], axis=1)
print('This is combined data', combined_data.shape)

# Storing cleaned data into a csv file

# combined_data.to_csv('cleaned_train_data.csv')

# Feature Selection via - ANOVA F-test

X = combined_data[[col for col in new_combined_features if col != 'SalePrice']]
y = combined_data['SalePrice']

selector = SelectKBest(score_func=f_classif, k=10)
selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()].to_list()

# Turning the X and y into tensors

X = combined_data[selected_features].values
X = torch.tensor(X, dtype = torch.float32)
y = torch.tensor(y.values, dtype = torch.float32)


print('this is X',X)
print(X.shape)
print('this is y', y)
print(y.shape)

# Preparing the Deep Learning model

class deepLearningModel(nn.Module):
    def __init__(self, input_features, hidden_features, dropout = 0.3):
        super(deepLearningModel, self).__init__()

        self.layer_stack = nn.Sequential(
            nn.Linear(input_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, hidden_features // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features // 2, 1),
        )
    
    def forward(self, x):
        return self.layer_stack(x)

# Initializing the model

model = deepLearningModel(input_features=X.shape[1], hidden_features=128)

# Intializing the loss and optimizer

loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

# Preparing for training

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = model.to(device)
X = X.to(device)
y = y.to(device)

# training phase

num_epochs = 3000

for epoch in range(num_epochs):
    model.train()

    # Forward pass
    outputs = model(X)
    loss_fn = loss(outputs, y)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss_fn.backward()
    optimizer.step()

    # Evaluate on validation set
    if (epoch+1) % 10 == 0:
        model.eval()
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss_fn.item():.4f}")


