import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.nn import BCELoss
from torch.optim import Adam

### Data Cleaning and Preparation

train_dataset = pd.read_csv('datasets/train.csv')
test_dataset = pd.read_csv('datasets/test.csv')

## Data Cleaning

# Dropping useless column - PassengerId and Name
train_dataset.drop(['PassengerId', 'Name'], axis=1, inplace=True)
test_dataset.drop([ 'Name'], axis=1, inplace=True)
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
train_dataset.reset_index(inplace = True, drop = True)
test_dataset.reset_index()

print(len(train_dataset))
print(len(test_dataset))

# for column in train_dataset.columns:
#     print(train_dataset[column].value_counts().sort_index())

## Data Normalization

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


### Data Training and Testing - Deep Learning Classification

## Creating the data for training and testing - X_train, X_test, y_train, y_test

features_column = [col for col in train_dataset if col != 'Transported']
label = ['Transported']

# Extracting the values from the dataset
X_train = train_dataset[features_column].values
X_test = test_dataset[features_column].values
y_train = train_dataset[label[0]].values

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# Turning the data into tensor
X_train_tensor = torch.tensor(X_train_scaled, dtype = torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype = torch.float32)
y_train_tensor = torch.tensor(y_train, dtype = torch.float32)

## Defining the model

class DeepLearning(nn.Module):
    def __init__(self, input_features, hidden_features, output_features, dropout=0.3):
        super(DeepLearning, self).__init__()

        self.layer_stack = nn.Sequential(
            nn.Linear(input_features, hidden_features),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, hidden_features // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features // 2, output_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layer_stack(x)
    
## Instantiating the model

model = DeepLearning(X_test_tensor.shape[1], 128, 1)

## Preparing for criterion and optimizer

loss_fn = BCELoss()
optimizer = Adam(model.parameters(), lr = 0.001)

## Training the model 

# Initializing epoch and loading the model and data into the device

device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 100

model = model.to(device)
X_train = X_train_tensor.to(device)
X_test = X_test_tensor.to(device)
y_train = y_train_tensor.to(device)

print(y_train)

for epoch in range(epochs):

    ## Training phase
    model.train()
    # 1. Forward pass
    y_pred = model(X_train)
    # 2. Calculate loss
    loss = loss_fn(y_pred, y_train.unsqueeze(dim = 1))
    # 3. Zero grad optimizer
    optimizer.zero_grad()
    # 4. Loss backward
    loss.backward()
    # 5. Step the optimizer
    optimizer.step()

    if epoch % 10 == 0 or epoch == (epochs - 1):
            print(f"Epoch: {epoch} | Train loss: {loss}")


## Predicting X_test

model.eval()
y_pred = model(X_test)
y_pred = y_pred >= 0.5
y_pred = y_pred.squeeze(dim = 1).cpu().detach().numpy()

## Submission

submission = pd.DataFrame({'PassengerId': test_dataset['PassengerId'], 'Transported': y_pred})
submission.to_csv('submission.csv', index = False)