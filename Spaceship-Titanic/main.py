import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam

### Data Cleaning and Preparation

train_dataset = pd.read_csv('datasets/train.csv')
test_dataset = pd.read_csv('datasets/test.csv')

## Data Cleaning

# Store PassengerId for test set before dropping
test_passenger_ids = test_dataset['PassengerId'].copy()

# Dropping useless column - PassengerId and Name
train_dataset.drop(['PassengerId', 'Name'], axis=1, inplace=True)
test_dataset.drop(['PassengerId', 'Name'], axis=1, inplace=True)
print(train_dataset.dtypes)

# Checking for missing and duplicate values
print(train_dataset.isnull().sum())
print(test_dataset.isnull().sum())
print(train_dataset.duplicated().sum())
print(test_dataset.duplicated().sum())

# Handling missing and duplicate values
# FIT ONLY ON TRAINING DATA to avoid data leakage

mean_imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
mode_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

numeric_columns = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
non_numeric_columns = ['HomePlanet', 'Destination', 'CryoSleep', 'VIP']

# Fit on training data only, then transform both
train_dataset[numeric_columns] = mean_imputer.fit_transform(train_dataset[numeric_columns])
test_dataset[numeric_columns] = mean_imputer.transform(test_dataset[numeric_columns])

train_dataset[non_numeric_columns] = mode_imputer.fit_transform(train_dataset[non_numeric_columns])
test_dataset[non_numeric_columns] = mode_imputer.transform(test_dataset[non_numeric_columns])

# Reorganize the index
train_dataset.reset_index(inplace=True, drop=True)
test_dataset.reset_index(inplace=True, drop=True)

print(len(train_dataset))
print(len(test_dataset))

## Data Normalization

# Splitting columns Cabin into three different columns - dec/num/side and dropping column Cabin
train_dataset[['Cabin_d', 'Cabin_n', 'Cabin_s']] = train_dataset['Cabin'].str.split('/', expand=True)
test_dataset[['Cabin_d', 'Cabin_n', 'Cabin_s']] = test_dataset['Cabin'].str.split('/', expand=True)

train_dataset.drop('Cabin', axis=1, inplace=True)
test_dataset.drop('Cabin', axis=1, inplace=True)

# Convert Cabin_n to numeric
train_dataset['Cabin_n'] = pd.to_numeric(train_dataset['Cabin_n'], errors='coerce')
test_dataset['Cabin_n'] = pd.to_numeric(test_dataset['Cabin_n'], errors='coerce')

# Fill NaN values in Cabin_n
train_dataset['Cabin_n'].fillna(train_dataset['Cabin_n'].median(), inplace=True)
test_dataset['Cabin_n'].fillna(train_dataset['Cabin_n'].median(), inplace=True)

# binning categorical data
categorical_columns = ['HomePlanet', 'CryoSleep', 'Cabin_d', 'Cabin_s', 'Destination', 'VIP']
if 'Transported' in train_dataset.columns:
    target_col = ['Transported']
else:
    target_col = []

# Get dummies for categorical columns
train_dummies = pd.get_dummies(train_dataset[categorical_columns], prefix=categorical_columns)
test_dummies = pd.get_dummies(test_dataset[categorical_columns], prefix=categorical_columns)

# Ensure both datasets have the same columns
all_columns = sorted(set(train_dummies.columns) | set(test_dummies.columns))
for col in all_columns:
    if col not in train_dummies.columns:
        train_dummies[col] = 0
    if col not in test_dummies.columns:
        test_dummies[col] = 0

train_dummies = train_dummies[all_columns]
test_dummies = test_dummies[all_columns]

# Concatenate numerical features with dummy variables
numerical_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Cabin_n']
train_final = pd.concat([train_dataset[numerical_cols + target_col], train_dummies], axis=1)
test_final = pd.concat([test_dataset[numerical_cols], test_dummies], axis=1)

### Data Training and Testing - Deep Learning Classification

## Creating the data for training and testing

features_columns = [col for col in train_final.columns if col != 'Transported']
X_train = train_final[features_columns].values
X_test = test_final[features_columns].values
y_train = train_final['Transported'].astype(int).values

# Scale the data - FIT ONLY ON TRAINING DATA
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Only transform, don't fit!

# Check for any remaining NaN or infinite values
print("NaN in X_train_scaled:", np.isnan(X_train_scaled).sum())
print("NaN in X_test_scaled:", np.isnan(X_test_scaled).sum())
print("Inf in X_train_scaled:", np.isinf(X_train_scaled).sum())
print("Inf in X_test_scaled:", np.isinf(X_test_scaled).sum())

# Replace any remaining NaN or inf values
X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0, posinf=1.0, neginf=-1.0)
X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0, posinf=1.0, neginf=-1.0)

# Turning the data into tensor
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

print(f"Feature shape: {X_train_tensor.shape}")
print(f"Target shape: {y_train_tensor.shape}")

## Defining the model with better initialization and architecture

class DeepLearning(nn.Module):
    def __init__(self, input_features, hidden_features, output_features, dropout=0.3):
        super(DeepLearning, self).__init__()

        self.layer_stack = nn.Sequential(
            nn.Linear(input_features, hidden_features),
            nn.ReLU(),  # Changed from Tanh to ReLU
            nn.BatchNorm1d(hidden_features),  # Added batch normalization
            nn.Dropout(dropout),
            nn.Linear(hidden_features, hidden_features // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_features // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_features // 2, hidden_features // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features // 4, output_features),
        )
        
        # Initialize weights properly
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.layer_stack(x)

## Instantiating the model

model = DeepLearning(X_train_tensor.shape[1], 256, 1)  # Increased hidden size

## Preparing for criterion and optimizer

loss_fn = BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=0.0001)  # Reduced learning rate

## Training the model 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 250

model = model.to(device)
X_train_tensor = X_train_tensor.to(device)
X_test_tensor = X_test_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)

print("Unique values in y_train_tensor:", y_train_tensor.unique())

# Training loop with gradient clipping
for epoch in range(epochs):
    model.train()
    
    # Forward pass
    y_pred = model(X_train_tensor)
    
    # Calculate loss
    loss = loss_fn(y_pred.squeeze(), y_train_tensor)
    
    # Check for NaN loss
    if torch.isnan(loss):
        print(f"NaN loss detected at epoch {epoch}")
        break
    
    # Zero grad optimizer
    optimizer.zero_grad()
    
    # Loss backward
    loss.backward()
    
    # Gradient clipping to prevent explosion
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # Step the optimizer
    optimizer.step()

    if epoch % 10 == 0 or epoch == (epochs - 1):
        print(f"Epoch: {epoch} | Train loss: {loss:.4f}")

## Predicting X_test

model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor)
    y_pred_probs = torch.sigmoid(y_pred)
    y_pred_binary = (y_pred_probs >= 0.5).squeeze().cpu().numpy()

## Submission

submission = pd.DataFrame({
    'PassengerId': test_passenger_ids, 
    'Transported': y_pred_binary.astype(bool)
})
submission.to_csv('submission.csv', index=False)
print("Submission file created successfully!")
print(f"Predictions shape: {y_pred_binary.shape}")
print(f"Submission shape: {submission.shape}")