import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier

import seaborn as sns
import matplotlib.pyplot as plt

train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')
sample_submission = pd.read_csv('dataset/sample_submission.csv')

train_col = [col for col in train.columns if col not in ['y', 'id']]
id = test['id'].copy()

# Dropping useless columns

train.drop(columns=['id'], inplace=True)
test.drop(columns=['id'], inplace=True)

# Describe the data

train.info()

# Correlation of the data

train.corr(numeric_only=True, method='spearman').style.background_gradient(cmap='coolwarm')

# Encoding categorical data

cat_col = train.select_dtypes(include=['object']).columns.to_list()
encoder = LabelEncoder()

for col in cat_col:
    train[col] = encoder.fit_transform(train[col])
    test[col] = encoder.fit_transform(test[col])

# Scaling the data

scaler = StandardScaler()
train[train_col] = pd.DataFrame(scaler.fit_transform(train[train_col]), columns=train_col)
test = pd.DataFrame(scaler.transform(test), columns=test.columns)

# Train the model on the dataset

model = CatBoostClassifier()
model.fit(train[train_col], train['y'])

# Predict the test data

predictions = model.predict(test)
pred_probs = model.predict_proba(test)

# Creating submission file

submission = pd.DataFrame({'id': id, 'y': pred_probs[:, 0]})
submission.to_csv('submission.csv', index=False)
