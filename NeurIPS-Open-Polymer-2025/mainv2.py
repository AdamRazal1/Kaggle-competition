import pandas as pd
import torch

from torch import optim
from torch import nn
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import r_regression
from sklearn.ensemble import RandomForestRegressor

pd.set_option('display.max_rows', None)

train = pd.read_csv('datasets/train.csv')
test = pd.read_csv('datasets/test.csv')
id = test['id'].copy()

## Preprocessing - Training

# Visualizing the data information

print(train.info())
print(train.dtypes)

# Vectorizing the SMILES

all_unique_smiles = sorted(set(''.join(train['SMILES'].dropna().astype(str))))

def count_chars(smiles, all_unique_chars):

    smiles_count = {('char_' + symbol): 0 for symbol in all_unique_chars}
    for letter in smiles:
        if letter in all_unique_chars:
            smiles_count['char_' + letter] += 1 
    return smiles_count

# Apply and expand each dictionary into its own columns

smile_df = train['SMILES'].apply(lambda x: pd.Series(count_chars(x, all_unique_smiles)))
test_smile_df = test['SMILES'].apply(lambda x: pd.Series(count_chars(x, all_unique_smiles)))

# Merge smile_df with the train dataset

train = pd.concat([train, smile_df], axis=1)
test = pd.concat([test, test_smile_df], axis=1)

# Storing the train df #1

train.to_csv('#1-after-smiles-processing.csv')

# Converting SMILES to MOL

train['Mol'] = train['SMILES'].apply(Chem.MolFromSmiles)
test['Mol'] = test['SMILES'].apply(Chem.MolFromSmiles)

# Drop rows where RDKit failed to generate a molecule

train = train[train['Mol'].notnull()].reset_index(drop=True)
test = test[test['Mol'].notnull()].reset_index(drop=True)

# Storing the train df #2

train.to_csv('#2-after-mol-convertion.csv')

# Extracting Molecular Informations

descriptor = {desc: [] for desc, _ in Descriptors.descList}

for mol in train['Mol']:
    for desc, func in Descriptors.descList:
        descriptor[desc].append(func(mol))

# Combining the train df with descriptor df

train = pd.concat([train, pd.DataFrame(descriptor)], axis=1)
test = pd.concat([test, pd.DataFrame(descriptor)], axis=1)

# eliminating columns containing > 50% missing values from descriptors

thres = 0.5 * (len(train.values))
eliminated_columns = [col for col in descriptor if train[col].isnull().sum() >= thres]
print(f'number of columns to be eliminated : {len(eliminated_columns)}')

train.drop(eliminated_columns, axis=1, inplace=True)
test.drop(eliminated_columns, axis=1, inplace=True)

# Storing the train df #3

train.to_csv('#3-after-descriptor-interpretion.csv')

# handling the input and label features

label_cols = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
input_cols = [col for col in train.columns if col not in label_cols + ['SMILES', 'Mol', 'id']]

# Dropping id, smiles, and mol columns

train.drop(['id', 'SMILES', 'Mol'], axis=1, inplace=True)
test.drop(['id', 'SMILES', 'Mol'], axis=1, inplace=True)

# Storing the train df #4

train.to_csv('#4-after-eliminate-unuse-columns.csv')

# Seperating into different df for each labels

tg = train[input_cols + ['Tg']]
ffv = train[input_cols + ['FFV']]
tc = train[input_cols + ['Tc']]
density = train[input_cols + ['Density']]
rg = train[input_cols + ['Rg']]

# Checking for missing values for each target label df

for item, name in [(tg, 'Tg'), (ffv, 'FFV'), (tc, 'Tc'), (density, 'Density'), (rg, 'Rg')]:
    print(f'number of rows of missing values for {name} : {item.isnull().sum()}')

# Handling missing values for each target label df

for item, name in [(tg, 'Tg'), (ffv, 'FFV'), (tc, 'Tc'), (density, 'Density'), (rg, 'Rg')]:
    item.dropna(subset=[name],inplace=True)
    item.reset_index(drop=True, inplace=True)

# Checking any missing values for the taget  label

for item, name in [(tg, 'Tg'), (ffv, 'FFV'), (tc, 'Tc'), (density, 'Density'), (rg, 'Rg')]:
    print(f'number of rows of missing values for {name} : {item.isnull().sum()}')

# Initialize and store scalers for each label
scaler_tg = StandardScaler()
scaler_ffv = StandardScaler()
scaler_tc = StandardScaler()
scaler_density = StandardScaler()
scaler_rg = StandardScaler()
scaler_test = StandardScaler()

# Fit-transform training data
scaled_tg = scaler_tg.fit_transform(tg)
scaled_ffv = scaler_ffv.fit_transform(ffv)
scaled_tc = scaler_tc.fit_transform(tc)
scaled_density = scaler_density.fit_transform(density)
scaled_rg = scaler_rg.fit_transform(rg)

scaled_test = scaler_test.fit_transform(test)


# Converting the data type of the data for training

scaled_tg = pd.DataFrame(scaled_tg, columns=tg.columns, index=tg.index)
scaled_ffv = pd.DataFrame(scaled_ffv, columns=ffv.columns, index=ffv.index)
scaled_tc = pd.DataFrame(scaled_tc, columns=tc.columns, index=tc.index)
scaled_density = pd.DataFrame(scaled_density, columns=density.columns, index=density.index)
scaled_rg = pd.DataFrame(scaled_rg, columns=rg.columns, index=rg.index)

scaled_test = pd.DataFrame(scaled_test, columns=test.columns, index=test.index)

# Checking the shape for each target label df

for item, name in [(scaled_tg, 'Tg'), (scaled_ffv, 'FFV'), (scaled_tc, 'Tc'), (scaled_density, 'Density'), (scaled_rg, 'Rg')]:
    print(f'shape of {name} : {item.shape}')

# Storing the target labels of each df

for item, name in [(scaled_tg, 'Tg'), (scaled_ffv, 'FFV'), (scaled_tc, 'Tc'), (scaled_density, 'Density'), (scaled_rg, 'Rg'), (scaled_test, 'test')]:
    print(f"{name} successfully stored to csv file {item.to_csv(name + '.csv')}")

# df of each target label X and y

tg_df_x = scaled_tg[[col for col in scaled_tg.columns if col != 'Tg']]
tg_df_y = scaled_tg['Tg']

ffv_df_x = scaled_ffv[[col for col in scaled_ffv.columns if col != 'FFV']]
ffv_df_y = scaled_ffv['FFV']

tc_df_x = scaled_tc[[col for col in scaled_tc.columns if col != 'Tc']]
tc_df_y = scaled_tc['Tc']

density_df_x = scaled_density[[col for col in scaled_density.columns if col != 'Density']]
density_df_y = scaled_density['Density']

rg_df_x = scaled_rg[[col for col in scaled_rg.columns if col != 'Rg']]
rg_df_y = scaled_rg['Rg']


# feature selection for each target label df

tg_score = r_regression(tg_df_x.values, tg_df_y.values)
ffv_score = r_regression(ffv_df_x.values, ffv_df_y.values)
tc_score = r_regression(tc_df_x.values, tc_df_y.values)
density_score = r_regression(density_df_x.values, density_df_y.values)
rg_score = r_regression(rg_df_x.values, rg_df_y.values)

selected_features_tg = []
selected_features_ffv = []
selected_features_tc = []
selected_features_density = []
selected_features_rg = []

for i in range(tg_df_x.shape[1]):
    if tg_score[i] > 0.2 or tg_score[i] < -0.2:
        selected_features_tg.append(i)

for i in range(ffv_df_x.shape[1]):
    if ffv_score[i] > 0.2 or ffv_score[i] < -0.2:
        selected_features_ffv.append(i)

for i in range(tc_df_x.shape[1]):
    if tc_score[i] > 0.2 or tc_score[i] < -0.2:
        selected_features_tc.append(i)

for i in range(density_df_x.shape[1]):
    if density_score[i] > 0.2 or density_score[i] < -0.2:
        selected_features_density.append(i)

for i in range(rg_df_x.shape[1]):
    if rg_score[i] > 0.4 or rg_score[i] < -0.4:
        selected_features_rg.append(i)

tg_features = tg_df_x.iloc[:, selected_features_tg].values
ffv_features = ffv_df_x.iloc[:, selected_features_ffv].values
tc_features = tc_df_x.iloc[:, selected_features_tc].values
density_features = density_df_x.iloc[:, selected_features_density].values
rg_features = rg_df_x.iloc[:, selected_features_rg].values

scaled_test = scaled_test.dropna(axis=0, how='any')

tg_features_test = scaled_test.iloc[:, selected_features_tg].values
ffv_features_test = scaled_test.iloc[:, selected_features_ffv].values
tc_features_test = scaled_test.iloc[:, selected_features_tc].values
density_features_test = scaled_test.iloc[:, selected_features_density].values
rg_features_test = scaled_test.iloc[:, selected_features_rg].values

# using random forest regressor as the model for each target

model_tg = RandomForestRegressor(n_estimators= 200,random_state=42, criterion='absolute_error')
model_ffv = RandomForestRegressor(n_estimators= 200,random_state=42, criterion='absolute_error')
model_tc = RandomForestRegressor(n_estimators= 200,random_state=42, criterion='absolute_error')
model_density = RandomForestRegressor(n_estimators= 200,random_state=42, criterion='absolute_error')
model_rg = RandomForestRegressor(n_estimators= 200,random_state=42, criterion='absolute_error')

model_tg.fit(tg_features, tg_df_y)
model_ffv.fit(ffv_features, ffv_df_y)
model_tc.fit(tc_features, tc_df_y)
model_density.fit(density_features, density_df_y)
model_rg.fit(rg_features, rg_df_y)

y_pred_tg = model_tg.predict(tg_features_test)
y_pred_ffv = model_ffv.predict(ffv_features_test)
y_pred_tc = model_tc.predict(tc_features_test)
y_pred_density = model_density.predict(density_features_test)
y_pred_rg = model_rg.predict(rg_features_test)

import numpy as np

# A helper function to make the process repeatable
def unscale_predictions(predictions, original_df, target_name, scaler):
    """Unscales predictions using the original scaler."""
    # Create a temporary dataframe with the correct shape and columns
    temp_df = pd.DataFrame(np.zeros((len(predictions), len(original_df.columns))), columns=original_df.columns)
    
    # Place the scaled predictions into the target column
    temp_df[target_name] = predictions
    
    # Inverse transform the entire dataframe
    unscaled_data = scaler.inverse_transform(temp_df)
    
    # Extract the unscaled predictions from the correct column
    target_col_index = original_df.columns.get_loc(target_name)
    unscaled_predictions = unscaled_data[:, target_col_index]
    
    return unscaled_predictions

# Unscale all your predictions
final_preds_tg = unscale_predictions(y_pred_tg, tg, 'Tg', scaler_tg)
final_preds_ffv = unscale_predictions(y_pred_ffv, ffv, 'FFV', scaler_ffv)
final_preds_tc = unscale_predictions(y_pred_tc, tc, 'Tc', scaler_tc)
final_preds_density = unscale_predictions(y_pred_density, density, 'Density', scaler_density)
final_preds_rg = unscale_predictions(y_pred_rg, rg, 'Rg', scaler_rg)

print(final_preds_tg)
print(final_preds_ffv)
print(final_preds_tc)
print(final_preds_density)
print(final_preds_rg)