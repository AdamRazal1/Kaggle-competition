import sklearn
import pandas as pd
import numpy as np
import rdkit

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error

pd.set_option('display.max_columns', None)

train = pd.read_csv('datasets/train.csv')
test = pd.read_csv('datasets/test.csv')

id = test['id'].copy()

# expanding the rows with extra data

dataset1 = pd.read_csv('datasets/dataset1.csv')
dataset3 = pd.read_csv('datasets/dataset3.csv')
dataset4 = pd.read_csv('datasets/dataset4.csv')

# checking if each row of smiles from tc_mean has same smiles as train

# Canonicalize SMILES
def canon(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol else None

train['canonical'] = train['SMILES'].apply(canon)
test['canonical'] = test['SMILES'].apply(canon)
dataset1['canonical'] = dataset1['SMILES'].apply(canon)
dataset3['canonical'] = dataset3['SMILES'].apply(canon)
dataset4['canonical'] = dataset4['SMILES'].apply(canon)

new_tg = {'canonical': [], 'Tg': []}
new_tc = {'canonical': [], 'Tc': []}
new_ffv = {'canonical': [], 'FFV': []}

# Drop invalid SMILES
train = train.dropna(subset=['canonical'])
dataset1 = dataset1.dropna(subset=['canonical'])
dataset3 = dataset3.dropna(subset=['canonical'])
dataset4 = dataset4.dropna(subset=['canonical'])

# for tg

for smile in dataset3['canonical']:
    if smile not in train['canonical'].values:
        new_tg['canonical'].append(smile)
        new_tg['Tg'].append(dataset3[dataset3['canonical'] == smile]['Tg'].values[0])

# for tc

for smile in dataset1['canonical']:
    if smile not in train['canonical'].values:
        new_tc['canonical'].append(smile)
        new_tc['Tc'].append(dataset1[dataset1['canonical'] == smile]['TC_mean'].values[0])

# for ffv

for smile in dataset4['canonical']:
    if smile not in train['canonical'].values:
        new_ffv['canonical'].append(smile)
        new_ffv['FFV'].append(dataset4[dataset4['canonical'] == smile]['FFV'].values[0])

# for key, item in new_tg.items():
#     print(f'{key} : {item}')

# for key, item in new_tc.items():
#     print(f'{key} : {item}')

# for key, item in new_ffv.items():
#     print(f'{key} : {item}')

# concatenating the train set with the new tg, ffv, and tc

train = pd.concat([train, pd.DataFrame(new_tg)], axis=0)
train = pd.concat([train, pd.DataFrame(new_tc)], axis=0)
train = pd.concat([train, pd.DataFrame(new_ffv)], axis=0)

train.to_csv('added_row_train.csv')

train.drop_duplicates(subset=['canonical'], inplace=True)
train.reset_index(drop=True, inplace=True)

print(train.shape)

# Converting the smiles into mol

train['Mol'] = train['canonical'].apply(Chem.MolFromSmiles)
test['Mol'] = test['canonical'].apply(Chem.MolFromSmiles)

# Drop rows if the rdkit cannot generate the mol for the smiles

train.dropna(subset=['Mol'], inplace=True)
test.dropna(subset=['Mol'], inplace=True)

train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)

# Feature engineering - Getting morgan fingerprints for each mol

morgan_gen = GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
morgan_gen_1 = GetMorganGenerator(radius=1, fpSize=2048, includeChirality=True)

# Get numpy fingerprints
train_fps = np.array([morgan_gen.GetFingerprintAsNumPy(mol) for mol in train['Mol']])
test_fps = np.array([morgan_gen.GetFingerprintAsNumPy(mol) for mol in test['Mol']])

# # Get numpy fingerprints
# train_fps_1 = np.array([morgan_gen_1.GetFingerprintAsNumPy(mol) for mol in train['Mol']])
# test_fps_1 = np.array([morgan_gen_1.GetFingerprintAsNumPy(mol) for mol in test['Mol']])

# Convert to DataFrames
train_fp_df = pd.DataFrame(train_fps, columns=[f'fp_{i}' for i in range(train_fps.shape[1])])
test_fp_df = pd.DataFrame(test_fps, columns=[f'fp_{i}' for i in range(test_fps.shape[1])])

# train_fp_df_1 = pd.DataFrame(train_fps_1, columns=[f'fp_{i}' for i in range(train_fps_1.shape[1])])
# test_fp_df_1 = pd.DataFrame(test_fps_1, columns=[f'fp_{i}' for i in range(test_fps_1.shape[1])])

train = pd.concat([train.reset_index(drop=True), train_fp_df], axis=1)
test = pd.concat([test.reset_index(drop=True), test_fp_df], axis=1)

print(train.shape)

# feature engineering - Extracting Molecular Informations from descriptors and descriptors 3d

desc_list_train = {desc : [] for desc, _ in Descriptors.descList}
desc_list_test = {desc : [] for desc, _ in Descriptors.descList}

for mol in train['Mol']:
    for desc, func in Descriptors.descList:
        desc_list_train[desc].append(func(mol))

for mol in test['Mol']:
    for desc, func in Descriptors.descList:
        desc_list_test[desc].append(func(mol))

# appending added features to the train and test dataset

train = pd.concat([train, pd.DataFrame(desc_list_train)], axis=1)
test = pd.concat([test, pd.DataFrame(desc_list_test)], axis=1)

# eliminating columns containing > 50% missing values from descriptors

thres = 0.5 * (len(train.values))
eliminated_columns = [col for col in desc_list_train.keys() if train[col].isnull().sum() >= thres]
print(f'number of columns to be eliminated : {len(eliminated_columns)}')

train.drop(eliminated_columns, axis=1, inplace=True)
test.drop(eliminated_columns, axis=1, inplace=True)

# Dropping id, smiles and mol columns

train.drop(['id','SMILES', 'Mol', 'canonical'], axis=1, inplace=True)
test.drop(['SMILES', 'Mol', 'canonical'], axis=1, inplace=True)

# seperating the data for each target label df

feature_cols = [col for col in train.columns if col not in ['Tg', 'FFV', 'Tc', 'Density', 'Rg']]

tg = train[feature_cols + ['Tg']].reset_index(drop=True)
ffv = train[feature_cols + ['FFV']].reset_index(drop=True)
tc = train[feature_cols + ['Tc']].reset_index(drop=True)
density = train[feature_cols + ['Density']].reset_index(drop=True)
rg = train[feature_cols + ['Rg']].reset_index(drop=True)

# Drop rows where target labels are missing for each target label df

for item, name in [(tg, 'Tg'), (ffv, 'FFV'), (tc, 'Tc'), (density, 'Density'), (rg, 'Rg')]:
    item.dropna(subset=[name],inplace=True)
    item.reset_index(drop=True, inplace=True)

# Checking for missing values for each target label df

for item, name in [(tg, 'Tg'), (ffv, 'FFV'), (tc, 'Tc'), (density, 'Density'), (rg, 'Rg')]:
    print(f'number of rows of missing values for {name} : {item.isnull().sum()}')

# shape of each df

for item, name in [(tg, 'Tg'), (ffv, 'FFV'), (tc, 'Tc'), (density, 'Density'), (rg, 'Rg')]:
    print(f'shape of {name} : {item.shape}')

# Storing the target labels of each df

for item, name in [(tg, 'Tg'), (ffv, 'FFV'), (tc, 'Tc'), (density, 'Density'), (rg, 'Rg'), (test, 'test')]:
    print(f"{name} successfully stored to csv file {item.to_csv(name + '.csv')}")

# scalingq the data for each target label df

scaler = StandardScaler()

scaled_tg = scaler.fit_transform(tg[feature_cols])
scaled_ffv = scaler.fit_transform(ffv[feature_cols])
scaled_tc = scaler.fit_transform(tc[feature_cols])
scaled_density = scaler.fit_transform(density[feature_cols])
scaled_rg = scaler.fit_transform(rg[feature_cols])
scaled_test = scaler.fit_transform(test[feature_cols])

scaled_tg = pd.DataFrame(scaled_tg, columns=feature_cols, index=tg.index)
scaled_ffv = pd.DataFrame(scaled_ffv, columns=feature_cols, index=ffv.index)
scaled_tc = pd.DataFrame(scaled_tc, columns=feature_cols, index=tc.index)
scaled_density = pd.DataFrame(scaled_density, columns=feature_cols, index=density.index)
scaled_rg = pd.DataFrame(scaled_rg, columns=feature_cols, index=rg.index)
scaled_test = pd.DataFrame(scaled_test, columns=feature_cols, index=test.index)

# Create one StandardScaler per target
scaler_tg = StandardScaler()
scaler_ffv = StandardScaler()
scaler_tc = StandardScaler()
scaler_density = StandardScaler()
scaler_rg = StandardScaler()

# Scale training features
scaled_tg = scaler_tg.fit_transform(tg[feature_cols])
scaled_ffv = scaler_ffv.fit_transform(ffv[feature_cols])
scaled_tc = scaler_tc.fit_transform(tc[feature_cols])
scaled_density = scaler_density.fit_transform(density[feature_cols])
scaled_rg = scaler_rg.fit_transform(rg[feature_cols])

# Scale test features using corresponding scalers
scaled_test_tg = scaler_tg.transform(test[feature_cols])
scaled_test_ffv = scaler_ffv.transform(test[feature_cols])
scaled_test_tc = scaler_tc.transform(test[feature_cols])
scaled_test_density = scaler_density.transform(test[feature_cols])
scaled_test_rg = scaler_rg.transform(test[feature_cols])

# Wrap in DataFrames to preserve structure
scaled_tg = pd.DataFrame(scaled_tg, columns=feature_cols, index=tg.index)
scaled_ffv = pd.DataFrame(scaled_ffv, columns=feature_cols, index=ffv.index)
scaled_tc = pd.DataFrame(scaled_tc, columns=feature_cols, index=tc.index)
scaled_density = pd.DataFrame(scaled_density, columns=feature_cols, index=density.index)
scaled_rg = pd.DataFrame(scaled_rg, columns=feature_cols, index=rg.index)

scaled_test_tg = pd.DataFrame(scaled_test_tg, columns=feature_cols, index=test.index)
scaled_test_ffv = pd.DataFrame(scaled_test_ffv, columns=feature_cols, index=test.index)
scaled_test_tc = pd.DataFrame(scaled_test_tc, columns=feature_cols, index=test.index)
scaled_test_density = pd.DataFrame(scaled_test_density, columns=feature_cols, index=test.index)
scaled_test_rg = pd.DataFrame(scaled_test_rg, columns=feature_cols, index=test.index)

# Converting the data type of the data for training
combined_tg = pd.concat([scaled_tg, tg['Tg']], axis=1)
combined_ffv = pd.concat([scaled_ffv, ffv['FFV']], axis=1)
combined_tc = pd.concat([scaled_tc, tc['Tc']], axis=1)
combined_density = pd.concat([scaled_density, density['Density']], axis=1)
combined_rg = pd.concat([scaled_rg, rg['Rg']], axis=1)

# Feature selection - choosing top 95% of the best feature

X = [col for col in train.columns if col not in ['Tg', 'FFV', 'Tc', 'Density', 'Rg']]

estimator = ExtraTreesRegressor(random_state=42)

selected_features_tg = []
selected_features_ffv = []
selected_features_tc = []
selected_features_density = []
selected_features_rg = []

for item, name, features in [
    (combined_tg, 'Tg', selected_features_tg),
    (combined_ffv, 'FFV', selected_features_ffv),
    (combined_tc, 'Tc', selected_features_tc),
    (combined_density, 'Density', selected_features_density),
    (combined_rg, 'Rg', selected_features_rg)
]:

    estimator.fit(item[X], item[name])
    importances = estimator.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]

    cumulative_importance = 0
    for idx in sorted_indices:
        cumulative_importance += importances[idx]
        if cumulative_importance <= 0.95:
            features.append(item[X].columns[idx])
        else:
            break

    print(f'number of selected features for {name} : {len(features)}')
    print(f'list of selected features for {name} : {features}')

# Using the Historical Gradient Boosting Regressor

model_tg = HistGradientBoostingRegressor(early_stopping=False,learning_rate=0.05,max_iter=500,loss = 'absolute_error', random_state=42, min_samples_leaf=3,max_leaf_nodes=100, verbose=0)
model_ffv = HistGradientBoostingRegressor(early_stopping=False,learning_rate=0.05,max_iter=500,loss = 'absolute_error', random_state=42, min_samples_leaf=3,max_leaf_nodes=100, verbose=0)
model_tc = HistGradientBoostingRegressor(early_stopping=False,learning_rate=0.05,max_iter=500,loss = 'absolute_error', random_state=42, min_samples_leaf=3,max_leaf_nodes=100, verbose=0)
model_density = HistGradientBoostingRegressor(early_stopping=False,learning_rate=0.05,max_iter=500,loss = 'absolute_error', random_state=42, min_samples_leaf=3,max_leaf_nodes=100, verbose=0)
model_rg = HistGradientBoostingRegressor(early_stopping=False,learning_rate=0.05,max_iter=500,loss = 'absolute_error', random_state=42, min_samples_leaf=3,max_leaf_nodes=100, verbose=0)

for item, name, features, model in [
    (combined_tg, 'Tg', selected_features_tg, model_tg),
    (combined_ffv, 'FFV', selected_features_ffv, model_ffv),
    (combined_tc, 'Tc', selected_features_tc, model_tc),
    (combined_density, 'Density', selected_features_density, model_density),
    (combined_rg, 'Rg', selected_features_rg, model_rg)
]:
    # Train the model using MAE loss
    model.fit(item[features], item[name])

    # print loss
    print(f'loss for {name} : {model.score(item[features], item[name])}')


# Predicting the test set for each target label df

submission = {
    'id': id,}

for item, name, features, test, model in [
    (combined_tg, 'Tg', selected_features_tg, scaled_test_tg, model_tg),
    (combined_ffv, 'FFV', selected_features_ffv, scaled_test_ffv, model_ffv),
    (combined_tc, 'Tc', selected_features_tc,scaled_test_tc, model_tc),
    (combined_density, 'Density', selected_features_density,scaled_test_density, model_density),
    (combined_rg, 'Rg', selected_features_rg, scaled_test_rg, model_rg)
]:
    predictions = model.predict(test[features])
    submission[name] = predictions
    print(f'predictions for {name} : {predictions}')

submission = pd.DataFrame(submission)
submission.to_csv('submission.csv', index=False)
    

