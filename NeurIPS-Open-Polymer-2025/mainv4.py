import sklearn
import pandas as pd
import numpy as np
import rdkit

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


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
dataset1['canonical'] = dataset1['SMILES'].apply(canon)
dataset3['canonical'] = dataset3['SMILES'].apply(canon)
dataset4['canonical'] = dataset4['SMILES'].apply(canon)

new_tg = {'SMILES': [], 'Tg': []}
new_tc = {'SMILES': [], 'Tc': []}
new_ffv = {'SMILES': [], 'FFV': []}

# Drop invalid SMILES
train = train.dropna(subset=['canonical'])
dataset1 = dataset1.dropna(subset=['canonical'])
dataset3 = dataset3.dropna(subset=['canonical'])

# for tg

for smile in dataset3['canonical']:
    if smile not in train['canonical'].values:
        new_tg['SMILES'].append(smile)
        new_tg['Tg'].append(dataset3[dataset3['canonical'] == smile]['Tg'].values)

# for tc

for smile in dataset1['canonical']:
    if smile not in train['canonical'].values:
        new_tc['SMILES'].append(smile)
        new_tc['Tc'].append(dataset1[dataset1['canonical'] == smile]['TC_mean'].values)
# for ffv

for smile in dataset4['canonical']:
    if smile not in train['canonical'].values:
        new_ffv['SMILES'].append(smile)
        new_ffv['FFV'].append(dataset4[dataset4['canonical'] == smile]['FFV'].values)

for key, item in new_tg.items():
    print(f'{key} : {item}')

for key, item in new_tc.items():
    print(f'{key} : {item}')

for key, item in new_ffv.items():
    print(f'{key} : {item}')

# concatenating the train set with the new tg, ffv, and tc

train = pd.concat([train, pd.DataFrame(new_tg)], ignore_index=True)
train = pd.concat([train, pd.DataFrame(new_tc)], ignore_index=True)
train = pd.concat([train, pd.DataFrame(new_ffv)], ignore_index=True)

print(train.shape())

train.drop_duplicates(subset=['SMILES'], inplace=True)
train.reset_index(drop=True, inplace=True)

print(train.shape)

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

# Converting the smiles into mol

train['Mol'] = train['canonical'].apply(Chem.MolFromSmiles)
test['Mol'] = test['canonical'].apply(Chem.MolFromSmiles)

# Drop rows if the rdkit cannot generate the mol for the smiles

train.dropna(subset=['Mol'], inplace=True)
test.dropna(subset=['Mol'], inplace=True)

train.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)

# feature engineering - Extracting Molecular Informations from descriptors and descriptors 3d

desc_list = {desc : [] for desc, _ in Descriptors.descList}

for mol in train['Mol']:
    for desc, func in Descriptors.descList:
        desc_list[desc].append(func(mol))


# appending added features to the train and test dataset

train = pd.concat([train, pd.DataFrame(desc_list)], axis=1)
test = pd.concat([test, pd.DataFrame(desc_list)], axis=1)

# eliminating columns containing > 50% missing values from descriptors

thres = 0.5 * (len(train.values))
eliminated_columns = [col for col in desc_list.keys() if train[col].isnull().sum() >= thres]
print(f'number of columns to be eliminated : {len(eliminated_columns)}')

train.drop(eliminated_columns, axis=1, inplace=True)
test.drop(eliminated_columns, axis=1, inplace=True)

# Dropping id, smiles and mol columns

train.drop(['id','SMILES', 'Mol', 'canonical'], axis=1, inplace=True)
test.drop(['SMILES', 'Mol'], axis=1, inplace=True)

# seperating the data for each target label df

tg = train[[col for col in train.columns if col not in ['Tg', 'FFV', 'Tc', 'Density', 'Rg']] + ['Tg']].reset_index(drop=True)
ffv = train[[col for col in train.columns if col not in ['Tg', 'FFV', 'Tc', 'Density', 'Rg']] + ['FFV']].reset_index(drop=True)
tc = train[[col for col in train.columns if col not in ['Tg', 'FFV', 'Tc', 'Density', 'Rg']] + ['Tc']].reset_index(drop=True)
density = train[[col for col in train.columns if col not in ['Tg', 'FFV', 'Tc', 'Density', 'Rg']] + ['Density']].reset_index(drop=True)
rg = train[[col for col in train.columns if col not in ['Tg', 'FFV', 'Tc', 'Density', 'Rg']] + ['Rg']].reset_index(drop=True)



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

for item in [tg, ffv, tc, density, rg]:
    scaler = StandardScaler()
    item.iloc[:, :-1] = scaler.fit_transform(item.iloc[:, :-1])

# Feature selection - choosing top 95% of the best feature

X = [col for col in train.columns if col not in ['Tg', 'FFV', 'Tc', 'Density', 'Rg']]

estimator = ExtraTreesRegressor()

selected_features_tg = []
selected_features_ffv = []
selected_features_tc = []
selected_features_density = []
selected_features_rg = []

for item, name, features in [
    (tg, 'Tg', selected_features_tg),
    (ffv, 'FFV', selected_features_ffv),
    (tc, 'Tc', selected_features_tc),
    (density, 'Density', selected_features_density),
    (rg, 'Rg', selected_features_rg)
]:

    estimator.fit(item[X], item[name])
    importances = estimator.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]

    cumulative_importance = 0
    for idx in sorted_indices:
        cumulative_importance += importances[idx]
        if cumulative_importance <= 0.9:
            features.append(item[X].columns[idx])
        else:
            break

    print(f'number of selected features for {name} : {len(features)}')
    print(f'list of selected features for {name} : {features}')

# Using the Historical Gradient Boosting Regressor

model_tg = HistGradientBoostingRegressor(early_stopping=True,learning_rate=0.2,max_iter=1000,loss = 'absolute_error', random_state=42, min_samples_leaf=5,max_leaf_nodes=1000, verbose=1)
model_ffv = HistGradientBoostingRegressor(early_stopping=True,learning_rate=0.2,max_iter=1000,loss = 'absolute_error', random_state=42, min_samples_leaf=5,max_leaf_nodes=1000, verbose=1)
model_tc = HistGradientBoostingRegressor(early_stopping=True,learning_rate=0.2,max_iter=1000,loss = 'absolute_error', random_state=42, min_samples_leaf=5,max_leaf_nodes=1000, verbose=1)
model_density = HistGradientBoostingRegressor(early_stopping=True,learning_rate=0.2,max_iter=1000,loss = 'absolute_error', random_state=42, min_samples_leaf=5,max_leaf_nodes=1000, verbose=1)
model_rg = HistGradientBoostingRegressor(early_stopping=True,learning_rate=0.2,max_iter=1000,loss = 'absolute_error', random_state=42, min_samples_leaf=5,max_leaf_nodes=1000, verbose=1)

for item, name, features, model in [
    (tg, 'Tg', selected_features_tg, model_tg),
    (ffv, 'FFV', selected_features_ffv, model_ffv),
    (tc, 'Tc', selected_features_tc, model_tc),
    (density, 'Density', selected_features_density, model_density),
    (rg, 'Rg', selected_features_rg, model_rg)
]:
    model.fit(item[features], item[name])
    print(f'performance for {name} : {model.score(item[features], item[name])}')

# Predicting the test set for each target label df

test.dropna(subset=['id'], inplace=True)

for item, name, features, model in [
    (tg, 'Tg', selected_features_tg, model_tg),
    (ffv, 'FFV', selected_features_ffv, model_ffv),
    (tc, 'Tc', selected_features_tc, model_tc),
    (density, 'Density', selected_features_density, model_density),
    (rg, 'Rg', selected_features_rg, model_rg)
]:
    predictions = model.predict(test[features])
    print(f'predictions for {name} : {predictions}')
