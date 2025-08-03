import sklearn
import pandas as pd
import numpy as np
import rdkit

from rdkit import Chem
from rdkit.Chem import Descriptors, rdmolops
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
from catboost import CatBoostRegressor

pd.set_option('display.max_columns', None)

train = pd.read_csv('datasets/train.csv')
test = pd.read_csv('datasets/test.csv')
dataset1 = pd.read_csv('datasets/dataset1.csv')
dataset3 = pd.read_csv('datasets/dataset3.csv')
dataset1.rename(columns={'TC_mean': 'Tc'}, inplace=True)
dataset4 = pd.read_csv('datasets/dataset4.csv')
id = test['id'].copy()

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

for smile, new, key_col in [(dataset4, new_ffv, 'FFV')]:
    for smile_row in smile['canonical']:
        if smile_row not in train['canonical'].values:
            new['canonical'].append(smile[smile['canonical'] == smile_row]['canonical'].values[0])
            new[key_col].append(smile[smile['canonical'] == smile_row][key_col].values[0])

# train = pd.concat([train, pd.DataFrame(new_tg)], axis=0)
# train = pd.concat([train, pd.DataFrame(new_tc)], axis=0)
train = pd.concat([train, pd.DataFrame(new_ffv)], axis=0)

train = train.dropna(subset=['canonical'])
train.reset_index(drop=True, inplace=True)

# Converting the smiles into mol

train['Mol'] = train['canonical'].apply(Chem.MolFromSmiles)
test['Mol'] = test['canonical'].apply(Chem.MolFromSmiles)

# Feature engineering - Getting morgan fingerprints for each mol

morgan_gen = GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)

train_fps = np.array([morgan_gen.GetFingerprintAsNumPy(mol) for mol in train['Mol']])
test_fps = np.array([morgan_gen.GetFingerprintAsNumPy(mol) for mol in test['Mol']])

train_fp_df = pd.DataFrame(train_fps, columns=[f'fp_{i}' for i in range(train_fps.shape[1])])
test_fp_df = pd.DataFrame(test_fps, columns=[f'fp_{i}' for i in range(test_fps.shape[1])])

train = pd.concat([train.reset_index(drop=True), train_fp_df], axis=1)
test = pd.concat([test.reset_index(drop=True), test_fp_df], axis=1)

print(train.shape)

# feature engineering - Extracting Molecular Informations from descriptors and descriptors 3d

desc_list_train = {desc : [] for desc, _ in Descriptors.descList}
desc_list_test = {desc : [] for desc, _ in Descriptors.descList}

for mol in train['Mol']:
    for desc, func in Descriptors.descList:
        try:
            desc_list_train[desc].append(func(mol))
        except:
            desc_list_train[desc].append(None)

for mol in test['Mol']:
    for desc, func in Descriptors.descList:
        try:
            desc_list_test[desc].append(func(mol))
        except:
            desc_list_test[desc].append(None)

# appending added features to the train and test dataset

train = pd.concat([train, pd.DataFrame(desc_list_train)], axis=1)
test = pd.concat([test, pd.DataFrame(desc_list_test)], axis=1)

print(train.shape)

# We'll separate train to be one model for each target variable.

morgan = [col for col in train.columns if 'fp_' in col]
descriptor = [col for col in train.columns if col  in [name for name, _ in Descriptors.descList]]

tg=train[['canonical','Tg'] + morgan + descriptor].copy().dropna(subset = ['Tg'])
ffv=train[['canonical','FFV'] + morgan + descriptor].copy().dropna(subset = ['FFV'])
tc=train[['canonical','Tc'] + morgan + descriptor].copy().dropna(subset = ['Tc'])
density=train[['canonical','Density'] + morgan + descriptor].copy().dropna(subset = ['Density'])
rg=train[['canonical','Rg'] + morgan + descriptor].copy().dropna(subset = ['Rg'])

print(tg.shape, ffv.shape, tc.shape, density.shape, rg.shape)

# for df in (tg, ffv, tc, density, rg):
#     df.replace([np.inf, -np.inf], np.nan, inplace=True)

# dropping canonical, mol and id

for i in (tg, ffv, tc, density, rg):
    i.drop(['canonical',], axis=1, inplace=True)
    i.reset_index(drop=True, inplace=True)

# eliminating columns containing > 30% missing values from descriptors

thres = 0.3 * (len(train.values))
eliminated_columns = [col for col in desc_list_train.keys() if train[col].isnull().sum() >= thres]
print(f'number of columns to be eliminated : {len(eliminated_columns)}')

tg.drop(eliminated_columns, axis=1, inplace=True)
ffv.drop(eliminated_columns, axis=1, inplace=True)
tc.drop(eliminated_columns, axis=1, inplace=True)
density.drop(eliminated_columns, axis=1, inplace=True)
rg.drop(eliminated_columns, axis=1, inplace=True)
test.drop(eliminated_columns, axis=1, inplace=True)

print(tg.shape, ffv.shape, tc.shape, density.shape, rg.shape)

# feature selection - selecting top 95% features using ExtraTreeRegressor

selected_features_tg = []
selected_features_ffv = []
selected_features_tc = []
selected_features_density = []
selected_features_rg = []

scaler_tg = StandardScaler()
scaler_ffv = StandardScaler()
scaler_tc = StandardScaler()
scaler_density = StandardScaler()
scaler_rg = StandardScaler()

model_tg = CatBoostRegressor(random_seed=42, verbose=0, loss_function='MAE')
model_ffv = CatBoostRegressor(random_seed=42, verbose=0, loss_function='MAE')
model_tc = CatBoostRegressor(random_seed=42, verbose=0, loss_function='MAE')
model_density = CatBoostRegressor(random_seed=42, verbose=0, loss_function='MAE')
model_rg = CatBoostRegressor(random_seed=42, verbose=0, loss_function='MAE')

submission = {
    'id': id
}

tg_only = ['FFV', 'Tc', 'Density', 'Rg']
ffv_only = ['Tg', 'Tc', 'Density', 'Rg']
tc_only = ['Tg', 'FFV', 'Density', 'Rg']
density_only = ['Tg', 'FFV', 'Tc', 'Rg']
rg_only = ['Tg', 'FFV', 'Tc', 'Density']

for item, target, selected_features, scaler, model, only in ((tg, 'Tg', selected_features_tg, scaler_tg, model_tg, tg_only), (ffv, 'FFV', selected_features_ffv, scaler_ffv, model_ffv, ffv_only), (tc, 'Tc', selected_features_tc, scaler_tc, model_tc, tc_only), (density, 'Density', selected_features_density, scaler_density, model_density, density_only), (rg, 'Rg', selected_features_rg, scaler_rg, model_rg, rg_only)):
    
    X = item.drop([target], axis=1).copy()
    y = item[target].copy()    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

    estimator = ExtraTreesRegressor(random_state=42)

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

    true_test = test.drop(['id', 'canonical', 'Mol', 'SMILES'], axis=1)
    true_test = true_test[[col for col in true_test.columns if col not in only]]

    true_test_scaled = scaler.transform(true_test)
    true_test_scaled = pd.DataFrame(true_test_scaled, columns=true_test.columns)

    estimator.fit(X_train_scaled, y_train)
    
    importances = estimator.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]

    cumulative_importance = 0
    for idx in sorted_indices:
        cumulative_importance += importances[idx]
        if cumulative_importance <= 0.95:
            selected_features.append(X.columns[idx])
        else:
            break
    print(f'Features selected for {target}, is {len(selected_features)}')

    model.fit(X_train_scaled[selected_features], y_train)
    y_pred = model.predict(X_test_scaled[selected_features])
    error = mean_absolute_error(y_pred, y_test)
    print(f"MAE for {target} is {error}")

    true_y_pred = model.predict(true_test_scaled[selected_features])
    submission[target] = true_y_pred
    print(submission[target])    
    # y_pred =  model.predict(true_test_scaled)
    # submission[target] = y_pred

pd.DataFrame(submission).to_csv('submission.csv', index=False)
