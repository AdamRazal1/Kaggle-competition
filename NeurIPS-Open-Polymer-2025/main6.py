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

pd.set_option('display.max_rows', None)

# Problematic RDKit descriptors to remove (from Dmitry's analysis)
USELESS_COLS = [
    # NaN data
    'BCUT2D_MWHI', 'BCUT2D_MWLOW', 'BCUT2D_CHGHI', 'BCUT2D_CHGLO',
    'BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW', 'BCUT2D_MRHI', 'BCUT2D_MRLOW', 'MaxPartialCharge',
    # Constant data  
    'NumRadicalElectrons', 'SMR_VSA8', 'SlogP_VSA9', 'fr_barbitur',
    'fr_benzodiazepine', 'fr_dihydropyridine', 'fr_epoxide', 'fr_isothiocyan',
    'fr_lactam', 'fr_nitroso', 'fr_prisulfonamd', 'fr_thiocyan',
    # High correlation >0.95
    'MaxEStateIndex', 'HeavyAtomMolWt', 'ExactMolWt', 'NumValenceElectrons',
    'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Kappa1',
    'LabuteASA', 'HeavyAtomCount', 'MolMR', 'Chi3n', 'BertzCT', 'Chi2v',
    'Chi4n', 'HallKierAlpha', 'Chi3v', 'Chi4v', 'MinAbsPartialCharge',
    'MinPartialCharge', 'MaxAbsPartialCharge', 'FpDensityMorgan2',
    'FpDensityMorgan3', 'Phi', 'Kappa3', 'fr_nitrile', 'SlogP_VSA6',
    'NumAromaticCarbocycles', 'NumAromaticRings', 'fr_benzene', 'VSA_EState6',
    'NOCount', 'fr_C_O', 'fr_C_O_noCOO', 'NumHDonors', 'fr_amide',
    'fr_Nhpyrrole', 'fr_phenol', 'fr_phenol_noOrthoHbond', 'fr_COO2',
    'fr_halogen', 'fr_diazo', 'fr_nitro_arom', 'fr_phos_ester'
]

pd.set_option('display.max_columns', None)

train = pd.read_csv('datasets/train.csv')
test = pd.read_csv('datasets/test.csv')
dataset1 = pd.read_csv('datasets/dataset1.csv')
dataset3 = pd.read_csv('datasets/dataset3.csv')
dataset1.rename(columns={'TC_mean': 'Tc'}, inplace=True)
dataset4 = pd.read_csv('datasets/dataset4.csv')
extra_tg = pd.read_csv('datasets/TgSS_enriched_cleaned.csv')
id = test['id'].copy()

# Canonicalize SMILES
def canon(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol else None

for item in (train, test, dataset1, dataset3, dataset4, extra_tg):
    item['canonical'] = item['SMILES'].apply(canon)

new_tg = {'canonical': [], 'Tg': []}
new_tc = {'canonical': [], 'Tc': []}
new_ffv = {'canonical': [], 'FFV': []}

# adding new ffv

new_tc_dict = dataset1.set_index(dataset1['canonical'])['Tc'].to_dict()
train['Tc'] = train['canonical'].map(new_tc_dict)

extra_tg.drop_duplicates(inplace=True)
extra_tg.reset_index(drop=True, inplace=True)

new_extra_tg = extra_tg[['canonical', 'Tg']].copy()

for smile, new, key_col in [(dataset4, new_ffv, 'FFV')]:
    for smile_row in smile['canonical']:
        if smile_row not in train['canonical'].values:
            new['canonical'].append(smile[smile['canonical'] == smile_row]['canonical'].values[0])
            new[key_col].append(smile[smile['canonical'] == smile_row][key_col].values[0])

train = pd.concat([train, new_extra_tg], axis=0, ignore_index=True)
train = pd.concat([train, pd.DataFrame(new_ffv)], axis=0, ignore_index=True)

train = train.dropna(subset=['canonical'], ignore_index=True, axis=0)

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

def compute_descriptors(mol_series):
    desc_result = {desc: [] for desc, _ in Descriptors.descList}
    for mol in mol_series:
        for desc, func in Descriptors.descList:
            try:
                desc_result[desc].append(func(mol))
            except:
                desc_result[desc].append(None)
    return desc_result

desc_list_train = compute_descriptors(train['Mol'])
desc_list_test = compute_descriptors(test['Mol'])

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

# dropping canonical, mol and id

for i in (tg, ffv, tc, density, rg, test):
    i.drop(['canonical',], axis=1, inplace=True)
    i.reset_index(drop=True, inplace=True)

# eliminating useless columns from descriptors

for i in (tg, ffv, tc, density, rg, test):
    i.drop(USELESS_COLS, axis=1, inplace=True)
    i.reset_index(drop=True, inplace=True)

print(tg.shape, ffv.shape, tc.shape, density.shape, rg.shape)

# feature selection - selecting top 95% features using ExtraTreeRegressor

scaler_tg = StandardScaler()
scaler_ffv = StandardScaler()
scaler_tc = StandardScaler()
scaler_density = StandardScaler()
scaler_rg = StandardScaler()

model_tg = CatBoostRegressor(task_type='CPU',random_seed=42, verbose=0, loss_function='MAE', iterations=6000,)
model_ffv = CatBoostRegressor(task_type='CPU',random_seed=42, verbose=0, loss_function='MAE', iterations=4000, )
model_tc = CatBoostRegressor(task_type='CPU',random_seed=42, verbose=0, loss_function='MAE', iterations=1500)
model_density = CatBoostRegressor(task_type='CPU',random_seed=42, verbose=0, loss_function='MAE', iterations=1200 , )
model_rg = CatBoostRegressor(task_type='CPU',random_seed=42, verbose=0, loss_function='MAE', )

submission = {
    'id': id
}

tg_only = ['FFV', 'Tc', 'Density', 'Rg']
ffv_only = ['Tg', 'Tc', 'Density', 'Rg']
tc_only = ['Tg', 'FFV', 'Density', 'Rg']
density_only = ['Tg', 'FFV', 'Tc', 'Rg']
rg_only = ['Tg', 'FFV', 'Tc', 'Density']

def training_and_evaluation(item, target, scaler, model, only):

    X = item.drop([target], axis=1).copy()
    y = item[target].copy()    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

    true_test = test.drop(['id', 'Mol', 'SMILES'], axis=1)
    true_test = true_test[[col for col in true_test.columns if col not in only]]

    true_test_scaled = scaler.transform(true_test)
    true_test_scaled = pd.DataFrame(true_test_scaled, columns=true_test.columns)

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    error = mean_absolute_error(y_pred, y_test)
    print(f"MAE for {target} is {error}")

    true_y_pred = model.predict(true_test_scaled)
    submission[target] = true_y_pred
    print(submission[target])   


training_and_evaluation(tg, 'Tg',  scaler_tg, model_tg, tg_only)
training_and_evaluation(ffv, 'FFV',  scaler_ffv, model_ffv, ffv_only)
training_and_evaluation(tc, 'Tc',  scaler_tc, model_tc, tc_only)
training_and_evaluation(density, 'Density',  scaler_density, model_density, density_only)
training_and_evaluation(rg, 'Rg',  scaler_rg, model_rg, rg_only)
