import pandas as pd
import torch

from torch import optim
from torch import nn
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import StandardScaler, LabelEncoder

pd.set_option('display.max_rows', None)

train = pd.read_csv('datasets/train.csv')
test = pd.read_csv('datasets/test.csv')
id = train['id'].copy()

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

# Creating the neural network model

class DeepLearningModel(nn.Module):
    def __init__(self, input_features, hidden_features, dropout = 0.3):
        super(DeepLearningModel, self).__init__()

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

# Initializing the model for each target

model_tg = DeepLearningModel(248, 128)
model_ffv = DeepLearningModel(248, 128)
model_tc = DeepLearningModel(248, 128)
model_density = DeepLearningModel(248, 128)
model_rg = DeepLearningModel(248, 128)


# Initializing the loss and optimizer for each target

loss_tg = nn.MSELoss()
loss_ffv = nn.MSELoss()
loss_tc = nn.MSELoss()
loss_density = nn.MSELoss()
loss_rg = nn.MSELoss()

optimizer_tg = optim.Adam(model_tg.parameters(), lr = 0.001)
optimizer_ffv = optim.Adam(model_ffv.parameters(), lr = 0.001)
optimizer_tc = optim.Adam(model_tc.parameters(), lr = 0.001)
optimizer_density = optim.Adam(model_density.parameters(), lr = 0.001)
optimizer_rg = optim.Adam(model_rg.parameters(), lr = 0.001)


# Preparing for training for each target

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_tg = model_tg.to(device)
model_ffv = model_ffv.to(device)
model_tc = model_tc.to(device)
model_density = model_density.to(device)
model_rg = model_rg.to(device)

tg_x = torch.tensor(scaled_tg[[col for col in scaled_tg.columns if col != 'Tg']].values, dtype=torch.float32).to(device)
ffv_x = torch.tensor(scaled_ffv[[col for col in ffv.columns if col != 'FFV']].values, dtype=torch.float32).to(device)
tc_x = torch.tensor(scaled_tc[[col for col in tc.columns if col != 'Tc']].values, dtype=torch.float32).to(device) 
density_x = torch.tensor(scaled_density[[col for col in density.columns if col != 'Density']].values, dtype=torch.float32).to(device) 
rg_x = torch.tensor(scaled_rg[[col for col in rg.columns if col != 'Rg']].values, dtype=torch.float32).to(device)

tg_y = torch.tensor(scaled_tg['Tg'].values, dtype=torch.float32).to(device)
ffv_y = torch.tensor(scaled_ffv['FFV'].values, dtype=torch.float32).to(device)
tc_y = torch.tensor(scaled_tc['Tc'].values, dtype=torch.float32).to(device)
density_y = torch.tensor(scaled_density['Density'].values, dtype=torch.float32).to(device)
rg_y = torch.tensor(scaled_rg['Rg'].values, dtype=torch.float32).to(device)

# training phase for each target

num_epochs = 500

for epoch in range(num_epochs):
    model_tg.train()
    model_ffv.train()
    model_tc.train()
    model_density.train()
    model_rg.train()

    # Forward pass
    outputs_tg = model_tg(tg_x)
    outputs_ffv = model_ffv(ffv_x)
    outputs_tc = model_tc(tc_x)
    outputs_density = model_density(density_x)
    outputs_rg = model_rg(rg_x)

    loss_tg_fn = loss_tg(outputs_tg, tg_y.unsqueeze(dim=1))
    loss_ffv_fn = loss_ffv(outputs_ffv, ffv_y.unsqueeze(dim=1))
    loss_tc_fn = loss_tc(outputs_tc, tc_y.unsqueeze(dim=1))
    loss_density_fn = loss_density(outputs_density, density_y.unsqueeze(dim=1))
    loss_rg_fn = loss_rg(outputs_rg, rg_y.unsqueeze(dim=1))

    # Backward pass and optimization

    optimizer_tg.zero_grad()
    optimizer_ffv.zero_grad()
    optimizer_tc.zero_grad()
    optimizer_density.zero_grad()
    optimizer_rg.zero_grad()

    loss_tg_fn.backward()
    loss_ffv_fn.backward()
    loss_tc_fn.backward()
    loss_density_fn.backward()
    loss_rg_fn.backward()

    optimizer_tg.step()
    optimizer_ffv.step()
    optimizer_tc.step()
    optimizer_density.step()
    optimizer_rg.step()


    if epoch % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss_tg_fn:.4f}, Loss: {loss_ffv_fn:.4f}, Loss: {loss_tc_fn:.4f}, Loss: {loss_density_fn:.4f}, Loss: {loss_rg_fn:.4f}')


# Combining all 5 of the models

class CombinedModel:
    def __init__(self, models):
        self.models = models  # Dict of models

    def predict(self, x):
        self.models['tg'].eval()
        self.models['ffv'].eval()
        self.models['tc'].eval()
        self.models['density'].eval()
        self.models['rg'].eval()

        with torch.no_grad():
            out_tg = self.models['tg'](x)
            out_ffv = self.models['ffv'](x)
            out_tc = self.models['tc'](x)
            out_density = self.models['density'](x)
            out_rg = self.models['rg'](x)

        # Combine into one tensor
        return torch.cat([out_tg, out_ffv, out_tc, out_density, out_rg], dim=1)

# Example usage
combined_model = CombinedModel({
    'tg': model_tg,
    'ffv': model_ffv,
    'tc': model_tc,
    'density': model_density,
    'rg': model_rg
})

# Preprocessing for testing
 
X_test = torch.tensor(test[[col for col in test.columns if col != ['Tg', 'FFV', 'Tc', 'Density', 'Rg']]].values, dtype = torch.float32).to(device)

with torch.no_grad():
    predictions = combined_model.predict(X_test).cpu().numpy()

predictions = scaler_test.inverse_transform(predictions)
predictions_df = pd.DataFrame(predictions, columns=['Tg', 'FFV', 'Tc', 'Density', 'Rg'])
predictions_df = predictions_df.dropna(axis=0, how='any')