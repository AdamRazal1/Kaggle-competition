import pandas as pd
import numpy as np
import glob 
import os
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder

print(os.getcwd())
pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 50)

# load and concat weekly input data

input_dir = glob.glob(os.path.join('dataset/train', "input_2023_w*.csv"))
input = pd.concat([pd.read_csv(file) for file in input_dir], ignore_index=True)
print(f"successfully load {len(input_dir)} input file")

# load and concate weekly output data

output_dir = glob.glob(os.path.join('dataset/train', "output_2023_w*.csv"))
output = pd.concat([pd.read_csv(file) for file in output_dir], ignore_index=True)
print(f"successfully load {len(output_dir)} output file")

# displaying information of the dataframe

print(input.info())
print(input.head())
print(output.info())
print(output.head())

# checking and eliminating duplicates and missing values for input and output

input.drop_duplicates(inplace=True)
input.dropna(inplace=True)
input.reset_index(drop=True, inplace=True)

output.drop_duplicates(inplace=True)
output.dropna(inplace=True)
output.reset_index(drop=True, inplace=True)

# fixing the data type of certain columns

input['play_direction'] = input['play_direction'].astype('string')
input['player_name'] = input['player_name'].astype('string')
input['player_height'] = input['player_height'].astype('string')
input['player_birth_date'] =  pd.to_datetime(input['player_birth_date'])
input['player_position'] = input['player_position'].astype('string')
input['player_side'] = input['player_side'].astype('string')
input['player_role'] = input['player_role'].astype('string')

# recalculating the height of the player in inches

input['player_height_inches'] = input['player_height'].apply(lambda x: int(x.split('-')[0]) * 12 + int(x.split('-')[1]))

# Extracting the age of the player based on the birth date

input['player_age'] = input['game_id'].astype('string').str[:4].astype(int) - input['player_birth_date'].dt.year

# combining input and output based on the game id, nfl id and play id based on the last input frame

combined = pd.merge(output, input, on=['game_id', 'play_id', 'nfl_id'])

# extracting only the last frame of the input data from the combined data

combined_last_frame = combined.loc[combined.groupby(['game_id', 'play_id', 'nfl_id'])['frame_id_y'].idxmax()].reset_index(drop=True)

# renaming the columns for readability and clarity

combined_last_frame = combined_last_frame.rename(columns={
    'frame_id_x' : 'output_frame_id', 'frame_id_y' : 'input_frame_id',
    'x_x' : 'x', 'y_x' : 'y',
    'x_y' : 'x_input', 'y_y' : 'y_input',
    's' : 'speed', 'a' : 'acceleration',
})

# eliminating useless columns

useless_cols = ['game_id', 'play_id', 'nfl_id', 'output_frame_id', 'player_name', 'player_height', 'player_birth_date',]

combined_data = combined_last_frame.drop(columns=useless_cols, axis=1).copy()

# Defining continuous and categorical features

num_cols = combined_data.select_dtypes("number").columns.to_list()
cat_cols = combined_data.select_dtypes("string").columns.to_list()

# Preparing input and output data for the machine learning model

input = combined_data.drop(columns = ['x', 'y'], axis=1).copy()
output = combined_data[['x', 'y']]

# modelling the regression model for machine learning

model = CatBoostRegressor(cat_features=cat_cols, verbose=0)

# cross validation via kfold

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# training seperately for x and y

accuracies = []
mae_scores = []

for train_index, test_index in kf.split(input):
    x_train, x_test = input.iloc[train_index], input.iloc[test_index]
    y_train_x = output.iloc[train_index]['x']
    y_train_y = output.iloc[train_index]['y']
    y_test_x = output.iloc[test_index]['x']
    y_test_y = output.iloc[test_index]['y']

    model_x = CatBoostRegressor(cat_features=cat_cols,verbose=0)
    model_y = CatBoostRegressor(cat_features=cat_cols,verbose=0)

    model_x.fit(x_train, y_train_x)
    model_y.fit(x_train, y_train_y)

    y_pred_x = model_x.predict(x_test)
    y_pred_y = model_y.predict(x_test)

    mae_x = mean_absolute_error(y_test_x, y_pred_x)
    mae_y = mean_absolute_error(y_test_y, y_pred_y)

    mae_scores.append((mae_x, mae_y))
