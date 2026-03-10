import pandas as pd
import numpy as np
import glob 
import os

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

# displaying basic information

print(input.info())
print(output.info())

# display first few rows - raw

print("first 5 rows of input\n")
input.head()

print("first 5 rows of output\n")
output.head()

# Preprocessing input and output data

# 1. Convert player_height to inches
# Example: '6-2' becomes 6*12 + 2 = 74 inches
input['player_height_inches'] = input['player_height'].apply(lambda x: int(x.split('-')[0]) * 12 + int(x.split('-')[1]))

# 2. Calculate player_age
# Extract year from game_id and birth year from player_birth_date
input['game_year'] = input['game_id'].astype(str).str[:4].astype(int)
input['birth_year'] = pd.to_datetime(input['player_birth_date']).dt.year
input['player_age'] = input['game_year'] - input['birth_year']

# 3. Standardize play direction
# If play_direction is 'left', we flip the x, y, dir, and o coordinates
input_left = input[input['play_direction'] == 'left'].copy()

# Flip x coordinate (120 yard field)
input_left['x'] = 120.0 - input_left['x']
input_left['ball_land_x'] = 120.0 - input_left['ball_land_x']

# Flip y coordinate (53.3 yard field width)
input_left['y'] = 53.3 - input_left['y']
input_left['ball_land_y'] = 53.3 - input_left['ball_land_y']

# Flip orientation and direction angles
input_left['o'] = (input_left['o'] + 180) % 360
input_left['dir'] = (input_left['dir'] + 180) % 360

# Get the right-direction plays
input_right = input[input['play_direction'] == 'right'].copy()

# Concatenate back together
processed_input = pd.concat([input_left, input_right], ignore_index=True)

# Drop intermediate and original columns that are no longer needed
processed_input = processed_input.drop(columns=[
'player_height', 'player_birth_date', 'game_year', 'birth_year', 'play_direction'
])

# We also need to standardize the output_df for later analysis/merging
# We only need play_direction from input_df to do this.
# Let's merge it onto output_df first.
play_info = input[['game_id', 'play_id', 'play_direction']].drop_duplicates()
output_merged = pd.merge(output, play_info, on=['game_id', 'play_id'])

# Now apply the coordinate standardization to output
output_left = output_merged[output_merged['play_direction'] == 'left'].copy()
output_left['x'] = 120.0 - output_left['x']
output_left['y'] = 53.3 - output_left['y']

output_right = output_merged[output_merged['play_direction'] == 'right'].copy()

processed_output = pd.concat([output_left, output_right], ignore_index=True)
processed_output = processed_output.drop(columns=['play_direction'])

# display first few rows - processed

print("first 5 rows of processed input\n")
processed_input.head()

print("first 5 rows of processed output\n")
processed_output.head()

# Combining input and output

# --- 1. Merge input and output data ---
# We need the last known position from the input data for each player on each play.
last_input_frame = processed_input.loc[processed_input.groupby(['game_id', 'play_id', 'nfl_id'])['frame_id'].idxmax()]

# Rename columns to avoid clashes after merging
last_input_frame = last_input_frame.rename(columns={
    'x': 'x_start', 'y': 'y_start',
    's': 's_start', 'a': 'a_start',
    'dir': 'dir_start', 'o': 'o_start',
    'frame_id': 'frame_id_start'
})

# Select necessary columns
last_input_frame_subset = last_input_frame[[
    'game_id', 'play_id', 'nfl_id', 'player_to_predict', 'player_role', 
    'x_start', 'y_start', 's_start', 'a_start', 'dir_start', 'o_start',
    'ball_land_x', 'ball_land_y'
]]

# Merge with the processed output data
# The output data contains the target x, y for each future frame
full_trajectory_df = pd.merge(
    processed_output, 
    last_input_frame_subset, 
    on=['game_id', 'play_id', 'nfl_id']
)

print("--- Merged Trajectory DataFrame Head ---")
print(full_trajectory_df.info())
print("\n" + "="*50 + "\n")

# modelling the machine learning



# EDA - Understanding the distributions of the data, univariate and bivariate

