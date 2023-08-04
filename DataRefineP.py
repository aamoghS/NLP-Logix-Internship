import pandas as pd

df = pd.read_csv('NFL Play by Play 2009-2018 (v5).csv')

nfl_filter = df[['posteam', 'play_type', 'quarter_seconds_remaining', 'pass_length', 'pass_location', 'run_location', 'ydstogo', 'first_down_rush',
                 'yrdln', 'game_id', 'down', 'score_differential_post', 'incomplete_pass', 'complete_pass', 'rush_attempt', 'pass_attempt',
                 'defteam', 'qtr', 'game_seconds_remaining', 'posteam_timeouts_remaining', 'defteam_timeouts_remaining',
                 'epa', 'wpa', 'yardline_100', 'half_seconds_remaining', 'goal_to_go', 'side_of_field',
                 'shotgun', 'drive', 'no_huddle', 'penalty']]

df = nfl_filter.copy()

df = df[(df['play_type'].isin(['pass', 'run']))]
df = df[(df['ydstogo'] < 36)]
df = df[~((df['incomplete_pass'] == 1) & (df['complete_pass'] == 1) & (df['play_type'] == 'pass'))]
df = df[~df[['pass_location', 'run_location']].notnull().all(1)]

mask1 = (df['down'] == 1) & (df['ydstogo'] <= 12)
mask2 = (df['down'] == 2) & (df['ydstogo'] <= 5)
mask3 = (df['down'] == 3) & (df['ydstogo'] <= 2)
mask4 = (df['down'] == 4) & (df['ydstogo'] <= 1)
df['down_ydstogo'] = ((mask1) | (mask2) | (mask3) | (mask4)).astype(int)

df['time_score'] = df['game_seconds_remaining']
df.loc[df['game_seconds_remaining'] < 900, 'time_score'] = df['score_differential_post'] * df['score_differential_post'] * df['game_seconds_remaining']

df['pass_length'] = df['pass_length'].fillna('n/a')
df['pass_location'] = df['pass_location'].fillna('n/a')
df['run_location'] = df['run_location'].fillna('n/a')
df['defteam'] = df['defteam'].fillna('n/a')
df['side_of_field'] = df['side_of_field'].fillna('n/a')
df['yrdln'] = df['yrdln'].fillna('n/a')

df = df.dropna()

categorical_columns = ['posteam', 'play_type', 'pass_length', 'pass_location', 'run_location', 'defteam', 'side_of_field', 'yrdln']
for column in categorical_columns:
    df[f'{column}_f'] = pd.factorize(df[column])[0]

rush_pass_ratio = df.groupby('posteam').sum()['play_type_f'] / df.groupby('posteam').count()['play_type_f']
rp_ratio_dict = rush_pass_ratio.to_dict()
df['posteam_rp_ratio'] = df['posteam'].map(rp_ratio_dict)

df.to_csv('NFL_Python_Filtered.csv', index=False)
