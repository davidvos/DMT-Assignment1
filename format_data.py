import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn import preprocessing

def read_dataset(dataset):
    dataset['time']= pd.to_datetime(dataset['time'])
    dataset['date'] = dataset['time'].dt.date
    dataset = dataset.drop(columns=['Unnamed: 0', 'time'])
    return dataset

def select_aggregate_type(row):
    mean_columns = ['mood', 'circumplex.arousal', 'circumplex.valence', 'activity']
    sum_columns = ['screen', 'call', 'sms', 'appCat.builtin', 'appCat.communication', 'appCat.entertainment', 
                   'appCat.finance', 'appCat.game', 'appCat.office', 'appCat.other', 'appCat.social', 
                   'appCat.travel', 'appCat.unknown', 'appCat.utilities', 'appCat.weather']
    if row['variable'] in mean_columns:
        val = row['mean_value']
    elif row['variable'] in sum_columns:
        val = row['sum_value']
    else:
        val = np.NaN
    return val

def aggregate_dataset(dataset):
    aggregated_dataset = dataset.groupby(['id', 'date', 'variable']).agg(
        mean_value = pd.NamedAgg(column='value', aggfunc='mean'), 
        sum_value = pd.NamedAgg(column='value', aggfunc='sum')
    ).reset_index()
    aggregated_dataset['final_value'] = aggregated_dataset.apply(select_aggregate_type, axis=1)
    aggregated_dataset = aggregated_dataset.drop(columns=['mean_value', 'sum_value'])
    pivot_table = aggregated_dataset.pivot_table(index=['id', 'date'], 
                                             columns=['variable'], values='final_value').reset_index()
    pivot_table['mood'] = pivot_table['mood']
    return pivot_table

def clean_dataset(dataset):

    # Remove rows without mood
    dataset = dataset[dataset['mood'].notnull()]

    # Fill NaN cells with average column value 
    dataset = dataset.fillna(dataset.mean())
    
    # Remove columns 
    drop_columns = ['appCat.builtin', 'appCat.communication', 'appCat.entertainment', 
                    'appCat.finance', 'appCat.game', 'appCat.office', 'appCat.other', 'appCat.social', 
                    'appCat.travel', 'appCat.unknown', 'appCat.utilities', 'appCat.weather']
    dataset = dataset.drop(columns=drop_columns)

    stored_ids = dataset['id']
    stored_dates = dataset['date']
    stored_moods = dataset['mood']

    dataset = dataset.drop(['id', 'date', 'mood'], axis=1)
    stored_columns = dataset.columns

    # dataset = dataset.values #returns a numpy array
    ss = preprocessing.StandardScaler()
    dataset = ss.fit_transform(dataset)
    # dataset = pd.DataFrame(x_scaled, columns=stored_columns)

    dataset.insert(0, 'id', list(stored_ids), allow_duplicates=True)
    dataset.insert(1, 'date', list(stored_dates), allow_duplicates=True)
    dataset.insert(2, 'mood', list(stored_moods), allow_duplicates=True)

    return dataset

def sliding_window(dataset):
    # Sliding window over all days per user. From each day look 5 days in the future and store the features
    # of available dates. Check if the 6th day has a mood label. If so, store the entire sequence with the
    # corresponding label. Each sequence also needs at least a certain amount of days available or it will
    # be discarded.
    interval_size = 5
    empty_threshold = 3

    users = list(dataset['id'].unique())

    samples = []

    for user in users:
        user_df = dataset[dataset['id'] == user]
        for index, row in user_df.iterrows():
            sample = []
            first_date = row['date']
            n_empty = 0

            current_date = None
            final_day_date = 0

            for interval_date in range(interval_size):

                final_day_date += 1
                
                current_date = first_date + timedelta(days=interval_date)
                current_date_features = user_df[user_df['date'] == current_date]
                if current_date_features.empty:
                    n_empty += 1
                else:
                    sample.append(np.squeeze(current_date_features.to_numpy())[3:])
                    
            label_date = first_date + timedelta(days=final_day_date+1)
            label = user_df[user_df['date'] == label_date]['label']
            
            if n_empty > empty_threshold or label.empty:
                break
            else:
                samples.append((sample, label.item()))          
    return samples        

def sliding_window_baseline(dataset):
    # Sliding window over all days per user. From each day look 5 days in the future and store the features
    # of available dates. Check if the 6th day has a mood label. If so, store the entire sequence with the
    # corresponding label. Each sequence also needs at least a certain amount of days available or it will
    # be discarded.
    interval_size = 5
    empty_threshold = 3

    users = list(dataset['id'].unique())

    samples = []

    for user in users:
        user_df = dataset[dataset['id'] == user]
        for index, row in user_df.iterrows():
            sample = []
            first_date = row['date']
            n_empty = 0

            current_date = None
            final_day_date = 0

            for interval_date in range(interval_size):

                final_day_date += 1

                current_date = first_date + timedelta(days=interval_date)
                current_date_features = user_df[user_df['date'] == current_date]
                if current_date_features.empty:
                    n_empty += 1
                else:
                    sample.append(current_date_features['label'].item())

            label_date = first_date + timedelta(days=final_day_date+1)
            label = user_df[user_df['date'] == label_date]['label']

            if n_empty > empty_threshold or label.empty:
                break
            else:
                samples.append((sample, label.item()))
    return samples