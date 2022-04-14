import pandas as pd

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
    aggregated_dataset['value'] = aggregated_dataset.apply(select_aggregate_type, axis=1)
    aggregated_dataset = aggregated_dataset.drop(columns=['mean_value', 'sum_value'])
    pivot_table = aggregated_dataset.pivot_table(index=['id', 'date'], 
                                             columns=['variable'], values='value').fillna(0).reset_index()
    return pivot_table