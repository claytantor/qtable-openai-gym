import os,sys
import pandas as pd



def load_dataset(name, index_name):
    path = os.path.join('/workspace/data', name + '.csv')
    df = pd.read_csv(path, parse_dates=True, index_col=index_name)
    return df


