import pandas as pd

def load_data(file_path):
    data = pd.read_csv(file_path)
    data = data.drop(labels=["Datetime"], axis=1)
    data.replace('#VALUE!', 0, inplace=True)
    data.dropna()
    return data
