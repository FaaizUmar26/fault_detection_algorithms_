import pandas as pd

def clean_data(data):
    data = data.drop(labels=["Datetime"], axis=1)
    data.replace('#VALUE!', 0, inplace=True)
    data.dropna()
    return data
