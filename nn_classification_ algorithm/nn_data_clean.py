import pandas as pd


def clean_data(df):
  df = df.dropna()  # Remove rows with any NaN values
  return df

