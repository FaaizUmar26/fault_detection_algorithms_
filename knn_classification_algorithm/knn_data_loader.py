from sklearn.model_selection import train_test_split
from knn_data_clean import load_data

def split_data(data):
    X = data.iloc[:, :-1]  # Features columns .all columns are slected except the last one
    y = data.iloc[:, -1]   # Labels columns .only the last column is selected
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
