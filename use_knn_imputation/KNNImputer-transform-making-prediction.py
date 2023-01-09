from numpy import nan
from pandas import read_csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

if __name__ == "__main__":
    df = read_csv("../use-statistical-imputation/data/horse-colic.csv", header=None, na_values="?")
    data = df.values
    ix = [i for i in range(data.shape[1]) if i != 23]
    X, y = data[:, ix], data[:, 23]
    pipeline = Pipeline(steps=[('i', KNNImputer(n_neighbors=21)), ('m', RandomForestClassifier())])
    pipeline.fit(X, y)
    row = [2, 1, 530101, 38.50, 66, 28, 3, 3, nan, 2, 5, 4, 4, nan, nan, nan, 3, 5, 45.00,
           8.40, nan, nan, 2, 11300, 00000, 00000, 2]
    yhat = pipeline.predict([row])
    print(f'Predicted class: %d ' % yhat)
