from numpy import isnan
from pandas import read_csv
from sklearn.impute import KNNImputer

if __name__ == "__main__":
    df = read_csv("../use-statistical-imputation/data/horse-colic.csv", header=None, na_values="?")
    data = df.values
    ix = [i for i in range(df.shape[1]) if i != 23]
    X, y = data[:, ix], data[:, 23]
    print('Missing: %d' % sum(isnan(X).flatten()))
    imputer = KNNImputer()
    imputer.fit(X)
    Xtrains = imputer.transform(X)
    print('Missing: %d' % sum(isnan(Xtrains).flatten()))
