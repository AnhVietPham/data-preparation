from numpy import isnan
from pandas import read_csv
from sklearn.impute import SimpleImputer

if __name__ == "__main__":
    df = read_csv("../use-statistical-imputation/data/horse-colic.csv", header=None, na_values='?')
    data = df.values
    ix = [i for i in range(data.shape[1]) if i != 23]
    X, y = data[:, ix], data[:, 23]
    print(f'Missing: {sum(isnan(X).flatten())}')
    print(X)
    imputer = SimpleImputer(strategy='mean')
    imputer.fit(X)
    Xtrans = imputer.transform(X)
    print("=" * 50)
    print(f'Missing: {sum(isnan(Xtrans).flatten())}')
    print(Xtrans)
