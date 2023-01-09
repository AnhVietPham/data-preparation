from numpy import isnan
from pandas import read_csv
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

if __name__ == "__main__":
    df = read_csv("../use-statistical-imputation/data/horse-colic.csv", header=None, na_values="?")
    data = df.values
    ix = [i for i in range(data.shape[1]) if i != 23]
    X, y = data[:, ix], data[:, 23]
    print(f'Missing: {sum(isnan(X).flatten())}')
    imputer = IterativeImputer()
    imputer.fit(X)
    Xtrains = imputer.transform(X)
    print(f'Missing: {sum(isnan(Xtrains).flatten())}')
