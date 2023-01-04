from numpy import isnan
from pandas import read_csv
from sklearn.impute import SimpleImputer

if __name__ == "__main__":
    df = read_csv("../use-statistical-imputation/data/horse-colic.csv")
    data = df.values
    ix = [i for i in range(data.shape[1]) if i != 23]
