from numpy import loadtxt
from numpy import unique
from pandas import read_csv

if __name__ == "__main__":
    # data = loadtxt("../data-clean/data/oil-spill.csv", delimiter=',')
    # for i in range(data.shape[1]):
    #     print(i, len(unique(data[:, i])))

    # load the dataset using Pandas function
    df = read_csv("../data-clean/data/oil-spill.csv", header=None)
    print(df.shape)
    counts = df.nunique()
    rm = [i for i, v in enumerate(counts) if v == 1]
    print(rm)
    df.drop(rm, axis=1, inplace=True)
    print(df.shape)
