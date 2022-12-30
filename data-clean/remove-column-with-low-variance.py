from pandas import read_csv

if __name__ == "__main__":
    df = read_csv("../data-clean/data/oil-spill.csv", header=None)
    print(df.shape)
    counts = df.nunique()
    rm = [i for i, v in enumerate(counts) if (float(v) / df.shape[0] * 100) < 1]
    print(rm)
    df.drop(rm, axis=1, inplace=True)
    print(df.shape)
