from pandas import read_csv

if __name__ == "__main__":
    df = read_csv("../data-clean/data/iris.csv")
    dups = df.duplicated()
    print(dups.any())
    print(df[dups])
