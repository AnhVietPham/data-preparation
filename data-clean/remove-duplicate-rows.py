from pandas import read_csv

if __name__ == "__main__":
    df = read_csv("../data-clean/data/iris.csv", header=None)
    print(df.shape)
    df.drop_duplicates(inplace=True)
    print(df.shape)
