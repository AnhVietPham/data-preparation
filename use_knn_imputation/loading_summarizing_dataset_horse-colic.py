from pandas import read_csv

if __name__ == "__main__":
    df = read_csv("../use-statistical-imputation/data/horse-colic.csv", header=None, na_values="?")
    print(df.head())
    for i in range(df.shape[1]):
        n_miss = df[[i]].isnull().sum()
        perc = n_miss / df.shape[0] * 100
        print('> %d, Missing: %d (%.1f%%)' % (i, n_miss, perc))
