from pandas import read_csv
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from numpy import nan

if __name__ == "__main__":
    datasets = read_csv("../mark-remove-missing-data/data/pima-indians-diabetes.csv", header=None)
    num_missing = (datasets[[1, 2, 3, 4, 5]] == 0).sum()
    datasets[[1, 2, 3, 4, 5]] = datasets[[1, 2, 3, 4, 5]].replace(0, nan)
    print(datasets.head(20))
    datasets.dropna(inplace=True)
    print(datasets.head(20))
    values = datasets.values
    X = values[:, : -1]
    y = values[:, -1]
    model = LinearDiscriminantAnalysis()
    cv = KFold(n_splits=3, shuffle=True, random_state=1)
    result = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"Accuracy: {result.mean()}")
