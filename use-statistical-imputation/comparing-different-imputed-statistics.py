from numpy import mean
from numpy import std
from pandas import read_csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from matplotlib import pyplot

if __name__ == "__main__":
    df = read_csv("../use-statistical-imputation/data/horse-colic.csv", header=None, na_values="?")
    data = df.values
    ix = [i for i in range(data.shape[1]) if i != 23]
    X, y = data[:, ix], data[:, 23]
    results = list()
    strategies = ['mean', 'median', 'most_frequent', 'constant']
    for s in strategies:
        pipeline = Pipeline(steps=[('i', SimpleImputer(strategy=s)), ('m', RandomForestClassifier())])
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        scores = cross_val_score(pipeline, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
        results.append(scores)
        print(f"{s, mean(scores), std(scores)}")
    pyplot.boxplot(results, labels=strategies, showmeans=True)
    pyplot.show()
