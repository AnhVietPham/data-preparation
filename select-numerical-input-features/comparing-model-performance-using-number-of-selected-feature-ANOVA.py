from numpy import mean
from numpy import std
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from matplotlib import pyplot


def load_dataset(filename):
    df = read_csv(filename, header=None)
    data = df.values
    X = data[:, : -1]
    y = data[:, -1]
    return X, y


def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores


if __name__ == "__main__":
    X, y = load_dataset("../select-numerical-input-features/data/pima-indians-diabetes.csv")
    num_features = [i + 1 for i in range(X.shape[1])]
    results = list()
    for k in num_features:
        model = LogisticRegression(solver='liblinear')
        fs = SelectKBest(score_func=f_classif, k=k)
        pipeline = Pipeline(steps=[('anova', fs), ('lr', model)])
        scores = evaluate_model(pipeline, X, y)
        results.append(scores)
        print('>%d %.3f (%.3f)' % (k, mean(scores), std(scores)))

    pyplot.boxplot(results, labels=num_features, showmeans=True)
    pyplot.show()
