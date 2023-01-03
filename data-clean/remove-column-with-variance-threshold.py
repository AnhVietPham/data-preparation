from pandas import read_csv
from sklearn.feature_selection import VarianceThreshold
from matplotlib import pyplot
import numpy as np

if __name__ == "__main__":
    df = read_csv("../data-clean/data/oil-spill.csv", header=None)
    print(df.shape)
    data = df.values
    X = data[:, :-1]
    y = data[:, -1]
    print(X.shape, y.shape)
    transform = VarianceThreshold()
    X_sel = transform.fit_transform(X)
    print(X_sel.shape)
    thresholds = np.arange(0.0, 0.55, 0.05)
    results = list()
    for t in thresholds:
        transform = VarianceThreshold(threshold=t)
        X_sel = transform.fit_transform(X)
        n_features = X_sel.shape[1]
        print(f'Threshold {t}, Features = {n_features}')
        results.append(n_features)
    pyplot.plot(thresholds, results)
    pyplot.show()