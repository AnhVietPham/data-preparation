from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

if __name__ == "__main__":
    # X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    # y = np.array([0, 0, 1, 1])
    # rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=2,
    #                                random_state=7)
    # print(rskf.get_n_splits(X, y))
    # for i, (train_index, test_index) in enumerate(rskf.split(X, y)):
    #     print(f"Fold {i}:")
    #     print(f"  Train: index={train_index}")
    #     print(f"  Test:  index={test_index}")
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    model = LogisticRegression()
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    print(f'Accuracy mean: {mean(scores) * 100}')
