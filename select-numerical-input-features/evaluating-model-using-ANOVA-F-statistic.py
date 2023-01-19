from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def load_dataset(filename):
    df = read_csv(filename, header=None)
    data = df.values
    X = data[:, :-1]
    y = data[:, -1]
    return X, y


def select_features(X_train, y_train, X_test):
    fs = SelectKBest(score_func=f_classif, k=4)
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs


if __name__ == "__main__":
    X, y = load_dataset("../select-numerical-input-features/data/pima-indians-diabetes.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
    model = LogisticRegression(solver='liblinear')
    model.fit(X_train_fs, y_train)
    yhat = model.predict(X_test_fs)
    accuracy = accuracy_score(y_test, yhat)
    print('Accuracy: %.2f' % (accuracy * 100))
