from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from matplotlib import pyplot


def load_dataset(filename):
    df = read_csv(filename, header=None)
    data = df.values
    X = data[:, :-1]
    y = data[:, -1]
    X = X.astype(str)
    return X, y


def prepare_inputs(X_train, X_test):
    oe = OrdinalEncoder()
    oe.fit(X_train)
    X_train_enc = oe.transform(X_train)
    X_test_enc = oe.transform(X_test)
    return X_train_enc, X_test_enc


def prepare_targets(y_train, y_test):
    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)
    return y_train_enc, y_test_enc


def select_features(X_train, y_train, X_test):
    fs = SelectKBest(score_func=chi2, k='all')
    fs.fit(X_train, y_train)
    X_train_fs = fs.transform(X_train)
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs


if __name__ == "__main__":
    X, y = load_dataset("../select-categorical-input-features/data/breast-cancer.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    X_train_enc, X_test_enc = prepare_inputs(X_train, X_test)
    y_train_enc, y_test_enc = prepare_targets(y_train, y_test)
    X_train_fs, X_test_fs, fs = select_features(X_train_enc, y_train_enc, X_test_enc)
    for i in range(len(fs.scores_)):
        print('Feature %d: %f' % (i, fs.scores_[i]))

    pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
    pyplot.show()
