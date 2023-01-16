from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


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


if __name__ == "__main__":
    X, y = load_dataset("../select-categorical-input-features/data/breast-cancer.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    X_train_enc, X_test_enc = prepare_inputs(X_train, X_test)
    y_train_enc, y_test_enc = prepare_targets(y_train, y_test)
    model = LogisticRegression(solver='lbfgs')
    model.fit(X_train_enc, y_train_enc)
    yhat = model.predict(X_test_enc)
    accuracy = accuracy_score(y_test_enc, yhat)
    print('Accuracy: %.2f' % (accuracy * 100))
