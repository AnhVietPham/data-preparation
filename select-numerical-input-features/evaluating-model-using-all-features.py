from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def load_dataset(filename):
    df = read_csv(filename, header=None)
    data = df.values
    X = data[:, :-1]
    y = data[:, -1]
    return X, y


if __name__ == "__main__":
    X, y = load_dataset('../select-numerical-input-features/data/pima-indians-diabetes.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    model = LogisticRegression(solver='liblinear')
    model.fit(X_train, y_train)
    yhat = model.predict(X_test)
    accuracy = accuracy_score(y_test, yhat)
    print('Accuracy: %.2f' % (accuracy * 100))
