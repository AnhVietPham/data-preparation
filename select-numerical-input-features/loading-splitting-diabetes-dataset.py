from pandas import read_csv
from sklearn.model_selection import train_test_split


def load_dataset(filename):
    df = read_csv(filename, header=None)
    data = df.values
    X = data[:, :-1]
    y = data[:, -1]
    return X, y


if __name__ == "__main__":
    X, y = load_dataset("../select-numerical-input-features/data/pima-indians-diabetes.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    print('Train', X_train.shape, y_train.shape)
    print('Test', X_test.shape, y_test.shape)
