from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import LocalOutlierFactor

if __name__ == "__main__":
    df = read_csv("../outlider-identification-and-remove/data/housing.csv", header=None)
    data = df.values
    X, y = data[:, :-1], data[:, -1]
    print(X.shape, y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    model = LinearRegression()
    model.fit(X_train, y_train)
    yhat = model.predict(X_test)
    mae = mean_absolute_error(y_test, yhat)
    print(f'MAE: {mae}')
    lof = LocalOutlierFactor()
    yhat = lof.fit_predict(X_train)
    mask = yhat != -1
    X_train, y_train = X_train[mask, :], y_train[mask]
    print(X_train.shape, y_train.shape)
    model = LinearRegression()
    model.fit(X_train, y_train)
    yhat = model.predict(X_test)
    mae = mean_absolute_error(y_test, yhat)
    print(f'Using LOF MAE: {mae}')
