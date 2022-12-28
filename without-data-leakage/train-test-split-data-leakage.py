from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=7)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    yhat = model.predict(X_test)
    accuracy = accuracy_score(y_test, yhat)
    print(f'Accuracy: {accuracy * 100}')
