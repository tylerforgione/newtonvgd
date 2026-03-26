import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from logisticregression import LogisticRegression
from softmax import SoftmaxRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import time
from itertools import product


def sklearn_to_df(data_loader):
    X_data = data_loader.data
    X_columns = data_loader.feature_names
    x = pd.DataFrame(X_data, columns=X_columns)

    y_data = data_loader.target
    y = pd.Series(y_data, name='target')

    return x, y


def run_logreg(x_train, y_train, x_val, y_val, x_test, y_test, method='gd', epochs=1000, lr=0.0001, batch_size=None):
    logreg = LogisticRegression()
    print("Training logistic regression using", method)
    start = time.time()
    metrics = logreg.fit(x_train, y_train, x_val, y_val, method, epochs, lr, batch_size)
    end = time.time()
    print("Training time: ", end - start)
    print("Val accuracy: ", logreg.score(x_test, y_test))
    return logreg, metrics, logreg.score(x_test, y_test)

def run_softmax(x_train, y_train, x_val, y_val, x_test, y_test, method='gd', epochs=1000, lr=0.0001, batch_size=None, lamb=0.0):
    softmax = SoftmaxRegression()
    print("Training softmax regression using", method)
    start = time.time()
    metrics = softmax.fit(x_train, y_train, x_val, y_val, method, epochs, lr, batch_size, lamb)
    end = time.time()
    print("Training time: ", end - start)
    print("Val accuracy: ", softmax.score(x_test, y_test))
    return softmax, metrics, softmax.score(x_test, y_test)


def main():
    # main just used for data then we run the regression based on the method
    x, y = fetch_openml(name='mnist_784', version=1, return_X_y=True)

    encoder = OneHotEncoder(sparse_output=False)
    y = encoder.fit_transform(y.to_numpy().reshape(-1, 1))

    x = x / 255.0

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=17)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=17)

    # run the actual thing
    # run_logreg(x_train, y_train, x_val, y_val, x_test, y_test, method='gd')
    # run_logreg(x_train, y_train, x_val, y_val, x_test, y_test, method='newton', epochs=1)
    # run_logreg(x_train, y_train, x_val, y_val, x_test, y_test, method='cholesky', epochs=1)
    # run_logreg(x_train, y_train, x_val, y_val, x_test, y_test, method='cg', epochs=1)

    # softmax time
    epochs_range = [10, 20, 50]
    lr_range = [1e-4, 5e-4, 1e-3, 5e-3]
    batch_size_range = [32, 64, 128]
    lambda_range = [0, 1e-5, 1e-4, 1e-3, 1e-2]
    best_val_acc = 0.0
    best_params = None
    best_model = None
    for epochs, lr, batch_size, lamb in product(epochs_range, lr_range, batch_size_range, lambda_range):
        print("\nnumber of epochs:",epochs, ", learning rate:", lr, ", batch size:", batch_size, ", lambda:", lamb)
        model, _, val_acc = run_softmax(x_train, y_train, x_val, y_val, x_val, y_val, method='gd', epochs=epochs, lr=lr,
                                        batch_size=batch_size, lamb=lamb)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = {'epochs': epochs, 'lr': lr, 'batch_size': batch_size, 'lamb': lamb}
            best_model = model

    print("\nBest validation accuracy:", best_val_acc)
    print("Best parameters:", best_params)
    print("Trying best model on test set")
    print(best_model.score(x_test, y_test))

    best_val_acc = 0.0
    best_params = None
    best_model = None
    lambda_range = [0, 1e-5, 1e-4, 1e-3, 1e-2]
    for epochs, lamb in product(range(1, 10), lambda_range):
        print("\nnumber of epochs:",epochs, ", lambda:", lamb)
        model, _, val_acc = run_softmax(x_train, y_train, x_val, y_val, x_val, y_val, method='cg', epochs=epochs, lamb=lamb)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = {'epochs': epochs, 'lamb': lamb}
            best_model = model

    print("\nBest validation accuracy:", best_val_acc)
    print("Best parameters:", best_params)
    print("Trying best model on test set")
    print(best_model.score(x_test, y_test))


if __name__ == '__main__':
    main()
