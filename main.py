from pickle import GLOBAL

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from logisticregression import LogisticRegression
from softmax import SoftmaxRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import time
from itertools import product
from joblib import Parallel, delayed, parallel_backend
import os
from tqdm import tqdm

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def run_logreg(x_train, y_train, x_val, y_val, x_eval, y_eval, method='gd', epochs=1000, lr=0.0001, batch_size=None):
    logreg = LogisticRegression()
    print("Training logistic regression using", method)
    start = time.time()
    metrics = logreg.fit(x_train, y_train, x_val, y_val, method, epochs, lr, batch_size)
    end = time.time()
    print("Training time: ", end - start)
    print("Val accuracy: ", logreg.score(x_eval, y_eval))
    return logreg, metrics, logreg.score(x_eval, y_eval)


def run_softmax(x_train, y_train, x_val, y_val, x_eval, y_eval, method='gd', epochs=1000, lr=0.0001, batch_size=None,
                lamb=0.0, verbose=False):
    softmax = SoftmaxRegression()
    if verbose:
        print("Training softmax regression using", method)
    start = time.time()
    metrics = softmax.fit(x_train, y_train, x_val, y_val, method, epochs, lr, batch_size, lamb)
    end = time.time()
    if verbose:
        print("Training time: ", end - start)
    acc = softmax.score(x_eval, y_eval)
    if verbose:
        print("Validation accuracy: ", acc)
    return softmax, metrics, acc


GLOBAL_DATA = {}


def evaluate_config(method, params):
    x_train = GLOBAL_DATA["x_train"]
    y_train = GLOBAL_DATA["y_train"]
    x_val = GLOBAL_DATA["x_val"]
    y_val = GLOBAL_DATA["y_val"]

    epochs, lr, batch_size, lamb = params

    model, _, val_acc = run_softmax(
        x_train, y_train, x_val, y_val, x_val, y_val,
        method=method,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        lamb=lamb
    )

    return val_acc, params, model


def sample_grid_gd(num_samples):
    params = []
    for _ in range(num_samples):
        epochs = np.random.choice([10, 20, 50])
        lr = 10 ** np.random.uniform(-4, -2)
        batch_size = np.random.choice([32, 64, 128])
        lamb = 10 ** np.random.uniform(-5, -2)

        if np.random.rand() < 0.5:
            lamb = 0

        params.append([epochs, lr, batch_size, lamb])

    return params


def sample_grid_cg(num_samples):
    params = []
    for _ in range(num_samples):
        epochs = np.random.randint(1, 10)
        lamb = 10 ** np.random.uniform(-5, -2)
        if np.random.rand() < 0.2:
            lamb = 0
        params.append([epochs, [None], [None], lamb])

    return params


def grid_search_softmax(x_train, y_train, x_val, y_val, method='gd'):
    if method == 'gd':
        param_list = sample_grid_gd(40)
    elif method == 'cg':
        param_list = sample_grid_cg(15)
    else:
        raise ValueError('Unknown method')

    GLOBAL_DATA["x_train"] = x_train
    GLOBAL_DATA["y_train"] = y_train
    GLOBAL_DATA["x_val"] = x_val
    GLOBAL_DATA["y_val"] = y_val

    results = Parallel(n_jobs=4)(
        delayed(evaluate_config)(
            method, param
        )
        for param in tqdm(param_list)
    )

    best_val_acc, best_params, best_model = max(results, key=lambda x: x[0])

    return best_model, best_params, best_val_acc


def main():
    # main just used for data then we run the regression based on the method
    x, y = fetch_openml(name='mnist_784', version=1, return_X_y=True)

    encoder = OneHotEncoder(sparse_output=False)
    y = encoder.fit_transform(y.to_numpy().reshape(-1, 1))

    x = x / 255.0

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=17)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=17)

    # softmax time
    best_model, best_params, best_val_acc = grid_search_softmax(x_train, y_train, x_val, y_val, 'gd')
    print('Best parameters:', best_params)
    print('Best validation accuracy:', best_val_acc)
    print('Score on test set:', best_model.score(x_test, y_test))

    best_model, best_params, best_val_acc = grid_search_softmax(x_train, y_train, x_val, y_val, 'cg')
    print('Best parameters:', best_params)
    print('Best validation accuracy:', best_val_acc)
    print('Score on test set:', best_model.score(x_test, y_test))


if __name__ == '__main__':
    main()
