import time
from joblib import Parallel, delayed
import os
import numpy as np
from tqdm import tqdm
from logisticregression import LogisticRegression
from softmax import SoftmaxRegression
import csv

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


def run_logreg(x_train, y_train, x_val, y_val, x_eval, y_eval, method='gd', epochs=1000, lr=0.0001, batch_size=None):
    logreg = LogisticRegression()
    print("Training logistic regression using", method)
    start = time.time()
    metrics = logreg.fit(x_train, y_train, x_val, y_val, method, epochs, lr, batch_size)
    end = time.time() - start
    print("Training time: ", end)
    print("Val accuracy: ", logreg.score(x_eval, y_eval))
    return logreg, metrics, logreg.score(x_eval, y_eval), end


def run_softmax(x_train, y_train, x_val, y_val, x_eval, y_eval, method='gd', epochs=1000, lr=0.0001, batch_size=None,
                lamb=0.0, verbose=False, early_stopping=True, patience=3, tolerance=1e-3):
    softmax = SoftmaxRegression()
    if verbose:
        print("Training softmax regression using", method)
    start = time.time()
    metrics = softmax.fit(x_train, y_train, x_val, y_val, method, epochs, lr, batch_size, lamb, early_stopping,
                          patience, tolerance)
    end = time.time() - start
    if verbose:
        print("Training time: ", end)
    acc = softmax.score(x_eval, y_eval)
    if verbose:
        print("Validation accuracy: ", acc)
    return softmax, metrics, acc, end


GLOBAL_DATA = {}


def evaluate_config(method, params):
    x_train = GLOBAL_DATA["x_train"]
    y_train = GLOBAL_DATA["y_train"]
    x_val = GLOBAL_DATA["x_val"]
    y_val = GLOBAL_DATA["y_val"]

    epochs, lr, batch_size, lamb = params

    model, _, val_acc, train_time = run_softmax(
        x_train, y_train, x_val, y_val, x_val, y_val,
        method=method,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        lamb=lamb
    )

    return val_acc, params, model, train_time


def sample_grid_gd(num_samples):
    params = []
    for _ in range(num_samples):
        epochs = np.random.choice([10, 20, 50, 75])
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
        params.append([epochs, None, None, lamb])

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

    best_val_acc, best_params, best_model, _ = max(results, key=lambda x: x[0])

    return best_model, best_params, best_val_acc, results


def get_gridsearch_results(x_train, y_train, x_val, y_val, x_test, y_test, method):
    best_model, best_params, best_val_acc, results = grid_search_softmax(x_train, y_train, x_val, y_val, method)
    save_results(results, 'output/gd.csv')
    print('Best parameters:', best_params)
    print('Best validation accuracy:', best_val_acc)
    print('Score on test set:', best_model.score(x_test, y_test))


def save_results(results, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    results = sorted(results, key=lambda x: x[0], reverse=True)

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['rank', 'val_acc', 'epochs', 'lr', 'batch_size', 'lamb', 'time'])

        for i, (val_acc, params, _, train_time) in enumerate(results):
            epochs, lr, batch_size, lamb = params
            writer.writerow([i + 1, val_acc, epochs, lr, batch_size, lamb, train_time])


def evaluate_best_config(x_train, y_train, x_val, y_val, x_test, y_test, method, best_params, runs=5):
    result = []

    os.environ["OPENBLAS_NUM_THREADS"] = "-1"
    os.environ["OMP_NUM_THREADS"] = "-1"
    os.environ["MKL_NUM_THREADS"] = "-1"

    for i in range(runs):
        model, _, val_acc, train_time = run_softmax(
            x_train, y_train, x_val, y_val, x_val, y_val,
            method=method,
            epochs=best_params[0],
            lr=best_params[1],
            batch_size=best_params[2],
            lamb=best_params[3]
        )

        test_acc = model.score(x_test, y_test)

        result.append((val_acc, test_acc, train_time))

    return result
