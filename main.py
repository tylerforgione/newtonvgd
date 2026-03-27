import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import fetch_openml
from evaluate_models import evaluate_best_config


def main():
    # main just used for data then we run the regression based on the method
    x, y = fetch_openml(name='mnist_784', version=1, return_X_y=True)

    x = x.astype(np.float32)

    encoder = OneHotEncoder(sparse_output=False)
    y = encoder.fit_transform(y.to_numpy().reshape(-1, 1))

    x = x / 255.0

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=17)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=17)

    # softmax time
    # get_gridsearch_results(x_train, y_train, x_val, y_val, x_test, y_test, method='gd')
    # get_gridsearch_results(x_train, y_train, x_val, y_val, x_test, y_test, method='cg')

    # now run best
    best_params = (50, 0.00966829572656517, 32, 0)
    results = evaluate_best_config(x_train, y_train, x_val, y_val, x_test, y_test, method='gd', best_params=best_params)
    print(tuple(sum(col) / len(col) for col in zip(*results)))

    best_params = (6, None, None, 6.067738140370665e-05)
    results = evaluate_best_config(x_train, y_train, x_val, y_val, x_test, y_test, method='cg', best_params=best_params)
    print(tuple(sum(col) / len(col) for col in zip(*results)))

    best_params = (4, None, None, 0.0003103996573489349)
    results = evaluate_best_config(x_train, y_train, x_val, y_val, x_test, y_test, method='cg', best_params=best_params)
    print(tuple(sum(col) / len(col) for col in zip(*results)))


if __name__ == '__main__':
    main()
