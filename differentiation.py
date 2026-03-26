import numpy as np


# two point methods
def forward_difference_1(func, x, h=1e-6) -> float:
    return (func(x + h) - func(x)) / h


def backward_difference_1(func, x, h=1e-6) -> float:
    return (func(x) - func(x - h)) / h


def central_difference_1(func, x, h=1e-6) -> float:
    return (func(x + h) - func(x - h)) / (2 * h)


def complex_step(func, x, h=1e-20) -> float:
    return np.imag(func(x + h * 1j)) / h


# second derivatives
def forward_difference_2(func, x, h=1e-6) -> float:
    return (func(x + 2 * h) - 2 * func(x + h) + func(x)) / (h * h)


def backward_difference_2(func, x, h=1e-6) -> float:
    return (func(x) - 2 * func(x - h) + func(x - 2 * h)) / (h * h)


def central_difference_2(func, x, h=1e-6) -> float:
    return (func(x - h) - 2 * func(x) + func(x + h)) / (h * h)


def main():
    import time
    func = np.sin
    start = time.time()
    print(forward_difference_1(func, 0))
    end = time.time()
    print((end - start) * 1000 * 1000, "ns")
    start = time.time()
    print(backward_difference_1(func, 0))
    end = time.time()
    print((end - start) * 1000 * 1000, "ns")
    start = time.time()
    print(central_difference_1(func, 0))
    end = time.time()
    print((end - start) * 1000 * 1000, "ns")
    start = time.time()
    print(complex_step(func, 0))
    end = time.time()
    print((end - start) * 1000 * 1000, "ns")

    print("real:", func(0))


if __name__ == "__main__":
    main()
