#! /usr/bin/python3

import numpy as np
import matplotlib
matplotlib.use("agg")
# import matplotlib.pyplot as plt  # noqa: E402
from scipy import optimize as opt  # noqa: E402

import error_functions as erf  # noqa: E402
import models as mod  # noqa: E402

# ERROR_FUNCTION = erf.rms
ERROR_FUNCTION = erf.mae
# ERROR_FUNCTION = erf.maxd
# ERROR_FUNCTION = erf.smr


def loss_fun(p, x, y, eval_fun, error_fun):
    predictions = eval_fun(p, x)
    deviations = predictions - y
    error = error_fun(deviations)
    return error


def split_data(x=None, y=None, extrapolate=False):
    n_train = int(y.size * .7)
    if extrapolate:
        # Extrapolation
        x_train = x[:n_train]
        x_test = x[n_train:]
        y_train = y[:n_train]
        y_test = y[n_train:]
    else:
        # Interpolation
        i_data = np.cumsum(np.ones(x.size), dtype=np.int) - 1
        i_train = np.sort(
            np.random.choice(i_data, size=n_train, replace=False))
        i_test = np.setdiff1d(i_data, i_train)
        x_train = x[i_train]
        x_test = x[i_test]
        y_train = y[i_train]
        y_test = y[i_test]

    return x_train, y_train, x_test, y_test


def compare_models(x_train, y_train, x_test, y_test):
    """
    Fit a data set with a variety of models and figure out which fits best.
    """
    training_errors = []
    testing_errors = []

    models = mod.all_models
    for model in models:
        res = train(model, x_train, y_train, n_iter=model.n_iter_default)
        p_final = res.x

        training_errors.append(loss_fun(
            p_final, x_train, y_train, model.evaluate, ERROR_FUNCTION))
        testing_errors.append(loss_fun(
            p_final, x_test, y_test, model.evaluate, ERROR_FUNCTION))

    return models, training_errors, testing_errors


def train(model, x_train, y_train, n_iter=3):
    best_res = None
    best_loss = 1e10
    for _ in range(n_iter):
        # The arguments that will get passed to the error function,
        # in addition to the model parameters of the current iteration.
        error_fun_args = (x_train, y_train, model.evaluate, ERROR_FUNCTION)
        p_initial = model.initial_guess(x=x_train, y=y_train)
        # Confusingly the `x0` argument is not a request for the x values of
        # the data points or for the 0th element of a list. It is for the
        # initial guess for the parameter values.
        res = opt.minimize(
            fun=loss_fun,
            x0=p_initial,
            method="Nelder-Mead",
            args=error_fun_args,
        )
        loss = loss_fun(
            res.x,
            x_train,
            y_train,
            model.evaluate,
            ERROR_FUNCTION,
        )
        if loss < best_loss:
            best_loss = loss
            best_res = res

    return best_res


def test():
    """
    Check whether the code is doing what it should.
    """
    compare_models()


if __name__ == "__main__":
    test()
