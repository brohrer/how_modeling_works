#! /usr/bin/python3

import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt  # noqa: E402

import data_generator as gen  # noqa: E402
# import models as mod  # noqa: E402
import model_selector as sel  # noqa: E402


def prepare_data():
    y = gen.simulate_annual()
    x = np.cumsum(np.ones(y.size)) - 1
    return x, y


def plot_data(x, y):
    plt.figure(num=7893, figsize=(8, 4.5))
    plt.clf()
    plt.style.use('dark_background')
    plt.plot(x, y, linestyle="none", marker='.', markersize=6)
    plt.xlabel("Year")
    plt.ylabel("Typical high temp")
    plt.gca().set_xlim(-3, 123)
    plt.gca().set_ylim(8, 16)
    return


def finalize_plot(plotname):
    plt.title(plotname)
    filename = "_".join(plotname.split()) + ".png"
    plt.savefig(filename, dpi=300)
    return


def show_just_data(x, y):
    plot_data(x, y)
    finalize_plot("Typical daily highs")
    return


def show_data_connected(x, y):
    plot_data(x, y)
    plt.plot(x, y)
    finalize_plot("Interpolation model")
    return


def show_data_connected_test(x_train, y_train, x_test, y_test):
    plot_data(x_test, y_test)
    plt.plot(x_train, y_train)
    finalize_plot("Interpolation model test")
    return


def show_model(model, x, y):
    plot_data(x, y)
    res = sel.train(model, x, y, n_iter=model.n_iter_default)
    p_final = res.x
    y_predicted = model.evaluate(p_final, x)
    plt.plot(x, y_predicted, linewidth=2)
    finalize_plot(model.name)
    return


def show_model_test(model, x_train, y_train, x_test, y_test):
    plot_data(x_test, y_test)
    res = sel.train(model, x_train, y_train, n_iter=model.n_iter_default)
    p_final = res.x
    y_predicted = model.evaluate(p_final, x_test)
    plt.plot(x_test, y_predicted, linewidth=2)
    finalize_plot(model.name + " test")
    return


def show_errors(models, training_errors, testing_errors):
    plt.figure(num=9437, figsize=(8, 4.5))
    plt.clf()
    plt.style.use('dark_background')
    for i_model, model in enumerate(models):
        try:
            order = model.order
        except Exception:
            continue
        if order > 8:
            continue

        plt.plot(
            order,
            testing_errors[i_model],
            linestyle="none",
            color="white",
            marker='.',
            markersize=12,
        )
        plt.plot(
            order,
            training_errors[i_model],
            linestyle="none",
            marker='o',
            markersize=6,
            markerfacecolor="none",
            markeredgecolor="white",
        )

    plt.xlabel("Polynomial model order")
    plt.ylabel("Error (hollow=training, solid=testing)")
    plt.gca().set_xlim(0, 9)
    plt.gca().set_ylim(.5, 1)
    finalize_plot("Fit errors")
    return


def main():
    """
    Check whether the code is doing what it should.
    """
    x, y = prepare_data()
    # show_just_data(x, y)
    # for model in mod.all_models:
    #     show_model(model, x, y)
    # show_data_connected(x, y)

    x_train, y_train, x_test, y_test = sel.split_data(x, y)
    # for model in mod.all_models:
    #     show_model_test(model, x_train, y_train, x_test, y_test)
    # show_data_connected_test(x_train, y_train, x_test, y_test)

    models, training_errors, testing_errors = sel.compare_models(
        x_train, y_train, x_test, y_test)
    show_errors(models, training_errors, testing_errors)

    return


if __name__ == "__main__":
    main()
