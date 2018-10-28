#! /usr/bin/python3

import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt  # noqa: E402

import data_generator as gen  # noqa: E402
import models as mod  # noqa: E402
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


def show_model(model, x, y):
    plot_data(x, y)
    res = sel.train(model, x, y, n_iter=model.n_iter_default)
    p_final = res.x
    y_predicted = model.evaluate(p_final, x)
    plt.plot(x, y_predicted, linewidth=2)
    finalize_plot(model.name)
    return


def main():
    """
    Check whether the code is doing what it should.
    """
    x, y = prepare_data()
    show_just_data(x, y)
    for model in mod.all_models:
        show_model(model, x, y)
    show_data_connected(x, y)
    # compare_models(x, y)
    return


if __name__ == "__main__":
    main()
