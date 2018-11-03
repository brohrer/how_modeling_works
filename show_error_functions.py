import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_error_function(x, y, name):
    plt.figure(num=7878767, figsize=(8, 4.5))
    x_min = -1.5
    x_max = 1.5
    y_min = 0
    y_max = 1.7
    border = 0
    plt.clf()
    plt.style.use('dark_background')
    plt.plot(x, y, linewidth=4)
    plt.plot(
        [x_min, x_max],
        [0, 0],
        linewidth=.5,
        color="gray",
    )
    plt.plot(
        [0, 0],
        [y_min, y_max],
        linewidth=.5,
        color="gray",
    )
    for x in [-2, -1, 1, 2]:
        plt.plot(
            [x, x],
            [y_min, y_max],
            linewidth=.5,
            linestyle="--",
            color="gray",
        )
    for y in [1, 2]:
        plt.plot(
            [x_min, x_max],
            [y, y],
            linewidth=.5,
            linestyle="--",
            color="gray",
        )

    plt.xlabel("Deviation")
    plt.ylabel("Error")
    plt.gca().set_xlim(x_min - border, x_max + border)
    plt.gca().set_ylim(y_min - border, y_max + border)
    plt.title(name)
    filename = "_".join(name.lower().split()) + ".png"
    plt.savefig(filename, dpi=300)
    return


devs = np.linspace(-2, 2, 200)

errors = devs**2
plot_error_function(devs, errors, "Squared deviation")

errors = np.abs(devs)
plot_error_function(devs, errors, "Absolute deviation")

errors = np.minimum(1, np.abs(devs))
plot_error_function(devs, errors, "Absolute deviation with saturation")

errors = np.zeros(devs.size)
i_dev = np.where(devs < -.5)
errors[i_dev] = (np.abs(devs[i_dev]) - .5)**2
i_dev = np.where(devs > .5)
errors[i_dev] = (np.abs(devs[i_dev]) - .5)**2
plot_error_function(devs, errors, "Squared deviation with dead zone")

i_dev = np.where(devs > 0)
errors[i_dev] = (devs[i_dev] - .3)**3

i_dev = np.where(devs < 1.3)
errors[i_dev] = devs[i_dev] - .3

i_dev = np.where(devs < .6)
errors[i_dev] = .3

i_dev = np.where(devs < .25)
errors[i_dev] = 0

i_dev = np.where(devs < 0)
errors[i_dev] = devs[i_dev]**2

i_dev = np.where(devs < -1)
errors[i_dev] = np.sqrt(np.abs(devs[i_dev]))

plot_error_function(devs, errors, "Custom error function")
