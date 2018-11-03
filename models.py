#! /usr/bin/python3

import numpy as np


class AbstractModel(object):
    """
    An abstract base model, consisting of a function with m float parameters,
    which makes float predictions.
    """
    def __init__(self):
        self.n_params = 0
        self.name = ""
        self.n_iter_default = 3

    def initial_guess(self, x=None, y=None):
        return np.random.normal(size=self.n_params)

    def evaluate(self, p, x):
        return 0


class Polynomial(AbstractModel):
    """
    y = p0 + p1 * x + p2 * x^2 + p3 * x^3 + ...
    where the highest power of x is the order of the polynomial.
    """
    def __init__(self, order=1):
        self.order = int(order)
        self.n_params = self.order + 1
        names = {
            1: "Linear",
            2: "Quadratic",
            3: "Cubic",
            4: "Quartic",
            5: "Quintic",
            6: "Sextic",
            7: "Septic",
            8: "Octic",
        }
        if names.get(self.order):
            self.name = names.get(self.order) + " model"
        else:
            self.name = str(self.order) + "-order polynomial model"
        self.n_iter_default = self.order * 10

    def initial_guess(self, x, y):
        params = [np.random.normal(
            loc=np.mean(y),
            scale=np.std(y))]
        for i_term in range(self.order):
            params.append(np.random.normal(
                loc=0,
                scale=np.std(y) / np.std(x)**(i_term + 1)))
        return np.array(params)

    def evaluate(self, p, x):
        assert len(p) == self.n_params

        y = p[0]
        for i_term in range(self.order):
            y += p[i_term + 1] * x ** (i_term + 1)
        return y


class Line(Polynomial):
    """
    y = p0 + p1 * x
    """
    def __init__(self):
        super().__init__(order=1)


class Quadratic(Polynomial):
    """
    y = p0 + p1 * x + p2 * x - ^2
    """
    def __init__(self):
        super().__init__(order=2)


class Cubic(Polynomial):
    """
    y = p0 + p1 * x + p2 * x^2 + p3 * x^3
    """
    def __init__(self):
        super().__init__(order=3)


class Quartic(Polynomial):
    """
    y = p0 + p1 * x + p2 * x^2 + p3 * x^3 + p4 * x^4
    """
    def __init__(self):
        super().__init__(order=4)


class Exponential(AbstractModel):
    """
    y = p0 + exp(p2 * (x - p1))
    """
    def __init__(self):
        self.n_params = 3
        self.name = "Exponential model"
        self.n_iter_default = 10

    def initial_guess(self, x, y):
        p_0 = np.random.normal(loc=np.mean(y), scale=np.sqrt(np.var(y)))
        p_1 = np.random.normal(loc=np.mean(x), scale=np.sqrt(np.var(x)))
        p_2 = np.random.normal()
        return np.array([p_0, p_1, p_2])

    def evaluate(self, p, x):
        assert len(p) == self.n_params
        return p[0] + np.exp(p[2] * (x - p[1]))


all_models = [
    Line(),
    Quadratic(),
    Cubic(),
    Quartic(),
    Polynomial(order=5),
    Polynomial(order=6),
    Polynomial(order=7),
    Polynomial(order=8),
    Polynomial(order=9),
    Polynomial(order=10),
    Exponential(),
]
