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


class Line(AbstractModel):
    """
    y = p0 + p1 * x
    """
    def __init__(self):
        super().__init__()
        self.n_params = 2
        self.name = "Linear model"

    def evaluate(self, p, x):
        assert len(p) == self.n_params
        return p[0] + p[1] * x


class Quadratic(AbstractModel):
    """
    y = p0 + p1 * x + p2 * x - ^2
    rearranged to be
    y = p0' + p1' * (x - p2')^2
    """
    def __init__(self):
        super().__init__()
        self.n_params = 3
        self.name = "Quadratic model"

    def evaluate(self, p, x):
        assert len(p) == self.n_params
        return p[0] + p[1] * (x - p[2])**2


class Cubic(AbstractModel):
    """
    y = p0 + p1 * x + p2 * x^2 + p3 * x^3
    """
    def __init__(self):
        self.n_params = 4
        self.name = "Cubic model"
        self.n_iter_default = 10

    def initial_guess(self, x, y):
        p_0 = np.random.normal(
            loc=np.mean(y),
            scale=np.std(y))
        p_1 = np.random.normal(
            loc=0,
            scale=np.std(y) / np.std(x))
        p_2 = np.random.normal(
            loc=0,
            scale=np.std(y) / np.std(x)**2)
        p_3 = np.random.normal(
            loc=0,
            scale=np.std(y) / np.std(x)**3)
        return np.array([p_0, p_1, p_2, p_3])

    def evaluate(self, p, x):
        assert len(p) == self.n_params
        return p[0] + p[1] * x + p[2] * x**2 + p[3] * x**3


class Quartic(AbstractModel):
    """
    y = p0 + p1 * x + p2 * x^2 + p3 * x^3 + p4 * x^4
    """
    def __init__(self):
        self.n_params = 5
        self.name = "Quartic model"
        self.n_iter_default = 20

    def initial_guess(self, x, y):
        p_0 = np.random.normal(
            loc=np.mean(y),
            scale=np.std(y))
        p_1 = np.random.normal(
            loc=0,
            scale=np.std(y) / np.std(x))
        p_2 = np.random.normal(
            loc=0,
            scale=np.std(y) / np.std(x)**2)
        p_3 = np.random.normal(
            loc=0,
            scale=np.std(y) / np.std(x)**3)
        p_4 = np.random.normal(
            loc=0,
            scale=np.std(y) / np.std(x)**4)
        return np.array([p_0, p_1, p_2, p_3, p_4])

    def evaluate(self, p, x):
        assert len(p) == self.n_params
        return p[0] + p[1] * x + p[2] * x**2 + p[3] * x**3 + p[4] * x**4


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
    Exponential(),
]
