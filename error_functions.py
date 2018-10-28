#! /usr/bin/python3
"""
Functions that convert an array of deviations (differences between data and
model-predicted values) into a single error value.
"""
import numpy as np


def rms(devs: np.ndarray) -> float:
    """
    Find the root-mean-square of deviations.
    """
    if devs.size == 0:
        return np.nan
    return np.sqrt(np.mean(devs ** 2))


def mae(devs: np.ndarray) -> float:
    """
    Find the mean absolute value of deviations.
    """
    if devs.size == 0:
        return np.nan
    return np.mean(np.abs(devs))


def maxd(devs: np.ndarray) -> float:
    """
    Find the maximum of the magnitude of deviations.
    """
    if devs.size == 0:
        return np.nan
    return np.max(np.abs(devs))


def smr(devs: np.ndarray) -> float:
    """
    Find the square-mean-root of deviations.
    """
    if devs.size == 0:
        return np.nan
    return (np.mean(np.sqrt(np.abs(devs)))) ** 2


def test():
    """
    Check whether the code is doing what it should.
    """
    # Arbitrary case 1.
    devs = np.array([1, 2, 3, 4, 5])
    assert rms(devs) == np.sqrt(11)
    assert mae(devs) == 3
    assert maxd(devs) == 5
    assert smr(devs) == (np.mean(np.sqrt(np.abs(devs)))) ** 2

    # Arbitrary case 2.
    devs = np.array([1, -2.5, 3.2, -4.04, 5.9])
    assert rms(devs) == np.sqrt(np.mean(devs ** 2))
    assert mae(devs) == np.mean(np.abs(devs))
    assert maxd(devs) == np.max(np.abs(devs))
    assert smr(devs) == (np.mean(np.sqrt(np.abs(devs)))) ** 2

    # All-zero case.
    devs = np.array([0, 0, 0])
    assert rms(devs) == 0
    assert mae(devs) == 0
    assert maxd(devs) == 0
    assert smr(devs) == 0

    # All-ones case.
    devs = np.array([1, 1, 1])
    assert rms(devs) == 1
    assert mae(devs) == 1
    assert maxd(devs) == 1
    assert smr(devs) == 1

    # Empty case.
    devs = np.array([])
    assert np.isnan(rms(devs))
    assert np.isnan(mae(devs))
    assert np.isnan(maxd(devs))
    assert np.isnan(smr(devs))


if __name__ == "__main__":
    test()
