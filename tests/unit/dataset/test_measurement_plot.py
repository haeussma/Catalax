"""Tests for the y-range of Measurement.plot: it must show everything drawn, not only the data."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from catalax.dataset.measurement import Measurement


def _measurement(values: list[float]) -> Measurement:
    return Measurement(
        initial_conditions={"P": 0.0},
        time=[0.0, 1.0, 2.0],
        data={"P": values},
    )


def test_y_range_includes_model_above_the_data():
    data = _measurement([0.0, 0.5, 1.0])
    model = _measurement([0.0, 2.5, 5.0])
    _, ax = plt.subplots()

    data.plot(show=False, ax=ax, model_data=model)

    bottom, top = ax.get_ylim()
    assert bottom == 0.0
    assert top >= 5.0
    plt.close("all")


def test_y_range_includes_negative_values_without_inverting():
    data = _measurement([-0.008, -0.006, -0.007])
    _, ax = plt.subplots()

    data.plot(show=False, ax=ax)

    bottom, top = ax.get_ylim()
    assert bottom <= -0.008
    assert top >= 0.0
    assert bottom < top
    plt.close("all")


@pytest.mark.parametrize("values", [[0.0, 0.0, 0.0], [float("nan")] * 3])
def test_y_range_stays_valid_without_spread(values):
    data = _measurement(values)
    _, ax = plt.subplots()

    data.plot(show=False, ax=ax)

    bottom, top = ax.get_ylim()
    assert bottom < top
    plt.close("all")
