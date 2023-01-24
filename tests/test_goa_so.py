import os
import pathlib

import numpy as np

from numpy.testing import assert_allclose

import pytest

from src.geometric_optics_approx import goa_so

THETA, PHI = np.meshgrid(
    np.linspace(1e-5, 90, 30) * np.pi / 180, np.linspace(0, 360, 30) * np.pi / 180
)

TEST_PARAMETERS = [
    (0.25, 30 * np.pi / 180, 0, 30 * np.pi / 180, 30 * np.pi / 180, 9, 16, np.ndarray),
    (0.5, 45 * np.pi / 180, 40 * np.pi / 180, THETA, PHI, 9, 4, np.ndarray),
]


@pytest.mark.parametrize(
    "lambda_inc, theta_inc, phi_inc, theta, phi, epsilon_1, epsilon_2, expected",
    TEST_PARAMETERS,
    ids=["scalar", "vectorized"],
)
def test_ray_trace():
    pass
