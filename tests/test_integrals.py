import numpy as np

from numpy.testing import assert_allclose

import pytest

from deep_scattering_models.geometric_optics_approx import integrals

THETA, PHI = np.meshgrid(
    np.linspace(1e-5, 90, 30) * np.pi / 180, np.linspace(0, 360, 30) * np.pi / 180
)


@pytest.mark.parametrize("theta, phi, expected", [(THETA, PHI, np.pi ** 2 / 2)])
def test_trapezoid_fix_value(theta, phi, expected):
    # Test with known result
    integrand = np.sin(THETA) ** 2
    assert_allclose(integrals.trapezoid_2d(integrand), expected, rtol=1e-6)


@pytest.mark.parametrize("theta, phi, limits", [(THETA, PHI, (0, np.pi / 2))])
def test_trapezoid_primitive(theta, phi, limits):
    # Define function and primitive
    f = lambda x: np.sin(x) * x
    F = lambda x: np.sin(x) - x * np.cos(x)

    # Unpack theta limits of integration
    a, b = limits

    # Expected integral result
    expected = 2 * np.pi * (F(b) - F(a))

    assert_allclose(integrals.trapezoid_2d(f(THETA)), expected, rtol=1e-3)
