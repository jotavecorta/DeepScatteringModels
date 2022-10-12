import os
import pathlib

import numpy as np

from numpy.testing import assert_allclose

import pytest

from deep_scattering_models.geometric_optics_approx import goa


THETA, PHI = np.meshgrid(
    np.linspace(1e-5, 90, 30) * np.pi / 180, np.linspace(0, 360, 30) * np.pi / 180
)

TEST_PARAMETERS = [
    (0.25, 30 * np.pi / 180, 0, 30 * np.pi / 180, 30 * np.pi / 180, 9, float),
    (0.5, 45 * np.pi / 180, 40 * np.pi / 180, THETA, PHI, 3, np.ndarray),
]


@pytest.mark.parametrize(
    "lambda_inc, theta_inc, phi_inc, theta, phi, epsilon, expected",
    TEST_PARAMETERS,
    ids=["scalar", "vectorized"],
)
def test_vectors(lambda_inc, theta_inc, phi_inc, theta, phi, epsilon, expected):
    vectors = goa.wave_vectors(lambda_inc, theta_inc, phi_inc, theta, phi, epsilon)

    # Unpack vectors
    k_ix, k_iy, k_iz, k = vectors["incident"]
    k_x, k_y, k_z = vectors["reflected"]

    # Properties assertations: check equal norm
    assert_allclose(np.sqrt(k_ix ** 2 + k_iy ** 2 + k_iz ** 2), k)
    assert_allclose(np.sqrt(k_x ** 2 + k_y ** 2 + k_z ** 2), k)

    assert isinstance(k, float)
    assert all([isinstance(k_i, float) for k_i in (k_ix, k_iy, k_iz)])
    assert all([isinstance(k_s, expected) for k_s in (k_x, k_y, k_z)])
    assert all(
        [
            k_s.shape == theta.shape
            for k_s in (k_x, k_y, k_z)
            if isinstance(k_s, np.ndarray)
        ]
    )


@pytest.mark.parametrize(
    "lambda_inc, theta_inc, phi_inc, theta, phi, epsilon, expected",
    TEST_PARAMETERS,
    ids=["scalar", "vectorized"],
)
def test_slopes(lambda_inc, theta_inc, phi_inc, theta, phi, epsilon, expected):
    vectors = goa.wave_vectors(lambda_inc, theta_inc, phi_inc, theta, phi, epsilon)

    # Calculate slopes
    gamma_x, gamma_y = goa.slopes(vectors)["reflected"]

    assert all([isinstance(gamma, expected) for gamma in (gamma_x, gamma_y)])
    assert all(
        [
            gamma.shape == theta.shape
            for gamma in (gamma_x, gamma_y)
            if isinstance(gamma, np.ndarray)
        ]
    )


@pytest.mark.parametrize(
    "lambda_inc, theta_inc, phi_inc, theta, phi, epsilon, expected",
    TEST_PARAMETERS,
    ids=["scalar", "vectorized"],
)
def test_fresnell(lambda_inc, theta_inc, phi_inc, theta, phi, epsilon, expected):
    vectors = goa.wave_vectors(lambda_inc, theta_inc, phi_inc, theta, phi, epsilon)

    # Unpack incident vectors
    k_ix, k_iy, k_iz, k = vectors["incident"]

    # Surface slopes on MSP
    gamma_x, gamma_y = goa.slopes(vectors)["reflected"]

    # Normal Vector module
    n_mod = np.sqrt(1 + gamma_x ** 2 + gamma_y ** 2)

    # Cos and squared Sin of local angle of incidence
    ctheta_li = (gamma_x * k_ix + gamma_y * k_iy + k_iz) / (k * n_mod)
    stheta_li = 1 - ctheta_li ** 2

    # Calculate transmited local coefficients
    t = {
        "horizontal": 2 * ctheta_li / (ctheta_li + np.sqrt(epsilon - stheta_li)),
        "vertical": 2
        * ctheta_li
        / (np.sqrt(epsilon) * ctheta_li + np.sqrt(epsilon - stheta_li)),
    }

    # Calculate coefficients
    fresnel_coef = goa.local_fresnel_coefficients(vectors, epsilon)

    # Local conservation of energy
    assert_allclose(-fresnel_coef["horizontal"] + t["horizontal"], 1)
    assert_allclose(-fresnel_coef["vertical"] + np.sqrt(epsilon) * t["vertical"], 1)

    assert all([isinstance(R, expected) for R in fresnel_coef.values()])
    assert all(
        [
            R.shape == theta.shape
            for R in fresnel_coef.values()
            if isinstance(R, np.ndarray)
        ]
    )


@pytest.mark.parametrize(
    "lambda_inc, theta_inc, phi_inc, theta, phi, epsilon",
    [
        pytest.param(
            0.25, 0, 0, 30 * np.pi / 180, 30 * np.pi / 180, 9, id="Normal Incidence"
        ),
        pytest.param(
            0.05,
            89 * np.pi / 180,
            0,
            30 * np.pi / 180,
            30 * np.pi / 180,
            9,
            id="grazing Incidence",
        ),
    ],
)
def test_local_polarization(lambda_inc, theta_inc, phi_inc, theta, phi, epsilon):
    vectors = goa.wave_vectors(lambda_inc, theta_inc, phi_inc, theta, phi, epsilon)
    local_pol = goa.local_polarization_vectors(vectors)

    # Unpack local polarizarion and normal
    nx, ny, nz = local_pol["normal"]
    px, py, pz = local_pol["parallel"]
    qx, qy, qz = local_pol["perpendicular"]

    # Scalar products
    n_dot_q = nx * qx + ny * qy + nz * qz
    p_dot_q = px * qx + py * qy + pz * qz

    # Check perpendicular components
    assert_allclose(n_dot_q, 0, atol=1e-7)
    assert_allclose(p_dot_q, 0, atol=1e-7)

    # Check Unitary
    assert_allclose(nx ** 2 + ny ** 2 + nz ** 2, 1)
    assert_allclose(px ** 2 + py ** 2 + pz ** 2, 1)
    assert_allclose(qx ** 2 + qy ** 2 + qz ** 2, 1)


@pytest.mark.parametrize(
    "theta_inc, phi_inc, theta, phi",
    [
        pytest.param(
            45 * np.pi / 180, 0, 30 * np.pi / 180, 30 * np.pi / 180, id="Scalar"
        ),
        pytest.param(30 * np.pi / 180, 30 * np.pi / 180, THETA, PHI, id="vectorized"),
    ],
)
def test_global_polarization(theta_inc, phi_inc, theta, phi):

    # Unpack global polarization
    incident_pol, scattered_pol = goa.global_polarization_vectors(
        theta_inc, phi_inc, theta, phi
    )

    h_ix, h_iy, h_iz = incident_pol["horizontal"]
    v_ix, v_iy, v_iz = incident_pol["vertical"]

    h_x, h_y, h_z = scattered_pol["horizontal"]
    v_x, v_y, v_z = scattered_pol["vertical"]

    # Scalar products
    hi_dot_vi = h_ix * v_ix + h_iy * v_iy + h_iz * v_iz
    h_dot_v = h_x * v_x + h_y * v_y + h_z * v_z

    # Check perpendicular components
    assert_allclose(hi_dot_vi, 0, atol=1e-8)
    assert_allclose(h_dot_v, 0, atol=1e-8)

    # Check Unitary
    assert_allclose(h_x ** 2 + h_y ** 2 + h_z ** 2, 1)
    assert_allclose(v_x ** 2 + v_y ** 2 + v_z ** 2, 1)


@pytest.mark.parametrize(
    "theta_inc, phi_inc, theta, phi, transmited, theta_t, phi_t",
    [
        pytest.param(
            45 * np.pi / 180,
            0,
            30 * np.pi / 180,
            30 * np.pi / 180,
            True,
            30 * np.pi / 180,
            30 * np.pi / 180,
            id="Scalar",
        ),
        pytest.param(
            30 * np.pi / 180,
            30 * np.pi / 180,
            THETA,
            PHI,
            True,
            THETA,
            PHI,
            id="vectorized",
        ),
    ],
)
def test_transmited_polarization(
    theta_inc, phi_inc, theta, phi, transmited, theta_t, phi_t
):

    # Unpack global polarization
    incident_pol, scattered_pol, transmited_pol = goa.global_polarization_vectors(
        theta_inc,
        phi_inc,
        THETA,
        PHI,
        transmited=transmited,
        theta_t=theta_t,
        phi_t=phi_t,
    )

    h_tx, h_ty, h_tz = transmited_pol["horizontal"]
    v_tx, v_ty, v_tz = transmited_pol["vertical"]

    # Scalar products
    ht_dot_vt = h_tx * v_tx + h_ty * v_ty + h_tz * v_tz

    # Check perpendicular components
    assert_allclose(ht_dot_vt, 0, atol=1e-8)

    # Check Unitary
    assert_allclose(h_tx ** 2 + h_ty ** 2 + h_tz ** 2, 1)
    assert_allclose(v_tx ** 2 + v_ty ** 2 + v_tz ** 2, 1)
