from ast import In
import numpy as np

from numpy.testing import assert_allclose, assert_array_less

import pytest

from deep_scattering_models.small_perturbation_method import t_matrix

LOW_SURF_ARGS = [.004, 6*.004, 3]

LOW_SURF_KWARGS = {
        'two_layer' : True,
        'epsilon_2' : 9, 
        'rms_high_2' : .004, 
        'corr_length_2' : 6*.004,
        'distance' : .1,
}

HIGH_SURF_ARGS = [.012, 6*.012, 35]

HIGH_SURF_KWARGS = {
        'two_layer' : True,
        'epsilon_2' : 38, 
        'rms_high_2' : .012, 
        'corr_length_2' : 6*.012,
        'distance' : .7,
}

ONE_SURF_ARGS = [.008, 6*.008, 20]

INCIDENCE_PARAMETERS = {
    "lambda_" : .245,
    "phi_inc" : 0,
    "theta_inc" : 38.5*np.pi/180
}


@pytest.mark.parametrize(
    "surf_args, surf_kwargs, incidence",
    [
        pytest.param(LOW_SURF_ARGS, LOW_SURF_KWARGS,
                     INCIDENCE_PARAMETERS, id="Borde inferior"),
        pytest.param(HIGH_SURF_ARGS, HIGH_SURF_KWARGS,
                     INCIDENCE_PARAMETERS, id="Borde Superior"),
        pytest.param(ONE_SURF_ARGS, dict(), INCIDENCE_PARAMETERS,
                     id="Una superficie sola")
    ]
)
def test_mueller_matrix_fry_kattawar(surf_args, surf_kwargs, incidence):

    surf = t_matrix.SpmSurface(*surf_args, **surf_kwargs)
    M = surf.mueller_matrix(**incidence)

    assert_allclose(np.sum(M**2), 4 * M[0,0]**2, atol=5e-4)


# @pytest.mark.parametrize(
#     "surf_args, surf_kwargs, incidence",
#     [
#         pytest.param(LOW_SURF_ARGS, LOW_SURF_KWARGS,
#                      INCIDENCE_PARAMETERS, id="Borde inferior"),
#         pytest.param(HIGH_SURF_ARGS, HIGH_SURF_KWARGS,
#                      INCIDENCE_PARAMETERS, id="Borde Superior"),
#         pytest.param(ONE_SURF_ARGS, dict(), INCIDENCE_PARAMETERS,
#                      id="Una superficie sola")
#     ]
# )
# def test_muller_matrix_det(surf_args, surf_kwargs, incidence):

#     surf = t_matrix.SpmSurface(*surf_args, **surf_kwargs)
#     M = surf.mueller_matrix(**incidence)

#     assert_allclose(
#         np.linalg.det(M)**2,
#         M[0][0]**2 - M[1][0]**2 - M[2][0]**2 - M[3][0]**2
#     )

@pytest.mark.parametrize(
    "surf_args, surf_kwargs, incidence",
    [
        pytest.param(LOW_SURF_ARGS, LOW_SURF_KWARGS,
                     INCIDENCE_PARAMETERS, id="Borde inferior"),
        pytest.param(HIGH_SURF_ARGS, HIGH_SURF_KWARGS,
                     INCIDENCE_PARAMETERS, id="Borde Superior"),
        pytest.param(ONE_SURF_ARGS, dict(), INCIDENCE_PARAMETERS,
                     id="Una superficie sola")
    ]
)
def test_t_matrix_definite_positive(surf_args, surf_kwargs, incidence):

    surf = t_matrix.SpmSurface(*surf_args, **surf_kwargs)
    T = surf.t_matrix(**incidence)

    assert_allclose(np.zeros((3,)), np.imag(np.diagonal(T)), atol=1e-5)


@pytest.mark.parametrize(
    "surf_args, surf_kwargs, incidence",
    [
        pytest.param(LOW_SURF_ARGS, LOW_SURF_KWARGS,
                     INCIDENCE_PARAMETERS, id="Borde inferior"),
        pytest.param(HIGH_SURF_ARGS, HIGH_SURF_KWARGS,
                     INCIDENCE_PARAMETERS, id="Borde Superior"),
        pytest.param(ONE_SURF_ARGS, dict(), INCIDENCE_PARAMETERS,
                     id="Una superficie sola")
    ]
)
def test_t_matrix_hermitic(surf_args, surf_kwargs, incidence):

    surf = t_matrix.SpmSurface(*surf_args, **surf_kwargs)
    T = surf.t_matrix(**incidence)
    T = T - 1j*np.diag(np.imag(np.diagonal(T)))

    assert_allclose(T, T.conj().T)
