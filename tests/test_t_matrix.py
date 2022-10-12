from ast import In
import numpy as np

from numpy.testing import assert_allclose

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

INCIDENCE_PARAMETERS = {
    "lambda_" : .245,
    "phi_inc" : 0,
    "theta_inc" : 38.5*np.pi/180
}

@pytest.mark.parametrize(
    "surf_args, surf_kwargs, incidence",
    [
        pytest.param(LOW_SURF_ARGS, LOW_SURF_KWARGS, INCIDENCE_PARAMETERS),
        pytest.param(HIGH_SURF_ARGS, HIGH_SURF_KWARGS, INCIDENCE_PARAMETERS)
    ]
)
def test_mueller_matrix(surf_args, surf_kwargs, incidence):

    surf = t_matrix.SpmSurface(*surf_args, **surf_kwargs)
    M = surf.mueller_matrix(**incidence)

    assert_allclose(np.sum(M**2), 4 * M[0,0]**2)
