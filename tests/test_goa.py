import os
import pathlib

import numpy

from numpy.testing import assert_allclose

import pytest

from DLSM.geometric_optics_approx import goa

@pytest.fixture(scope=module, param=PARAMETERS_LIST)
def sigma_transmited(request):
    parameters = param.request
    def sigma_t(*parameters):
        pass
    return sigma_t

