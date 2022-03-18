"""Core script for first order surface scattering calculation module, 
using Geometric Optics Approximation. All the second order KA functions were developed based on the equations
from RADIO SCIENCE, VOL. 46, RS0E20, 2011"""

import numpy as np

from .goa import *


def sigma_2O(wave_vectors, p_slope, amplitudes, shadow):
    # Create integration variables
    TH_l, PH_l, TH_m, PH_l = np.meshgrid(
        np.linspace(1e-5, 90, 30) * np.pi / 180,
        np.linspace(0, 360, 30) * np.pi / 180,
        np.linspace(1e-5, 90, 30) * np.pi / 180,
        np.linspace(0, 360, 30) * np.pi / 180,
    )

    # Unpack vectors
    k_ix, k_iy, k_iz, k = wave_vectors["incident"]
    k_x, k_y, k_z = wave_vectors["reflected"]

    # Unpack Amplitudes
    f_hh, f_hv = amplitudes["horizontal"]
    f_vv, f_vh = amplitudes["vertical"]

    # Shadowing function
    S = 1 if shadow is None else shadow

    # Scattering Cross Section
    sigma = {
        f"{pol}": -k * np.pi * abs(f) ** 2 * p_slope * S / (k_z - k_iz) ** 2 / k_iz
        for pol, f in zip(["hh", "hv", "vv", "vh"], [f_hh, f_hv, f_vv, f_vh])
    }

    return sigma
