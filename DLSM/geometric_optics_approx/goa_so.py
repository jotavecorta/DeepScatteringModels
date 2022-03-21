"""Core script for first order surface scattering calculation module, 
using Geometric Optics Approximation. All the second order KA functions were developed based on the equations
from RADIO SCIENCE, VOL. 46, RS0E20, 2011"""

import numpy as np

from .goa import *


def ray_trace(lambda_inc, theta_inc, phi_inc, theta, phi, epsilon_1, epsilon_2,
              epsilon_i=1):
    # Create integration variables
    TH_m, PH_m, TH_p, PH_p = np.meshgrid(
        np.linspace(1e-5, 90, 30) * np.pi / 180,
        np.linspace(0, 360, 30) * np.pi / 180,
        np.linspace(1e-5, 90, 30) * np.pi / 180,
        np.linspace(0, 360, 30) * np.pi / 180,
    )

    # First transmition: media 1 --> media 2 (usefull for first reflection too)
    first_order = wave_vectors(lambda_inc, theta_inc, phi_inc, theta, phi, epsilon_1, 
    transmited=True, theta_t=TH_m, phi_t=PH_m)

    # Reflection: interface media 2 - media 3
    second_reflection = wave_vectors(
        lambda_inc, TH_m, PH_m, TH_p, PH_p, epsilon_2, epsilon_i=epsilon_1)

    # Second transmition: media 2 --> media 1
    second_transmition = wave_vectors(
        lambda_inc, TH_p, PH_p, theta, phi, epsilon=epsilon_i, epsilon_i=epsilon_1)

    return first_order, second_reflection, second_transmition


def sigma_2O(fi_vectors, p_slope, amplitudes, shadow):
    # Create integration variables
    TH_m, PH_m, TH_p, PH_p = np.meshgrid(
        np.linspace(1e-5, 90, 30) * np.pi / 180,
        np.linspace(0, 360, 30) * np.pi / 180,
        np.linspace(1e-5, 90, 30) * np.pi / 180,
        np.linspace(0, 360, 30) * np.pi / 180,
    )

    # Reflected and Transmited vectors (subindex indicates media interfaces)
    vectors_1_2, vectors_2_3, vectors_2_1 = ray_trace(
        lambda_inc, theta_inc, phi_inc, theta, phi, epsilon_1, epsilon_2
    )

    # Slopes pdf's 
    p_1_2 = slopes_prob_density(vectors_1_2, s_1, l_1, transmited=True)
    p_2_3 = slopes_prob_density(vectors_2_3, s_2, l_2)
    p_2_1 = slopes_prob_density(vectors_2_1, s_1, l_1, transmited=True)
    
    # Reflected and Transmited Amplitudes
    amps_1_2 = transmited_amplitudes(vectors_1_2, pol_1_2, fresnel_1)
    amps_2_3 = alternative_amplitudes(vectors_2_3, pol_2_3, fresnel_2)
    amps_2_1 = transmited_amplitudes(vectors_2_1, pol_2_1, fresnel_1)
    

    # Unpack vectors
    k_ix, k_iy, k_iz, k = wave_vectors["incident"]
    k_x, k_y, k_z = wave_vectors["reflected"]

    # Shadowing function
    S = 1 if shadow is None else shadow

    # Scattering Cross Section
    sigma = {
        f"{pol}": -k * np.pi * abs(f) ** 2 * p_slope * S / (k_z - k_iz) ** 2 / k_iz
        for pol, f in zip(["hh", "hv", "vv", "vh"], [f_hh, f_hv, f_vv, f_vh])
    }

    return sigma
