"""Core script for first order surface scattering calculation module, 
using Geometric Optics Approximation. All the second order KA functions were 
developed based on the equations from N. Pinel, J. T. Johnson, C. Bourlier,
'A Geometrical Optics Model of Three Dimensional Scattering From A Rough
Layer With Two Rough Surfaces', IEEE Trans. Antennas Propag., vol 58, no. 3,
pp. 809-816, 2010. """

import numpy as np

from .goa import *


def ray_trace(
    lambda_inc, theta_inc, phi_inc, theta, phi, epsilon_1, epsilon_2, epsilon_i=1
):

    # Create integration variables
    TH_m, PH_m, TH_p, PH_p = np.meshgrid(
        np.linspace(1e-5, 90, 30) * np.pi / 180,
        np.linspace(0, 360, 30) * np.pi / 180,
        np.linspace(1e-5, 90, 30) * np.pi / 180,
        np.linspace(0, 360, 30) * np.pi / 180,
    )

    # First transmition: media 1 --> media 2 (usefull for first reflection too)
    first_order = wave_vectors(
        lambda_inc,
        theta_inc,
        phi_inc,
        theta,
        phi,
        epsilon_1,
        transmited=True,
        theta_t=TH_m,
        phi_t=PH_m,
    )

    # Reflection: interface media 2 - media 3
    second_reflection = wave_vectors(
        lambda_inc, TH_m, PH_m, TH_p, PH_p, epsilon_2, epsilon_i=epsilon_1
    )

    # Second transmition: media 2 --> media 1
    second_transmition = wave_vectors(
        lambda_inc, TH_p, PH_p, theta, phi, epsilon=epsilon_i, epsilon_i=epsilon_1
    )

    return first_order, second_reflection, second_transmition


def sigma_2O(lambda_inc, theta_inc, phi_inc, theta, phi, epsilon_1, epsilon_2):
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

    # Global Polarization
    pol_1_2 = global_polarization_vectors(
        theta_inc, phi_inc, theta, phi, transmited=True, theta_t=TH_m, phi_t=PH_m
    )
    pol_2_3 = global_polarization_vectors(TH_m, PH_m, TH_p, PH_p)
    pol_1_2 = global_polarization_vectors(
        TH_p, PH_p, 0, 0, transmited=True, theta_t=theta, phi_t=phi
    )  # Third reflection doesn´t count

    # Fresnel coeficients
    fresnel_1_2 = local_fresnel_coefficients(vectors_1_2, epsilon_1)
    fresnel_2_3 = local_fresnel_coefficients(vectors_2_3, epsilon_2)
    fresnel_2_1 = local_fresnel_coefficients(vectors_2_1, epsilon_i)

    # Reflected and Transmited Amplitudes
    amps_1_2 = transmited_amplitudes(vectors_1_2, pol_1_2, fresnel_1)
    amps_2_3 = alternative_amplitudes(vectors_2_3, pol_2_3, fresnel_2)
    amps_2_1 = transmited_amplitudes(vectors_2_1, pol_2_1, fresnel_1)

    # Shadowing efect
    S_1_2 = transmited_shadowing(TH_m, theta_i, s_1, l_1)
    S_2_3 = shadowing(TH_m, PH_m, TH_p, PH_p, s_2, l_2)
    S_2_1 = transmited_shadowing(TH_p, theta, s_1, l_1)

    # Unpack vectors
    k_ix, k_iy, k_iz, k = vectors_1_2["incident"]
    k_x, k_y, k_z = vectors_1_2["reflected"]

    k_mx, k_my, k_mz, kt = vectors_1_2["transmited"]
    k_px, k_py, k_pz = vectors_2_3["reflected"]

    k_px, k_py, k_pz = vectors_2_1["transmited"]

    # Define auxiliar function
    denominator = lambda a, b: (a - b) ** 2

    # Amplitudes product
    f_hh = (
        amps_1_2["horizontal"][0]
        * amps_2_3["horizontal"][0]
        * amps_2_1["horizontal"][0]
    )
    f_hv = (
        amps_1_2["horizontal"][1]
        * amps_2_3["horizontal"][1]
        * amps_2_1["horizontal"][1]
    )
    f_vv = amps_1_2["vertical"][0] * amps_2_3["vertical"][0] * amps_2_1["vertical"][0]
    f_vh = amps_1_2["vertical"][1] * amps_2_3["vertical"][1] * amps_2_1["vertical"][1]

    # Scattering Cross Section
    p = p_1_2["transmited"] * p_2_3["reflected"] * p_2_1["transmited"]
    S = S_1_2 * S_2_3 * S_2_1
    d = (
        denominator(k_mz / kt, k_iz / k_t)
        * denominator(k_pz / k_t, k_mz / k_t)
        * denominator(k_z / k, k_pz / k)
    )

    sigma = {
        f"{pol}": -k / k_iz * abs(f) ** 2 * p * S / d
        for pol, f in zip(["hh", "hv", "vv", "vh"], [f_hh, f_hv, f_vv, f_vh])
    }

    return sigma
