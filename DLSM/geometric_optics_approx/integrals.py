"""Usefull numeric multiple integral methods. All methods te based 
on the equations of H. Gould, J. Tobochnik, W. Christian; 'An Introduction
to Computer Simulation Methods: Aplications to physical systems.', 
3rd edition, chapter 11."""

import numpy as np


def trapezoid_2d(integrand, th_lim=(0, np.pi / 2), ph_lim=(0, 2 * np.pi)):
    # Integration Deltas
    N_ph, N_th = integrand.shape
    dS = (ph_lim[1] - ph_lim[0]) / (N_ph - 1) * (th_lim[1] - th_lim[0]) / (N_th - 1)

    # Integral via 2D-trapezoid method
    borders = 3 / 4 * (
        integrand[0, 0] + integrand[0, -1] + integrand[-1, -1] + integrand[-1, 0]
    ) + 1 / 2 * (
        sum(integrand[1:-1, 0])
        + sum(integrand[1:-1, -1])
        + sum(integrand[0, 1:-1])
        + sum(integrand[-1, 1:-1])
    )

    integral = dS * (np.sum(np.sum(integrand)) - borders)

    return integral


def four_fold_integration():
    pass
