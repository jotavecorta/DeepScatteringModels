"""Script with functions to plot data and model results."""
from matplotlib import pyplot as plt
import numpy as np


def plot_polarization_signature(sigma, ax=None, plot_kw=None):
    
    # Set default values to kwargs 
    ax = plt.gca() if ax is None else ax

    plot_kw = {} if plot_kw is None else plot_kw

    # Grid Size
    n_chi, n_psi = sigma.shape

    # Orientation and ellipticity angles
    psi = np.linspace(0, 180, n_psi) * np.pi/180
    chi = np.linspace(-45, 45, n_chi) * np.pi/180

    PSI, CHI = np.meshgrid(psi, chi)

    # Plot polarization signature
    c = ax.pcolormesh(PSI, CHI, sigma, shading = 'gouraud', **plot_kw)
    
    # Plot Settings             
    ax.set_title(r'$\sigma(\psi,\chi)$', fontsize = 20)
    ax.set_xlabel(r"$\psi$", fontsize = 16)
    ax.set_ylabel(r"$\chi$", fontsize = 16)

    fig = ax.get_figure()
    fig.colorbar(c, ax=ax)

    return ax
