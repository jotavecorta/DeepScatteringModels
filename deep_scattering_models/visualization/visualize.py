"""Module with functions to plot data and model results."""
from matplotlib import pyplot as plt
import numpy as np


def plot_polarization_signature(sigma, ax=None, plot_kw=None):
    """Returns axes with a colormesh plot of surface polarization signature.
    
    """
    # Set default values to kwargs 
    ax = plt.gca() if ax is None else ax

    plot_kw = {} if plot_kw is None else plot_kw

    # Grid Size
    n_chi, n_psi = sigma.shape

    # Orientation and ellipticity angles
    psi = np.linspace(0, 180, n_psi) 
    chi = np.linspace(-45, 45, n_chi) 

    PSI, CHI = np.meshgrid(psi, chi)

    # Plot polarization signature
    c = ax.pcolormesh(PSI, CHI, sigma, shading = 'gouraud', **plot_kw)
    
    # Axes Settings             
    ax.set_title(r'$\sigma(\psi,\chi)$', fontsize = 20)
    
    ax.set_xlabel(r"Orientation Angle $\psi$ [°]", fontsize = 14)
    ax.set_ylabel(r"Ellipticity Angle $\chi$ [°]", fontsize = 14)

    ax.set_xticks(np.arange(0, 181, 30))
    ax.set_yticks(np.arange(-45, 46, 15))

    # Figure Settings
    fig = ax.get_figure()
    fig.colorbar(c, ax=ax)

    fig.set_figheight(8)
    fig.set_figwidth(10)

    return ax

def plot_histogram(data, n_bins=1000, ax=None, plot_kw=None):
    """Returns axes with a histogram plot of polarization 
    signatures.
    """    
    # Set default values to kwargs 
    ax = plt.gca() if ax is None else ax

    plot_kw = {} if plot_kw is None else plot_kw
    
    # Inicializo la cuenta de cada valor y el número de bins
    count = np.zeros(n_bins)

    # Guardo la cuenta de cada valor para cada firma
    for signature in data:
        hist, bins = np.histogram(signature, bins=n_bins)
        count += hist

    # Ploteo el resultado
    fig = ax.get_figure() 
    ax.plot(bins[:-1], count, **plot_kw) 
    ax.set_xlabel(r"$\sigma(\psi, \chi)$")  
    ax.set_ylabel("Count")
    ax.set_title("Polarization Signature values distribution") 

    fig.set_figheight(8)
    fig.set_figwidth(8)

    return ax
