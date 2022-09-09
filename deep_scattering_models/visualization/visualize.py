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

def plot_histogram(data, n_bins=100, ax=None, hist_kw=None, plot_kw=None):
    """Returns axes with a histogram plot of polarization 
    signatures.
    """    
    # Set default values to kwargs 
    ax = plt.gca() if ax is None else ax

    hist_kw = {} if hist_kw is None else hist_kw
    plot_kw = {} if plot_kw is None else plot_kw
    
    # Inicializo la cuenta de cada valor y el número de bins
    count = np.zeros(n_bins)

    # Store counts of each signature histogram
    for signature in data:
        hist, bins = np.histogram(signature, bins=n_bins, **hist_kw)
        count += hist

    # Plot value counts
    fig = ax.get_figure() 
    ax.plot(bins[:-1], count, **plot_kw) 
    ax.set_xlabel(r"$\sigma(\psi, \chi)$", fontsize=12)  
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Polarization Signature values distribution", fontsize=18) 

    fig.set_figheight(8)
    fig.set_figwidth(8)

    return ax

def plot_history(history, metric="mean_absolute_error", ax=None, plot_kw=None):
    """Returns axes with a line plot of training history.
    """    
    # Set default values to kwargs 
    ax = plt.gca() if ax is None else ax

    plot_kw = {} if plot_kw is None else plot_kw    

    # Unpack train and test metrics and scores
    train_metric = metric
    test_metric = f"val_{metric}"

    train_score = history.history[train_metric]
    test_score = history.history[test_metric]

    # Plot scores in ax
    ax.plot(train_score, label='Train Set', **plot_kw)
    ax.plot(test_score, label='Test Set', **plot_kw)

    ax.set_xlabel("Epochs", fontsize=12)
    ax.set_ylabel("".join(metric.split("_")).capitalize, fontsize=12)
    ax.set_title("Model train history", fontsize=15)
    ax.legend()

    fig = ax.get_figure()
    fig.set_figheight(4)
    fig.set_figwidth(10)
    fig.tigth_layout()

    return ax

    

