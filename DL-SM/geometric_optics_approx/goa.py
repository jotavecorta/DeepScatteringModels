"""Core script for first order surface scattering calculation module, 
using Geometric Optics Approximation. All the functions were developed 
based on the equations from RADIO SCIENCE, VOL. 46, RS0E20, 2011"""

import numpy as np

def wave_vectors(lambda_inc, theta_inc, phi_inc, theta, phi, epsilon):
    # Incident Wave
    k = 2*np.pi/lambda_inc
    k_ix = k*np.sin(theta_inc)*np.cos(phi_inc)
    k_iy = k*np.sin(theta_inc)*np.sin(phi_inc)
    k_iz = k*np.cos(theta_inc)
    
    # Scatter wave
    k_x = k*np.sin(theta)*np.cos(phi)
    k_y = k*np.sin(theta)*np.sin(phi)
    k_z = k*np.sqrt(epsilon - np.sin(theta)**2)

    # Pack vectors
    vectors = {'scatter': (k_x, k_y, k_z), 
               'incident': (k_ix, k_iy, k_iz, k)} 

    return vectors     

def slopes(wave_vectors):
    # Unpack vectors
    k_ix, k_iy, k_iz, k = wave_vectors['incident']
    k_x, k_y, k_z = wave_vectors['scatter']

    # Difference Wave Vector components
    k_dx, k_dy, k_dz = k_x - k_ix, k_y - k_iy, k_z - k_iz 

    return -k_dx/k_dz, -k_dy/k_dz           

def slopes_prob_density(wave_vectors, rms_high, corr_len):
    # Unpack MSP slopes
    gamma_x, gamma_y = slopes(wave_vectors)

    # Calculate variance
    sigma_sqr = 2*rms_high**2/corr_len**2
    
    return np.exp(-(gamma_x**2 + gamma_y**2)/sigma_sqr)/sigma_sqr/np.pi


def local_fresnel_coefficients(theta_inc, phi_inc, epsilon1):
    
    pass

def kirchhoff_amplitudes(wave_vectors, fresnel_coeff):
    pass

def four_fold_integration(theta_i, wave_vectors, slope_pdf, scatter_amplitudes):
    pass

def sigma():
    pass