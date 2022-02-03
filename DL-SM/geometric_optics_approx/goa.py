"""Core script for first order surface scattering calculation module, 
using Geometric Optics Approximation. All the functions were developed 
based on the equations from RADIO SCIENCE, VOL. 46, RS0E20, 2011"""

import numpy as np

def wave_vectors(lambda_inc, theta_inc, phi_inc, theta, phi, epsilon):
    # Incident Wave
    k = 2*np.pi/lambda_inc
    k_ix = k*np.sin(theta_inc)*np.cos(phi_inc)
    k_iy = k*np.sin(theta_inc)*np.sin(phi_inc)
    k_iz = -k*np.cos(theta_inc)
    
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


def local_fresnel_coefficients(wave_vectors, epsilon):
    # Unpack incident vectors
    k_ix, k_iy, k_iz, k = wave_vectors['incident']

    # Surface slopes on MSP
    gamma_x, gamma_y = slopes(wave_vectors)

    # Normal Vector module
    n_mod = np.sqrt(1 + gamma_x**2 + gamma_y**2) 

    # Cos and squared Sin of local angle of incidence
    ctheta_li = (gamma_x*k_ix + gamma_y*k_iy + k_iz)/(k*n_mod)
    stheta_li = 1 - ctheta_li**2

    # Fresnel coefficients
    Rh = (ctheta_li - np.sqrt(epsilon - stheta_li)) / \
        (ctheta_li + np.sqrt(epsilon - stheta_li))  
    Rv = (epsilon*ctheta_li - np.sqrt(epsilon - stheta_li)) / \
        (epsilon*ctheta_li + np.sqrt(epsilon - stheta_li))    
    
    return {'horizontal': Rh, 'vertical': Rv}

def local_polarization_vectors(wave_vectors):
    # Unpack wave vectors
    k_ix, k_iy, k_iz, k = wave_vectors['incident']

    # Surface slopes on MSP
    gamma_x, gamma_y = slopes(wave_vectors)

    # Surface normal unitary vector
    n_mod = np.sqrt(1 + gamma_x**2 + gamma_y**2) 
    nx, ny, nz = -gamma_x/n_mod, -gamma_y/n_mod, 1/n_mod

    # Local perpendicular vector
    qx = k_iy/k * nz - k_iz/k * ny 
    qy = - k_ix/k * nz + k_iz/k * nx
    qz = k_ix/k * ny - k_iy/k * nx
    q_mod = np.sqrt(qx**2 + qy**2 + qz**2)

    # Local parallel vector
    px = k_iz/k * qy/q_mod - k_iy/k * qz/q_mod 
    py = - k_iz/k * qx/q_mod + k_ix/k * qz/q_mod 
    pz = k_iy/k * qx/q_mod - k_ix/k * qy/q_mod 

    return {'normal': (nx, ny, nz),
            'parallel': (px, py, pz),
            'perpendicular': (qx/q_mod, qy/q_mod, qz/q_mod)}

def global_polarization_vectors(theta_inc, phi_inc, theta, phi):   
    # Incident vertical polarization
    v_ix = -np.cos(theta_inc)*np.cos(phi_inc)
    v_iy = -np.cos(theta_inc)*np.sin(phi_inc)
    v_iz = -np.sin(theta_inc)

    # Incident horizantal polarization
    h_ix = - np.sin(phi_inc)
    h_iy = np.cos(phi_inc)
    h_iz = 0

    incident_pol = {'horizonal': (h_ix, h_iy, h_iz), 'vertical': (v_ix, v_iy, v_iz)}

    # Scattered vertical polarization
    v_sx = -np.cos(theta)*np.cos(phi)
    v_sy = -np.cos(theta)*np.sin(phi)
    v_sz = -np.sin(theta)

    # Scattered horizantal polarization
    h_sx = - np.sin(phi)
    h_sy = np.cos(phi)
    h_sz = 0

    scattered_pol = {'horizonal': (h_sx, h_sy, h_sz), 'vertical': (v_sx, v_sy, v_sz)}    

    return incident_pol, scattered_pol
             

def kirchhoff_amplitudes(wave_vectors, polarization, fresnel_coeff):
    pass

def four_fold_integration(theta_i, wave_vectors, slope_pdf, scatter_amplitudes):
    pass

def sigma():
    pass