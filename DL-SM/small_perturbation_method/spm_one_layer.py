# =============================================================================
# First order SPM for dielectric rough surface problem
# =============================================================================
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
    
    # Transmited wave
    kt = np.sqrt(epsilon)*k
    kt_z = k*np.sqrt(epsilon - np.sin(theta)**2)
    kt_iz = k*np.sqrt(epsilon - np.sin(theta_inc)**2)
    
    # Pack vectors
    vectors = {'scatter': (k_x, k_y, k_z), 
               'incident': (k_ix, k_iy, k_iz, k), 
               'transmitted': (kt_z, kt, kt_iz)} 
    
    return vectors

def auxiliar_vectors(lambda_inc, theta_inc, phi_inc, theta, phi):
    '''Return K_1 and K_2 vectors for PSD'''
    
    k_1 = 2*np.pi/lambda_inc*(np.sin(theta)*np.cos(phi) - np.sin(theta_inc)*np.cos(phi_inc))
    k_2 = 2*np.pi/lambda_inc*(np.sin(theta)*np.sin(phi) - np.sin(theta_inc)*np.sin(phi_inc))
    
    return k_1, k_2

def first_order_amplitudes(wave_vectors, phi_inc, phi):
    # Unpack wave vectors
    (k_x, k_y, k_z) = wave_vectors['scatter']
    (k_ix, k_iy, k_iz, k) = wave_vectors['incident']
    (kt_z, kt, kt_iz) = wave_vectors['transmitted']
    
    # Parallel component wave vectors
    k_rho, k_irho = np.sqrt(k_x**2 + k_y**2), np.sqrt(k_ix**2 + k_iy**2)
    
    # Trigonometric functions
    c_hh = np.cos(phi - phi_inc)
    s_hv = np.sin(phi - phi_inc)
    s_vv = -kt_z*kt_iz*c_hh + k_rho*k_irho*kt**2/k**2
    
    # Coefficients calculation
    f_hh = (kt**2 - k**2) / (kt_z + k_z) * (2*k_iz/(k_iz + kt_iz)) * c_hh
    f_hv = (kt**2 - k**2) * (kt_z*k) / (k**2*kt_z + kt**2*k_z) * (2*k_z/(k_z + kt_z)) * s_hv
    f_vv = (kt**2 - k**2) / (k**2*kt_z + kt**2*k_z) * (2*k**2*k_iz/(k**2*kt_iz + kt**2*k_iz)) * s_vv 

    return {'horizontal': f_hh, 
            'crosspol': f_hv,
            'vertical': f_vv}

def power_spectral_density(k_1, k_2, rms_high, corr_length, acf_type = 'gaussian'):
    # Rename some variables
    s, l = rms_high, corr_length
    
    # Defines psd by autocorrelation function 
    auto_corr = {
        "gaussian": np.exp(-0.25*l**2*(k_1**2 + k_2**2))/(4*np.pi),
        "exponential": 1/(2*np.pi*(1 + (k_1**2 + k_2**2)*l**2)**(3 / 2)),
        "power_law": np.exp(-np.sqrt(k_1**2 + k_2**2)*l)/(2 * np.pi),
    }
    
    return s**2*l**2*auto_corr[acf_type]

def sigma_HH(k, theta, theta_inc, f_hh, W):    
    return 4*np.pi*k**2*np.cos(theta)*W*np.abs(f_hh)**2/np.cos(theta_inc)

def sigma_HV(k, theta, theta_inc, f_hv, W):
    return 4*np.pi*k**2*np.cos(theta)/np.cos(theta_inc)*W*np.abs(f_hv)**2

def sigma_VV(k, theta, theta_inc, f_vv, W):
    return 4*np.pi*k**2*np.cos(theta)/np.cos(theta_inc)*W*np.abs(f_vv)**2

def sigma(k, theta, theta_inc, scat_amplitudes, W):
    # Unpack first order scatering amplitudes
    f_hh = scat_amplitudes['horizontal']
    f_hv = scat_amplitudes['crosspol']
    f_vv = scat_amplitudes['vertical']
    
    # Calculate biestatic coeffiecients in all tree channels
    s_HH = sigma_HH(k, theta, theta_inc, f_hh, W)
    s_HV = sigma_HV(k, theta, theta_inc, f_hv, W)
    s_VV = sigma_VV(k, theta, theta_inc, f_vv, W)
    
    return {'horizontal': s_HH, 
            'crosspol': s_HV,
            'vertical': s_VV}