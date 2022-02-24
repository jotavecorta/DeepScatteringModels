"""Core script for first order surface scattering calculation module, 
using Geometric Optics Approximation. All the first order KA functions were 
developed based on the equations from L. Tsang, J. A. Kong, 'Scattering of 
Electromagnetic Waves Vol. 3: Advanced Topics', chapter 2.
All the second order KA functions were developed based on the equations
from RADIO SCIENCE, VOL. 46, RS0E20, 2011"""

import numpy as np
from scipy.special import erfc

def wave_vectors(lambda_inc, theta_inc, phi_inc, theta, phi, epsilon):
    # Incident Wave
    k = 2*np.pi/lambda_inc
    k_ix = k*np.sin(theta_inc)*np.cos(phi_inc)
    k_iy = k*np.sin(theta_inc)*np.sin(phi_inc)
    k_iz = -k*np.cos(theta_inc)
    
    # Reflected wave
    k_x = k*np.sin(theta)*np.cos(phi)
    k_y = k*np.sin(theta)*np.sin(phi)
    k_z = k*np.cos(theta)  

    # Pack vectors
    vectors = {'reflected': (k_x, k_y, k_z), 
               'incident': (k_ix, k_iy, k_iz, k)} 

    return vectors     


def transmited_vectors(theta_t, phi_t, epsilon):
    # Transmited wave
    kt = np.sqrt(epsilon)*k
    k_tx = kt*np.sin(theta_t)*np.cos(phi_t)
    k_ty = kt*np.sin(theta_t)*np.sin(phi_t)
    k_tz = -kt*np.cos(theta_t)  

    return (k_tx, k_ty, k_tz, kt)


def slopes(wave_vectors):
    # Unpack vectors
    k_ix, k_iy, k_iz, k = wave_vectors['incident']
    k_x, k_y, k_z = wave_vectors['reflected']

    # Difference Wave Vector components
    k_dx, k_dy, k_dz = k_x - k_ix, k_y - k_iy, k_z - k_iz 

    return -k_dx/k_dz, -k_dy/k_dz           


def slopes_prob_density(wave_vectors, rms_high, corr_len):
    # Unpack MSP slopes
    gamma_x, gamma_y = slopes(wave_vectors)

    # Calculate variance
    sigma_sqr = 4*rms_high**2/corr_len**2
    
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
    Rv = (np.sqrt(epsilon)*ctheta_li - np.sqrt(epsilon - stheta_li)) / \
        (np.sqrt(epsilon)*ctheta_li + np.sqrt(epsilon - stheta_li))    
    
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


def global_polarization_vectors(
    theta_inc, 
    phi_inc,
    theta, 
    phi, 
    transmited=False, 
    theta_t=None,
    phi_t=None
    ):   
    # Incident vertical polarization
    v_ix = -np.cos(theta_inc)*np.cos(phi_inc)
    v_iy = -np.cos(theta_inc)*np.sin(phi_inc)
    v_iz = -np.sin(theta_inc)

    # Incident horizantal polarization
    h_ix = - np.sin(phi_inc)
    h_iy = np.cos(phi_inc)
    h_iz = 0

    incident_pol = {'horizontal': (h_ix, h_iy, h_iz), 'vertical': (v_ix, v_iy, v_iz)}

    # Reflected vertical polarization
    v_x = np.cos(theta)*np.cos(phi)
    v_y = np.cos(theta)*np.sin(phi)
    v_z = -np.sin(theta)

    # Reflecteded horizantal polarization
    h_x = - np.sin(phi)
    h_y = np.cos(phi)
    h_z = 0

    reflected_pol = {'horizontal': (h_x, h_y, h_z), 'vertical': (v_x, v_y, v_z)} 
    
    if transmited:
        assert (theta_t is not None) and (
            phi_t is not None), 'theta_t and phi_t must have not null input' \
                                'value for transmited polarization'

        # Transmited vertical polarization
        v_tx = -np.cos(theta_t)*np.cos(phi_t)
        v_ty = -np.cos(theta_t)*np.sin(phi_t)
        v_tz = -np.sin(theta_t)

        # Transmited horizantal polarization
        h_tx = - np.sin(phi_t)
        h_ty = np.cos(phi_t)
        h_tz = 0

        transmited_pol = {'horizontal': (h_tx, h_ty, h_tz), 'vertical': (v_tx, v_ty, v_tz)}        

        return incident_pol, reflected_pol, transmited_pol
    else:    
        return incident_pol, reflected_pol     


def scattering_amplitudes(wave_vectors, polarization, fresnel_coeff):
    """N. Pinel, C. Boulier, 'Electromagnetic Wave Scattering from Random 
    Rough Surfaces: Asymptotic Models', Wiley-ISTE, 2013, page 66, eq. 2.57."""
    # Unpack vectors
    k_ix, k_iy, k_iz, k = wave_vectors['incident']
    k_x, k_y, k_z = wave_vectors['reflected']

    # Surface slopes on MSP
    gamma_x, gamma_y = slopes(wave_vectors)
    n_mod = np.sqrt(1 + gamma_x**2 + gamma_y**2)

    # Unpack global polarization
    incident_pol, scattered_pol = polarization

    h_ix, h_iy, h_iz = incident_pol['horizontal']
    v_ix, v_iy, v_iz = incident_pol['vertical']    

    h_x, h_y, h_z = scattered_pol['horizontal']
    v_x, v_y, v_z = scattered_pol['vertical']

    # Local polarization and normal
    local_pol = local_polarization_vectors(wave_vectors)
    nx, ny, nz = local_pol['normal']
    px, py, pz = local_pol['parallel']
    qx, qy, qz = local_pol['perpendicular']

    # Local Fresnel coefficients
    Rh = fresnel_coeff['horizontal']
    Rv = fresnel_coeff['vertical']

    # Horizontal incidence
    f_hh = n_mod/2 * ( 
        - (1 - Rh) * (h_ix*qx + h_iy*qy + h_iz*qz) * (nx*k_ix + ny*k_iy + nz*k_iz) * (h_x*qx + h_y*qy + h_z*qz)
        + (1 + Rv) * (h_ix*px + h_iy*py + h_iz*pz) * (h_x*(ny*qz - nz*qy) + h_y*(nz*qx - nx*qz) + h_z*(nx*qy - ny*qx))
        + (1 + Rh) * (h_ix*qx + h_iy*qy + h_iz*qz) * ((k_x*qx + k_y*qy + k_z*qz) * (h_x*nx + h_y*ny + h_z*nz) - (k_x*nx + k_y*ny + k_z*nz) * (h_x*qx + h_y*qy + h_z*qz))
        + (1 - Rv) * (h_ix*px + h_iy*py + h_iz*pz) * (nx*k_ix + ny*k_iy + nz*k_iz) * (h_x*(k_y*qz - k_z*qy) + h_y*(k_z*qx - k_x*qz) + h_z*(k_x*qy - k_y*qx))
    )

    f_hv = n_mod/2 * ( 
        - (1 - Rh) * (h_ix*qx + h_iy*qy + h_iz*qz) * (nx*k_ix + ny*k_iy + nz*k_iz) * (v_x*qx + v_y*qy + v_z*qz)
        + (1 + Rv) * (h_ix*px + h_iy*py + h_iz*pz) * (v_x*(ny*qz - nz*qy) + v_y*(nz*qx - nx*qz) + v_z*(nx*qy - ny*qx))
        + (1 + Rh) * (h_ix*qx + h_iy*qy + h_iz*qz) * ((k_x*qx + k_y*qy + k_z*qz) * (v_x*nx + v_y*ny + v_z*nz) - (k_x*nx + k_y*ny + k_z*nz) * (v_x*qx + v_y*qy + v_z*qz))
        + (1 - Rv) * (h_ix*px + h_iy*py + h_iz*pz) * (nx*k_ix + ny*k_iy + nz*k_iz) * (v_x*(k_y*qz - k_z*qy) + v_y*(k_z*qx - k_x*qz) + v_z*(k_x*qy - k_y*qx))
    )

    # Vertical incidence
    f_vh = n_mod/2 * ( 
        - (1 - Rh) * (v_ix*qx + v_iy*qy + v_iz*qz) * (nx*k_ix + ny*k_iy + nz*k_iz) * (h_x*qx + h_y*qy + h_z*qz)
        + (1 + Rv) * (v_ix*px + v_iy*py + v_iz*pz) * (h_x*(ny*qz - nz*qy) + h_y*(nz*qx - nx*qz) + h_z*(nx*qy - ny*qx))
        + (1 + Rh) * (v_ix*qx + v_iy*qy + v_iz*qz) * ((k_x*qx + k_y*qy + k_z*qz) * (h_x*nx + h_y*ny + h_z*nz) - (k_x*nx + k_y*ny + k_z*nz) * (h_x*qx + h_y*qy + h_z*qz))
        + (1 - Rv) * (v_ix*px + v_iy*py + v_iz*pz) * (nx*k_ix + ny*k_iy + nz*k_iz) * (h_x*(k_y*qz - k_z*qy) + h_y*(k_z*qx - k_x*qz) + h_z*(k_x*qy - k_y*qx))
    )

    f_vv = n_mod/2 * ( 
        - (1 - Rh) * (v_ix*qx + v_iy*qy + v_iz*qz) * (nx*k_ix + ny*k_iy + nz*k_iz) * (v_x*qx + v_y*qy + v_z*qz)
        + (1 + Rv) * (v_ix*px + v_iy*py + v_iz*pz) * (v_x*(ny*qz - nz*qy) + v_y*(nz*qx - nx*qz) + v_z*(nx*qy - ny*qx))
        + (1 + Rh) * (v_ix*qx + v_iy*qy + v_iz*qz) * ((k_x*qx + k_y*qy + k_z*qz) * (v_x*nx + v_y*ny + v_z*nz) - (k_x*nx + k_y*ny + k_z*nz) * (v_x*qx + v_y*qy + v_z*qz))
        + (1 - Rv) * (v_ix*px + v_iy*py + v_iz*pz) * (nx*k_ix + ny*k_iy + nz*k_iz) * (v_x*(k_y*qz - k_z*qy) + v_y*(k_z*qx - k_x*qz) + v_z*(k_x*qy - k_y*qx))
    )

    return {'horizontal': (f_hh, f_hv), 'vertical': (f_vv, f_vh)}


def alternative_amplitudes(wave_vectors, polarization, fresnel_coeff):
    """L. Tsang, J. A. Kong, 'Scattering of Electromagnetic Waves Vol. 3: 
    Advanced Topics', Wiley-Interscience, 2001, page 66, eqs. 2.1.122."""

    # Unpack vectors
    k_ix, k_iy, k_iz, k = wave_vectors['incident']
    k_x, k_y, k_z = wave_vectors['reflected']

    # Unpack global polarization
    incident_pol, scattered_pol = polarization

    h_ix, h_iy, h_iz = incident_pol['horizontal']
    v_ix, v_iy, v_iz = incident_pol['vertical']    

    h_x, h_y, h_z = scattered_pol['horizontal']
    v_x, v_y, v_z = scattered_pol['vertical']

    # Local Fresnel coefficients
    R_h = fresnel_coeff['horizontal']
    R_v = fresnel_coeff['vertical']

    # Prefactor
    mod_ki_x_ks = (k_iy*k_z - k_iz*k_y)**2 + (k_iz*k_x - k_ix*k_z)**2 + (k_ix*k_y - k_iy*k_x)**2 
    C = k*((k_x - k_ix)**2 + (k_y - k_iy)**2 + (k_z - k_iz)**2)/(mod_ki_x_ks*(k_z - k_iz))

    # Horizontal incidence
    f_hh = C * ((v_ix*k_x + v_iy*k_y + v_iz*k_z) * (v_x*k_ix + v_y*k_iy + v_z*k_iz) * R_h + \
                (h_ix*k_x + h_iy*k_y + h_iz*k_z) * (h_x*k_ix + h_y*k_iy + h_z*k_iz) * R_v)
    f_hv = C * ((h_ix*k_x + h_iy*k_y + h_iz*k_z) * (v_x*k_ix + v_y*k_iy + v_z*k_iz) * R_h - \
                (v_ix*k_x + v_iy*k_y + v_iz*k_z) * (h_x*k_ix + h_y*k_iy + h_z*k_iz) * R_v)

    # Vertical incidence
    f_vv = C * ((v_ix*k_x + v_iy*k_y + v_iz*k_z) * (v_x*k_ix + v_y*k_iy + v_z*k_iz) * R_v + \
                (h_ix*k_x + h_iy*k_y + h_iz*k_z) * (h_x*k_ix + h_y*k_iy + h_z*k_iz) * R_h)
    f_vh = C * (-(h_ix*k_x + h_iy*k_y + h_iz*k_z) * (v_x*k_ix + v_y*k_iy + v_z*k_iz) * R_v + \
                (v_ix*k_x + v_iy*k_y + v_iz*k_z) * (h_x*k_ix + h_y*k_iy + h_z*k_iz) * R_h)

    return {'horizontal': (f_hh, f_hv), 'vertical': (f_vv, f_vh)}


def transmited_amplitudes(wave_vectors, t_vectors, polarization, fresnel_coeff):
    """L. Tsang, J. A. Kong, 'Scattering of Electromagnetic Waves Vol. 3: 
    Advanced Topics', Wiley-Interscience, 2001, page 66, eqs. 2.1.131."""

    # Unpack vectors
    k_ix, k_iy, k_iz, k = wave_vectors['incident']
    k_tx, k_ty, k_tz, kt = t_vectors

    # Unpack global polarization
    incident_pol, _, transmited_pol = polarization

    h_ix, h_iy, h_iz = incident_pol['horizontal']
    v_ix, v_iy, v_iz = incident_pol['vertical']    

    h_tx, h_ty, h_tz = transmited_pol['horizontal']
    v_tx, v_ty, v_tz = transmited_pol['vertical']

    # Local Fresnel coefficients
    R_h = fresnel_coeff['horizontal']
    R_v = fresnel_coeff['vertical']

    # Surface slopes on MSP
    gamma_x, gamma_y = slopes(wave_vectors)

    # Surface normal unitary vector
    n_mod = np.sqrt(1 + gamma_x**2 + gamma_y**2) 
    nx, ny, nz = -gamma_x/n_mod, -gamma_y/n_mod, 1/n_mod

    # Prefactor
    mod_ki_x_kt = (k_iy*k_tz - k_iz*k_ty)**2 + (k_iz*k_tx - k_ix*k_tz)**2 + (k_ix*k_ty - k_iy*k_tx)**2 
    C = np.sqrt((k_x - k_ix)**2 + (k_y - k_iy)**2 + (k_z - k_iz)**2) * (nx*k_tx + ny*k_ty + nz*k_tz) * 2 * kt \
        / (mod_ki_x_kt*(k_tz - k_iz))

    # Horizontal incidence
    ft_hh = C * ((v_ix*k_tx + v_iy*k_ty + v_iz*k_tz) * (v_tx*k_ix + v_ty*k_iy + v_tz*k_iz) * (1 + R_h) + \
                (h_ix*k_tx + h_iy*k_ty + h_iz*k_tz) * (h_tx*k_ix + h_ty*k_iy + h_tz*k_iz) * (k/kt) *(1 + R_v))
    ft_hv = C * (-(h_ix*k_tx + h_iy*k_ty + h_iz*k_tz) * (v_tx*k_ix + v_ty*k_iy + v_tz*k_iz) * (1 + R_h) + \
                (v_ix*k_tx + v_iy*k_ty + v_iz*k_tz) * (h_tx*k_ix + h_ty*k_iy + h_tz*k_iz) * (k/kt) * (1 + R_v))

    # Vertical incidence
    ft_vv = C * ((v_ix*k_tx + v_iy*k_ty + v_iz*k_tz) * (v_tx*k_ix + v_ty*k_iy + v_tz*k_iz) * (k/kt) * (1 + R_v) + \
                (h_ix*k_tx + h_iy*k_ty + h_iz*k_tz) * (h_tx*k_ix + h_ty*k_iy + h_tz*k_iz) * (1 + R_h))
    ft_vh = C * (-(h_ix*k_tx + h_iy*k_ty + h_iz*k_tz) * (v_tx*k_ix + v_ty*k_iy + v_tz*k_iz) * (k/kt) * (1 + R_v) + \
                (v_ix*k_tx + v_iy*k_ty + v_iz*k_tz) * (h_tx*k_ix + h_ty*k_iy + h_tz*k_iz) * (1 + R_h))

    return {'horizontal': (ft_hh, ft_hv), 'vertical': (ft_vv, ft_vh)}    

# Revisar condiciones y transmitidos
def shadowing(theta_i, phi_inc, theta, phi, rms_high, corr_len):
    # Define some operations
    v_a = lambda x : corr_len*abs(1/np.tan(x))/2/rms_high
    lambda_fun = lambda x : np.exp(-v_a(x)**2)/2/v_a(x)/np.sqrt(np.pi) - erfc(v_a(x))/2
    
    # Define different shadowing functions
    S1 = 1/(1 + lambda_fun(theta_i))
    S2 = 1/(1 + lambda_fun(theta))
    S3 = 1/(1 + lambda_fun(theta_i) + lambda_fun(theta))

    # Vectorized conditional
    S = np.where((phi == phi_inc + np.pi) & (theta >= theta_i), S1, S3)
    S = np.where((phi == phi_inc + np.pi) & (theta < theta_i), S2, S)

    return S


def sigma(wave_vectors, p_slope, amplitudes, shadow):
    # Unpack vectors
    k_ix, k_iy, k_iz, k = wave_vectors['incident']
    k_x, k_y, k_z = wave_vectors['reflected']

    # Unpack Amplitudes
    f_hh, f_hv = amplitudes['horizontal']
    f_vv, f_vh = amplitudes['vertical']
 
    # Shadowing function
    S = 1 if shadow is None else shadow

    # Scattering Cross Section 
    sigma = {f'{pol}': -k*np.pi*abs(f)**2*p_slope*S/(k_z - k_iz)**2/k_iz for pol, f 
    in zip(['hh', 'hv', 'vv', 'vh'], [f_hh, f_hv, f_vv, f_vh])}

    return sigma 


def sigma_t(wave_vectors, transmited_vectors, p_slope, amplitudes, shadow):
    # Unpack vectors
    k_ix, k_iy, k_iz, k = wave_vectors['incident']
    k_tx, k_ty, k_tz, kt = transmited_vectors 

    # Unpack Amplitudes
    ft_hh, ft_hv = amplitudes['horizontal']
    ft_vv, ft_vh = amplitudes['vertical']
 
    # Shadowing function
    S = 1 if shadow is None else shadow

    # Scattering Cross Section 
    #sigma = {f'{pol}': -k**3/k_iz*abs(f)**2*p_slope*S/(k_z - k_iz)**2 for pol, f 
    #in zip(['hh', 'hv', 'vv', 'vh'], [f_hh, f_hv, f_vh, f_vv])}
    sigma_t = {f'{pol}': -kt*np.pi*abs(f)**2*p_slope*S/(k_tz - k_iz)**2/k_iz for pol, f 
    in zip(['hh', 'hv', 'vv', 'vh'], [ft_hh, ft_hv, ft_vh, ft_vv])}

    return sigma_t

# Revisar inntegral
def energy(sigma, sigma_t):
    # Initialize domain of integration
    THETA, PHI = np.meshgrid(
        np.linspace(1e-5, 89, 30) * np.pi /
        180, np.linspace(0, 360, 30) * np.pi / 180
    )

    THETA_T, PHI_T = np.meshgrid(
        np.linspace(1e-5, 89, 30) * np.pi /
        180, np.linspace(0, 360, 30) * np.pi / 180
    )

    # Horizontally refracted wave
    r_h = sum([np.pi*np.mean(np.mean(np.sin(THETA)*sigma[pol]))
               for pol in ['hh', 'vh']])

    t_h = sum([np.pi*np.mean(np.mean(np.sin(THETA)*sigma_t[pol]))
               for pol in ['hh', 'vh']])

    # Vertically refracted wave
    r_v = sum([np.pi*np.mean(np.mean(np.sin(THETA)*sigma[pol]))
               for pol in ['hv', 'vv']])

    t_v = sum([np.pi*np.mean(np.mean(np.sin(THETA)*sigma_t[pol]))
               for pol in ['hv', 'vv']])

    return {'horizontal': r_h + t_h, 'vertical': r_v + t_v}


def four_fold_integration(theta_i, wave_vectors, p_slope, amplitudes):
    pass

