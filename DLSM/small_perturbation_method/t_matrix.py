"""Module for 'T' Scatering Matrix calculation using SMP approximation, 
for one and two rough surface stacked. All the SPM functions were 
developed based on the equations from ..."""
import numpy as np

from johnson_1999 import alpha1_h, alpha1_v
from spm1 import a1HHF1, a1HVF1, a1VHF1, a1VVF1, a1HHF2, a1VVF2, a1HVF2
from spm1 import w, w_f1f2

class SpmSurface(rms_high, corr_length, epsilon, two_layer=False,
                distance=None, epsilon_2=None, rms_high_2=None, corr_length_2=None):
    
    def __init__(self):
        # First layer attributes
        s_1 = self.rms_high
        l_1 = self.corr_length
        ep_1 = self.epsilon

        # Second layer attributes
        s_2 = self.rms_high_2
        l_2 = self.corr_length_2
        ep_2 = self.epsilon_2 
        d = self.distance

    def auxiliar_vectors(self, lambda_, theta_inc, phi_inc):
        """Return K_1 and K_2 vectors for PSD"""
        theta, phi = theta_inc, phi_inc + np.pi
        
        k_1 = 2*np.pi/lambda_*(np.sin(theta)*np.cos(phi) - np.sin(theta_inc)*np.cos(phi_inc))
        k_2 = 2*np.pi/lambda_*(np.sin(theta)*np.sin(phi) - np.sin(theta_inc)*np.sin(phi_inc))
        
        return k_1, k_2


    def spm1_amplitudes(self, lambda_, theta_inc, phi_inc):

        # Set backscattering for scattered angle
        theta_s, phi_s = theta_inc, phi_inc + np.pi

        # Wave vector norm
        k = 2*np.pi/lambda_

        if self.two_layer:

            return {
                'co-pol': (
                    a1HHF1(k, theta_inc, phi_inc, theta_s, phi_s, ep, ep_2, d),
                    a1HHF2(k, theta_inc, phi_inc, theta_s, phi_s, ep, ep_2, d),
                    a1VVF1(k, theta_inc, phi_inc, theta_s, phi_s, ep, ep_2, d),
                    a1VVF2(k, theta_inc, phi_inc, theta_s, phi_s, ep, ep_2, d)
                ),
                'cross-pol': (
                    a1HVF1(k, theta_inc, phi_inc, theta_s, phi_s, ep, ep_2, d),
                    a1HVF2(k, theta_inc, phi_inc, theta_s, phi_s, ep, ep_2, d),
                    a1VHF1(k, theta_inc, phi_inc, theta_s, phi_s, ep, ep_2, d),
                    a1VHF2(k, theta_inc, phi_inc, theta_s, phi_s, ep, ep_2, d)
                )}
        else:

            return {
                'co-pol': (
                    alpha1_h(k, theta_inc, phi_inc, theta_s, phi_s, ep),
                    alpha1_v(k, theta_inc, phi_inc, theta_s, phi_s, ep)
                )
            }


    def t_matrix(self, lambda_, theta_inc, phi_inc):
        
        # Unpack auxiliar vectors
        k_1, k_2 = auxiliar_vectors(lambda_, theta_inc, phi_inc)   

        # First layer Gaussian Power Spectrum Density
        acf=1
        W = w(s_1, l_1, k_1, k_2, acf=1)     
            
        if self.two_layer:   
            assert (d is not None) and (ep_2 is not None), ('Distance between ' 
            'layers (d) and second layer dielectric constant (ep_2) '
            'must have not null input value for two layer calculation.')

            # Second layer Gaussian Power Spectrum Density
            W_12 = w_f1f2(s_1, l_1, s_2, l_2, k_1, k_2)

        else:
            # Scattering Amplitudes
            alpha_h, alpha_v = self.spm1_amplitudes(
                lambda_, theta_inc, phi_inc)

            # T-Matrix coefficients
            t_11 = W * (np.abs(alpha_h)**2 + np.abs(alpha_v)
                        ** 2 + 2 * np.real(alpha_h * alpha_h))
            
            t_22 = W * (np.abs(alpha_h)**2 + np.abs(alpha_v)
                        ** 2 - 2 * np.real(alpha_h * alpha_h))

            t_12 = W * (np.abs(alpha_h)**2 - np.abs(alpha_v)
                        ** 2 - 2 * 1j* np.imag(alpha_h * alpha_h))  

            t_21 = np.conjugate(t_12) 

        return np.array([(t_11, t_12),(t_21, t_22)])                                                 
