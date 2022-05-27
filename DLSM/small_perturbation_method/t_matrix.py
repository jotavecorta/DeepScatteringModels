"""Module for 'T' Scatering Matrix calculation using Small Perturbation
Method (SPM) approximation, for one and two rough surface stacked. 
All the SPM functions were developed based on the equations from ..."""
import numpy as np
from scipy.integrate import simps

from johnson_1999 import alpha1_h, aux2_hh, aux2_hv, aux2_vh, aux2_vv, beta1_v
from spm1 import a1HHF1, a1HVF1, a1VVF1, a1HHF2, a1VVF2, a1HVF2
from spm1 import w


class SpmSurface:
    """ Generates a one, or two, layer random rough surface with specified 
    statistics. SPM methods such as T-Matrix availables.
    """
    def __init__(self, rms_high, corr_length, epsilon, two_layer=False,
                 distance=None, epsilon_2=None, rms_high_2=None, corr_length_2=None):
        # Global attributes
        self.acf = 1
        self.two_layer = two_layer

        # First layer attributes
        self.s_1 = rms_high
        self.l_1 = corr_length
        self.ep_1 = epsilon

        # Second layer attributes
        self.s_2 = rms_high_2
        self.l_2 = corr_length_2
        self.ep_2 = epsilon_2
        self.d = distance

    def _auxiliar_vectors(self, lambda_, theta_inc, phi_inc):
        """Return K_1 and K_2 vectors for PSD"""
        theta, phi = theta_inc, phi_inc + np.pi

        k_1 = 2*np.pi/lambda_*(np.sin(theta)*np.cos(phi) -
                               np.sin(theta_inc)*np.cos(phi_inc))
        k_2 = 2*np.pi/lambda_*(np.sin(theta)*np.sin(phi) -
                               np.sin(theta_inc)*np.sin(phi_inc))

        return k_1, k_2

    def _spm1_amplitudes(self, lambda_, theta_inc, phi_inc):

        # Set backscattering for scattered angle
        theta_s, phi_s = theta_inc, phi_inc + np.pi

        # Wave vector norm
        k = 2*np.pi/lambda_

        if self.two_layer:

            return {
                'co-pol': (
                    a1HHF1(k, theta_inc, phi_inc, theta_s,
                           phi_s, self.ep_1, self.ep_2, self.d),
                    a1HHF2(k, theta_inc, phi_inc, theta_s,
                           phi_s, self.ep_1, self.ep_2, self.d),
                    a1VVF1(k, theta_inc, phi_inc, theta_s,
                           phi_s, self.ep_1, self.ep_2, self.d),
                    a1VVF2(k, theta_inc, phi_inc, theta_s,
                           phi_s, self.ep_1, self.ep_2, self.d)
                ),
                'cross-pol': (
                    a1HVF1(k, theta_inc, phi_inc, theta_s,
                           phi_s, self.ep_1, self.ep_2, self.d),
                    a1HVF2(k, theta_inc, phi_inc, theta_s,
                           phi_s, self.ep_1, self.ep_2, self.d),
                )}
        else:

            return {
                'co-pol': (
                    alpha1_h(k, theta_inc, phi_inc, theta_s, phi_s, self.ep_1),
                    beta1_v(k, theta_inc, phi_inc, theta_s, phi_s, self.ep_1)
                )
            }

    def _spm2_amplitudes(self, lambda_, theta_inc, phi_inc, n=100, lim=1.5, mode='simpson'):
        """ """
        # Set backscattering for scattered angle
        theta_s, phi_s = theta_inc, phi_inc + np.pi

        # Wave vector norm
        k = 2*np.pi/lambda_

        # Integration domain
        kr_x, kr_y = np.meshgrid(np.linspace(-lim*k, lim*k, n),
                                 np.linspace(-lim*k, lim*k, n))

        if self.two_layer:
            pass
        else:
            # scattering amplitudes
            # Co-pol
            alpha1_h = aux2_hh(kr_x, kr_y, k, theta_inc, phi_inc, theta_s,
                               phi_s, self.ep_1, self.s_1, self.l_1)

            beta1_v = aux2_vv(kr_x, kr_y, k, theta_inc, phi_inc, theta_s,
                              phi_s, self.ep_1, self.s_1, self.l_1)

            # Cross-pol
            alpha1_v = aux2_hv(kr_x, kr_y, k, theta_inc, phi_inc, theta_s,
                               phi_s, self.ep_1, self.s_1, self.l_1)

            beta1_h = aux2_vh(kr_x, kr_y, k, theta_inc, phi_inc, theta_s,
                              phi_s, self.ep_1, self.s_1, self.l_1)

            return {
                'co-pol': (
                    simps(simps(alpha1_h.T, kr_y), kr_x),
                    simps(simps(beta1_v.T, kr_y), kr_x)
                ),
                'cross-pol': (
                    simps(simps(alpha1_v.T, kr_y), kr_x),
                    simps(simps(beta1_h.T, kr_y), kr_x)
                )
            }
       


    def t_matrix(self, lambda_, theta_inc, phi_inc):

        # Unpack auxiliar vectors
        k_1, k_2 = self._auxiliar_vectors(lambda_, theta_inc, phi_inc)

        # First layer Gaussian Power Spectrum Density
        acf = 1
        W = w(self.acf, self.s_1, self.l_1, k_1, k_2)

        if self.two_layer:
            assert (self.d is not None) and (self.ep_2 is not None), ('Distance between '
                    'layers (d) and second layer dielectric constant (ep_2) '
                    'must have not null input value for two layer calculation.')

            # Second layer Gaussian Power Spectrum Density
            W_2 = w(self.acf, self.s_2, self.l_2, k_1, k_2)

            # Scattering Amplitudes
            amps_dict = self._spm1_amplitudes(lambda_, theta_inc, phi_inc)

            alpha1_h, alpha2_h, beta1_v, beta2_v = amps_dict['co-pol']
            alpha1_v, alpha2_v = amps_dict['cross-pol']

            # T-Matrix coefficients
            # Upper Triangle
            t_11 = W * (np.abs(alpha1_h)**2 + np.abs(beta1_v) ** 2
                        + 2 * np.real(alpha1_h * np.conjugate(beta1_v)))
            + W_2 * (np.abs(alpha2_h)**2 + np.abs(beta2_v) ** 2
                     + 2 * np.real(alpha2_h * np.conjugate(beta2_v)))

            t_12 = W * (np.abs(alpha1_h)**2 - np.abs(beta1_v) ** 2
                        - 2 * 1j * np.imag(alpha1_h * np.conjugate(beta1_v)))
            + W_2 * (np.abs(alpha2_h)**2 - np.abs(beta2_v) ** 2
                     - 2 * 1j * np.imag(alpha2_h * np.conjugate(beta2_v)))

            t_13 = W * (np.conjugate(alpha1_v) * alpha1_h +
                        np.conjugate(alpha1_v) * beta1_v)
            + W_2 * (np.conjugate(alpha2_v) * alpha2_h +
                     np.conjugate(alpha2_v) * beta1_v)

            t_22 = W * (np.abs(alpha1_h)**2 + np.abs(beta1_v) ** 2
                        - 2 * np.real(alpha1_h * np.conjugate(beta1_v)))
            + W_2 * (np.abs(alpha2_h)**2 + np.abs(beta2_v) ** 2
                     - 2 * np.real(alpha2_h * np.conjugate(beta2_v)))

            t_23 = W * (np.conjugate(alpha1_v) * alpha1_h -
                        np.conjugate(alpha1_v) * beta1_v)
            + W_2 * (np.conjugate(alpha2_v) * alpha2_h -
                     np.conjugate(alpha2_v) * beta1_v)

            t_33 = 4 * (W * np.abs(alpha1_v) ** 2 +
                        W_2 * np.abs(alpha2_v) ** 2)

            # Lower Triangle
            t_21 = np.conjugate(t_12)

            t_31 = np.conjugate(t_13)

            t_32 = np.conjugate(t_23)

            return np.array([(t_11, t_12, t_13),
                             (t_21, t_22, t_23),
                             (t_31, t_32, t_33)])

        else:
            # Scattering Amplitudes
            alpha_h, beta_v = self._spm1_amplitudes(
                lambda_, theta_inc, phi_inc)['co-pol']

            # T-Matrix coefficients
            t_11 = W * (np.abs(alpha_h)**2 + np.abs(beta_v) ** 2
                        + 2 * np.real(alpha_h * np.conjugate(beta_v)))

            t_22 = W * (np.abs(alpha_h)**2 + np.abs(beta_v) ** 2
                        - 2 * np.real(alpha_h * np.conjugate(beta_v)))

            t_12 = W * (np.abs(alpha_h)**2 - np.abs(beta_v) ** 2
                        - 2 * 1j * np.imag(alpha_h * np.conjugate(beta_v)))

            t_21 = np.conjugate(t_12)

            return np.array([(t_11, t_12), (t_21, t_22)])

