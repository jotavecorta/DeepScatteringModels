"""Module for 'T' Scatering Matrix calculation using Small Perturbation
Method (SPM) approximation, for one and two rough surface stacked. 

All the SPM functions were developed based on the equations from 
J. T. Johnson, 'Third-order small-perturbation method for scattering 
from dielectric rough surfaces', J. Opt. Soc. Am., Vol. 16, No. 11, 
November 1999. 

All calculations are made for backscattering angle.

"""
import numpy as np
from scipy.integrate import simps

from johnson_1999 import alpha1_h, aux2_hh, aux2_hv, aux2_vh, aux2_vv, beta1_v
from spm1 import a1HHF1, a1HVF1, a1VVF1, a1HHF2, a1VVF2, a1HVF2
from spm1 import w
from spm2 import L1_11HH, L1_11HV, L1_11VV, L1_22HH, L1_22VV, L1_22HV


class SpmSurface:
    """ Generates a one, or two, layer random rough surface with specified 
    statistics. SPM methods such as T-Matrix availables for backscattering
    angle.
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
        """Return K_1 and K_2 vectors for PSD calculation.
        
        Parameters
        ----------
        lambda_ : float         
            Incident wavelength.
        theta_inc : float       
            Incident azimut angle in radians.
        phi_inc : float        
            Incident azimut polar in radians.

        Returns
        -------
        tuple       
            x and y components of k_s - k_i.
        
        """
        theta, phi = theta_inc, phi_inc + np.pi

        k_1 = 2*np.pi/lambda_*(np.sin(theta)*np.cos(phi) -
                               np.sin(theta_inc)*np.cos(phi_inc))
        k_2 = 2*np.pi/lambda_*(np.sin(theta)*np.sin(phi) -
                               np.sin(theta_inc)*np.sin(phi_inc))

        return k_1, k_2


    def _spm1_amplitudes(self, lambda_, theta_inc, phi_inc):
        """Returns integrated SPM first order amplitudes for one or 
        two layer random rough surface.

        Parameters
        ----------
        lambda_ : float         
            Incident wavelength.
        theta_inc : float         
            Incident azimut angle in radians.
        phi_inc : float        
            Incident azimut polar in radians.

        Returns
        -------
        dict         
            Cross and Co-pol scattering amplitudes.
            
        """
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


    def _spm2_amplitudes(self, lambda_, theta_inc, phi_inc, n=100, lim=1.5):
        """Returns SPM second order amplitudes pounded 
        with respective PSD for one or two layer random rough surface.

        Parameters
        ----------
        lambda_ : float          
            Incident wavelength.
        theta_inc : float         
            Incident azimut angle in radians.
        phi_inc : float        
            Incident azimut polar in radians.
        n : int, default=100         
            Number of points for integration variables.
        lim : float, default=1.5         
            Cut-off wavenumber for 2D integration, in 
            incident wavenumber units.

        Returns
        -------
        dict         
            Cross and Co-pol scattering amplitudes.

        """        
        # Set backscattering for scattered angle
        theta_s, phi_s = theta_inc, phi_inc + np.pi

        # Wave vector norm
        k = 2*np.pi/lambda_

        # Integration domain
        kr_x, kr_y = np.meshgrid(np.linspace(-lim*k, lim*k, n),
                                 np.linspace(-lim*k, lim*k, n))

        # Wave vector components
        k_x, k_y = np.sin(theta_inc) * \
            np.cos(phi_inc), np.sin(theta_inc)*np.sin(phi_inc)

        k_sx, k_sy = np.sin(theta_s) * \
            np.cos(phi_s), np.sin(theta_s)*np.cos(phi_s)

        # scattering amplitudes
        if self.two_layer:
            W_1 = w(self.acf, self.s_1, self.l_1, k_sx - kr_x, k_sy - kr_y) * \
                w(self.acf, self.s_1, self.l_1, kr_x - k_x, kr_y - k_y)

            W_2 = w(self.acf, self.s_2, self.l_2, k_sx - kr_x, k_sy - kr_y) * \
                w(self.acf, self.s_2, self.l_2, kr_x - k_x, kr_y - k_y)

            # Co-pol
            S_hh = W_1 * (
                abs(L1_11HH(kr_x, kr_y, k, theta_s, phi_s, theta_inc, phi_inc, self.ep_1, self.ep_2, self.d))**2 +
                L1_11HH(kr_x, kr_y, k, theta_s, phi_s, theta_inc, phi_inc, self.ep_1, self.ep_2, self.d) *
                np.conj(L1_11HH(k_x - kr_x, k_y - kr_y, k, theta_s, phi_s, theta_inc, phi_inc, self.ep_1, self.ep_2, self.d))) + \
                W_2 * (
                    abs(L1_22HH(kr_x, kr_x, k, theta_s, phi_s, theta_inc, phi_inc, self.ep_1, self.ep_2, self.d))**2 +
                    L1_22HH(kr_x, kr_x, k, theta_s, phi_s, theta_inc, phi_inc, self.ep_1, self.ep_2, self.d) *
                    np.conj(L1_22HH(k_x - kr_x, k_y - kr_y, k, theta_s, phi_s,
                                    theta_inc, phi_inc, self.ep_1, self.ep_2, self.d))
            )

            S_vv = W_1 * (
                abs(L1_11VV(kr_x, kr_y, k, theta_s, phi_s, theta_inc, phi_inc, self.ep_1, self.ep_2, self.d))**2 +
                L1_11VV(kr_x, kr_y, k, theta_s, phi_s, theta_inc, phi_inc, self.ep_1, self.ep_2, self.d) *
                np.conj(L1_11VV(k_x - kr_x, k_y - kr_y, k, theta_s, phi_s, theta_inc, phi_inc, self.ep_1, self.ep_2, self.d))) + \
                W_2 * (
                    abs(L1_22VV(kr_x, kr_x, k, theta_s, phi_s, theta_inc, phi_inc, self.ep_1, self.ep_2, self.d))**2 +
                    L1_22VV(kr_x, kr_x, k, theta_s, phi_s, theta_inc, phi_inc, self.ep_1, self.ep_2, self.d) *
                    np.conj(L1_22VV(k_x - kr_x, k_y - kr_y, k, theta_s, phi_s,
                                    theta_inc, phi_inc, self.ep_1, self.ep_2, self.d))
            )

            S_hh_S_vv = W_1 * (
                L1_11HH(kr_x, kr_y, k, theta_s, phi_s, theta_inc, phi_inc, self.ep_1, self.ep_2, self.d) *
                np.conj(L1_11VV(kr_x, kr_y, k, theta_s, phi_s, theta_inc, phi_inc, self.ep_1, self.ep_2, self.d)) +
                L1_11HH(kr_x, kr_y, k, theta_s, phi_s, theta_inc, phi_inc, self.ep_1, self.ep_2, self.d) *
                np.conj(L1_11VV(k_x - kr_x, k_y - kr_y, k, theta_s, phi_s, theta_inc, phi_inc, self.ep_1, self.ep_2, self.d))) + \
                W_2 * (
                    L1_22HH(kr_x, kr_y, k, theta_s, phi_s, theta_inc, phi_inc, self.ep_1, self.ep_2, self.d) *
                    np.conj(L1_22VV(kr_x, kr_y, k, theta_s, phi_s, theta_inc, phi_inc, self.ep_1, self.ep_2, self.d)) +
                    L1_22HH(kr_x, kr_x, k, theta_s, phi_s, theta_inc, phi_inc, self.ep_1, self.ep_2, self.d) *
                    np.conj(L1_22VV(k_x - kr_x, k_y - kr_y, k, theta_s, phi_s,
                                    theta_inc, phi_inc, self.ep_1, self.ep_2, self.d))
            )

            #Cross-pol
            S_hv = W_1 * (
                abs(L1_11HV(kr_x, kr_y, k, theta_s, phi_s, theta_inc, phi_inc, self.ep_1, self.ep_2, self.d))**2 +
                L1_11HV(kr_x, kr_y, k, theta_s, phi_s, theta_inc, phi_inc, self.ep_1, self.ep_2, self.d) *
                np.conj(L1_11HV(k_x - kr_x, k_y - kr_y, k, theta_s, phi_s, theta_inc, phi_inc, self.ep_1, self.ep_2, self.d))) + \
                W_2 * (
                    abs(L1_22HV(kr_x, kr_x, k, theta_s, phi_s, theta_inc, phi_inc, self.ep_1, self.ep_2, self.d))**2 +
                    L1_22HV(kr_x, kr_x, k, theta_s, phi_s, theta_inc, phi_inc, self.ep_1, self.ep_2, self.d) *
                    np.conj(L1_22HV(k_x - kr_x, k_y - kr_y, k, theta_s, phi_s,
                                    theta_inc, phi_inc, self.ep_1, self.ep_2, self.d))
            )

            S_hh_S_hv = W_1 * (
                L1_11HH(kr_x, kr_y, k, theta_s, phi_s, theta_inc, phi_inc, self.ep_1, self.ep_2, self.d) *
                np.conj(L1_11HV(kr_x, kr_y, k, theta_s, phi_s, theta_inc, phi_inc, self.ep_1, self.ep_2, self.d)) +
                L1_11HH(kr_x, kr_y, k, theta_s, phi_s, theta_inc, phi_inc, self.ep_1, self.ep_2, self.d) *
                np.conj(L1_11HV(k_x - kr_x, k_y - kr_y, k, theta_s, phi_s, theta_inc, phi_inc, self.ep_1, self.ep_2, self.d))) + \
                W_2 * (
                    L1_22HH(kr_x, kr_y, k, theta_s, phi_s, theta_inc, phi_inc, self.ep_1, self.ep_2, self.d) *
                    np.conj(L1_22HV(kr_x, kr_y, k, theta_s, phi_s, theta_inc, phi_inc, self.ep_1, self.ep_2, self.d)) +
                    L1_22HH(kr_x, kr_x, k, theta_s, phi_s, theta_inc, phi_inc, self.ep_1, self.ep_2, self.d) *
                    np.conj(L1_22HV(k_x - kr_x, k_y - kr_y, k, theta_s, phi_s,
                                    theta_inc, phi_inc, self.ep_1, self.ep_2, self.d))
            )

            S_vv_S_hv = W_1 * (
                L1_11VV(kr_x, kr_y, k, theta_s, phi_s, theta_inc, phi_inc, self.ep_1, self.ep_2, self.d) *
                np.conj(L1_11HV(kr_x, kr_y, k, theta_s, phi_s, theta_inc, phi_inc, self.ep_1, self.ep_2, self.d)) +
                L1_11VV(kr_x, kr_y, k, theta_s, phi_s, theta_inc, phi_inc, self.ep_1, self.ep_2, self.d) *
                np.conj(L1_11HV(k_x - kr_x, k_y - kr_y, k, theta_s, phi_s, theta_inc, phi_inc, self.ep_1, self.ep_2, self.d))) + \
                W_2 * (
                    L1_22VV(kr_x, kr_y, k, theta_s, phi_s, theta_inc, phi_inc, self.ep_1, self.ep_2, self.d) *
                    np.conj(L1_22HV(kr_x, kr_y, k, theta_s, phi_s, theta_inc, phi_inc, self.ep_1, self.ep_2, self.d)) +
                    L1_22VV(kr_x, kr_x, k, theta_s, phi_s, theta_inc, phi_inc, self.ep_1, self.ep_2, self.d) *
                    np.conj(L1_22HV(k_x - kr_x, k_y - kr_y, k, theta_s, phi_s,
                                    theta_inc, phi_inc, self.ep_1, self.ep_2, self.d))
            )

        else:
            # Co-pol
            S_hh = aux2_hh(kr_x, kr_y, k, theta_inc, phi_inc, theta_s,
                           phi_s, self.ep_1, self.s_1, self.l_1)

            S_vv = aux2_vv(kr_x, kr_y, k, theta_inc, phi_inc, theta_s,
                           phi_s, self.ep_1, self.s_1, self.l_1)

            S_hh_S_vv = aux2__hh_vv(kr_x, kr_y, k, theta_inc, phi_inc, theta_s,
                                    phi_s, self.ep_1, self.s_1, self.l_1)

            # Cross-pol
            S_hv = aux2_hv(kr_x, kr_y, k, theta_inc, phi_inc, theta_s,
                           phi_s, self.ep_1, self.s_1, self.l_1)

            S_hh_S_hv = aux2_hv_hh(kr_x, kr_y, k, theta_inc, phi_inc, theta_s,
                                   phi_s, self.ep_1, self.s_1, self.l_1)

            S_vv_S_hv = aux2_hv_vv(kr_x, kr_y, k, theta_inc, phi_inc, theta_s,
                                   phi_s, self.ep_1, self.s_1, self.l_1)

        return {
            'co-pol': (S_hh, S_vv, S_hh_S_vv),
            'cross-pol': (S_hv, S_hh_S_hv, S_vv_S_hv)
        }


    def _spm2_two_point(self, lambda_, theta_inc, phi_inc, n=100, lim=1.5):
        """Returns integrated SPM second order amplitudes pounded 
        with respective PSD for one or two layer random rough surface.

        Parameters
        ----------
        lambda_ : float          
            Incident wavelength.
        theta_inc : float         
            Incident azimut angle in radians.
        phi_inc : float        
            Incident azimut polar in radians.
        kwargs: 
            self._spm2_amplitudes integration kwargs

        Returns
        -------
        dict         
            Cross and Co-pol integrated scattering amplitudes.

        """
        # Wave vector norm
        k = 2*np.pi/lambda_

        # Scattering amplitudes
        amps_dict = self._spm2_amplitudes(lambda_, theta_inc, phi_inc, n, lim)
        
        S_hh, S_vv, S_hh_S_vv = amps_dict['co-pol']
        S_hv, S_hh_S_hv, S_vv_S_hv = amps_dict['cross-pol']

        # Integration domain
        kr_x, kr_y = (
            np.linspace(-lim*k, lim*k, n), np.linspace(-lim*k, lim*k, n)
            )
        
        return {
            'co-pol': (
                simps(simps(S_hh.T, kr_y), kr_x),
                simps(simps(S_vv.T, kr_y), kr_x),
                simps(simps(S_hh_S_vv.T, kr_y), kr_x)
            ),
            'cross-pol': (
                simps(simps(S_hv.T, kr_y), kr_x),
                simps(simps(S_hh_S_hv.T, kr_y), kr_x),
                simps(simps(S_vv_S_hv.T, kr_y), kr_x)
            )
        }
       

    def _t_matrix_coeficients(self, lambda_, theta_inc, phi_inc):
        """Returns T-Matrix for one or two layer random rough surface
        scattering in SPM approximation, up to first order.

        Parameters
        ----------
        lambda_ : float        
            Incident wavelength.
        theta_inc : float         
            Incident azimut angle in radians.
        phi_inc : float        
            Incident azimut polar in radians.

        Returns
        -------
        ``numpy.ndarray``      
            Cross and Co-pol integrated scattering amplitudes.

        """
        # Unpack auxiliar vectors
        k_1, k_2 = self._auxiliar_vectors(lambda_, theta_inc, phi_inc)

        # First layer Gaussian Power Spectrum Density
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

            # First order T-Matrix coefficients
            t_11 = W * (np.abs(alpha_h)**2 + np.abs(beta_v) ** 2
                        + 2 * np.real(alpha_h * np.conjugate(beta_v)))

            t_12 = W * (np.abs(alpha_h)**2 - np.abs(beta_v) ** 2
                        - 2 * 1j * np.imag(alpha_h * np.conjugate(beta_v)))

            t_22 = W * (np.abs(alpha_h)**2 + np.abs(beta_v) ** 2
                        - 2 * np.real(alpha_h * np.conjugate(beta_v)))

            t_21 = np.conjugate(t_12)            

            t_13, t_31 = 0, 0

            t_23, t_31 = 0, 0

            return np.array([(t_11, t_12, t_13), 
                             (t_21, t_22, t_23), 
                             (t_21, t_22, t_23)])


    def t_matrix(self, lambda_, theta_inc, phi_inc, second_order=True, **kwargs):
        """Returns T-Matrix for one or two layer random rough surface
        scattering in SPM approximation, up to second order.

        Parameters
        ----------
        lambda_ : float        
            Incident wavelength.
        theta_inc : float         
            Incident azimut angle in radians.
        phi_inc : float        
            Incident azimut polar in radians.
        second_order: bool         
            If true calculations are made up to second 
            order in SPM approximation.

        Returns
        -------
        t_matrix : ``numpy.ndarray``      
            Cross and Co-pol integrated scattering amplitudes.

        """
        # First order calculation
        t_matrix = self._t_matrix_coeficients(lambda_, theta_inc, phi_inc)

        # Add second order terms
        if second_order:
            amps_dict = self._spm2_two_point(lambda_, theta_inc, phi_inc, **kwargs)

            # Unpack amplitudes
            s_hh, s_vv, s_hh_vv = amps_dict['co-pol']
            s_hv, s_hh_hv, s_vv_hv = amps_dict['cross-pol']

            # Second order terms
            # Upper Triangle
            t_11 = s_hh + s_vv + 2 * np.real(s_hh_vv)

            t_12 = s_hh + s_vv - 2j * np.imag(s_hh_vv)

            t_13 = s_hh_hv + s_vv_hv

            t_22 = s_hh + s_vv - 2 * np.real(s_hh_vv)

            t_23 = s_hh_hv - s_vv_hv

            t_33 = 4 * s_hv

            # Lower Triangle
            t_21 = np.conj(t_12)

            t_31 = np.conj(t_13)

            t_32 = np.conj(t_23)

            # Add terms to first order T-Matrix
            t_matrix += np.array([(t_11, t_12, t_13), 
                                  (t_21, t_22, t_23), 
                                  (t_21, t_22, t_23)])

        return t_matrix                                                 
