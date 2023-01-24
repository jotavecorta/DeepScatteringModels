"""Module for 'T' Scatering Matrix calculation using Small Perturbation
Method (SPM) approximation, for one and two rough surface stacked. 

All the SPM functions were developed based on the equations from 
J. T. Johnson, 'Third-order small-perturbation method for scattering 
from dielectric rough surfaces', J. Opt. Soc. Am., Vol. 16, No. 11, 
November 1999. 

All calculations are made for backscattering angle.

"""
import numpy as np
from scipy.integrate import simpson

from .johnson_1999 import alpha1_h, alpha2_h, alpha2_v, beta1_v, beta2_v
from .spm1 import a1HHF1, a1HVF1, a1VVF1, a1HHF2, a1VVF2, a1HVF2
from .spm1 import w
from .spm2 import L0_11HH, L0_11HV, L0_11VV, L0_22HH, L0_22HV, L0_22VV
from .spm2 import L1_11HH, L1_11HV, L1_11VV, L1_22HH, L1_22VV, L1_22HV
from .spm2 import L1_12HH, L1_12HV, L1_12VV

from src.utils import cwishrnd, real_diagonal

class SpmSurface:
    """Generates a one, or two, layer random rough surface with specified 
    statistics. SPM methods such as T-Matrix are availables for backscattering
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

        if self.two_layer:
            assert (self.d is not None) and (self.ep_2 is not None), ('Distance between '
                        'layers (d) and second layer dielectric constant (ep_2) '
                        'must have not null input value for two layer calculation.')

            assert (self.s_2 is not None) and (self.l_2 is not None), ('Second layer rms '
                        'high (rms_high_2) and correlation length (corr_length_2) '
                        'must have not null input value for two layer calculation.')

    def _auxiliar_vectors(self, lambda_, theta_inc, phi_inc):
        """Return K_1 and K_2 vectors for PSD calculation.
        
        Parameters
        ----------
        lambda_ : ``float``         
            Incident wavelength.
        theta_inc : ``float``       
            Incident azimut angle in radians.
        phi_inc : ``float``        
            Incident polar angle in radians.

        Returns
        -------
        ``tuple``       
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
        lambda_ : ``float``         
            Incident wavelength.
        theta_inc : ``float``         
            Incident azimut angle in radians.
        phi_inc : ``float``        
            Incident polar angle in radians.

        Returns
        -------
        ``dict``         
            Cross and Co-pol scattering amplitudes.
            
        """
        # Set backscattering for scattered angle
        theta_s, phi_s = theta_inc, phi_inc + np.pi

        # Wave vector norm
        k = 2*np.pi/lambda_

        if self.two_layer:

            return {
                'first layer': {
                    'co-pol': (
                        a1HHF1(k, theta_inc, phi_inc, theta_s,
                               phi_s, self.ep_1, self.ep_2, self.d),
                        a1VVF1(k, theta_inc, phi_inc, theta_s,
                               phi_s, self.ep_1, self.ep_2, self.d),
                    ),
                    'cross-pol': (
                        a1HVF1(k, theta_inc, phi_inc, theta_s,
                               phi_s, self.ep_1, self.ep_2, self.d))
                },
                'second layer': {
                    'co-pol': (
                        a1HHF2(k, theta_inc, phi_inc, theta_s,
                               phi_s, self.ep_1, self.ep_2, self.d),
                        a1VVF2(k, theta_inc, phi_inc, theta_s,
                               phi_s, self.ep_1, self.ep_2, self.d)
                    ),
                    'cross-pol': (
                        a1HVF2(k, theta_inc, phi_inc, theta_s,
                               phi_s, self.ep_1, self.ep_2, self.d))
                }
            }
        else:
            return {
                'first layer': {
                    'co-pol': (
                        alpha1_h(k, theta_inc, phi_inc,
                                 theta_s, phi_s, self.ep_1),
                        beta1_v(k, theta_inc, phi_inc,
                                theta_s, phi_s, self.ep_1)
                    )
                }
            }

    def _spm2_amplitudes(self, lambda_, theta_inc, phi_inc, int_arguments):
        """Returns SPM second order amplitudes for each layer
         of two layer random rough surface.

        Parameters
        ----------
        lambda_ : ``float``          
            Incident wavelength.
        theta_inc : ``float``         
            Incident azimut angle in radians.
        phi_inc : ``float``        
            Incident polar angle in radians.
        int_argument : ``tuple``
            (x, y) where:
                - x, ``numpy.array`` first argument of L1_11 fuctions
                - y, ``numpy.array`` second argument of L1_11 fuctions 

        Returns
        -------
        ``dict``         
            Each layer Cross and Co-pol scattering amplitudes.

        """
        # Set backscattering for scattered angle
        theta_s, phi_s = theta_inc, phi_inc + np.pi

        # Wave vector norm
        k = 2*np.pi/lambda_

        # Unpack function arguments
        x, y = int_arguments

        if self.two_layer:

            # First layer
            # Co-pol amplitudes
            f01_hh = L0_11HH(k, theta_s, phi_s, theta_inc,
                             phi_inc, self.ep_1, self.ep_2, self.d)
            f1_hh = L1_11HH(x, y, k, theta_s, phi_s, theta_inc,
                            phi_inc, self.ep_1, self.ep_2, self.d)

            f01_vv = L0_11VV(k, theta_s, phi_s, theta_inc,
                             phi_inc, self.ep_1, self.ep_2, self.d)
            f1_vv = L1_11VV(x, y, k, theta_s, phi_s, theta_inc,
                            phi_inc, self.ep_1, self.ep_2, self.d)

            # Cross-pol amplitudes
            f01_hv = L0_11HV(k, theta_s, phi_s, theta_inc,
                             phi_inc, self.ep_1, self.ep_2, self.d)
            f1_hv = L1_11HV(x, y, k, theta_s, phi_s, theta_inc,
                            phi_inc, self.ep_1, self.ep_2, self.d)

            # Second Layer
            # Co-pol amplitudes
            f02_hh = L0_22HH(k, theta_s, phi_s, theta_inc,
                             phi_inc, self.ep_1, self.ep_2, self.d)
            f2_hh = L1_22HH(x, y, k, theta_s, phi_s, theta_inc,
                             phi_inc, self.ep_1, self.ep_2, self.d)

            f02_vv = L0_22VV(k, theta_s, phi_s, theta_inc,
                             phi_inc, self.ep_1, self.ep_2, self.d)
            f2_vv = L1_22VV(x, y, k, theta_s, phi_s, theta_inc,
                            phi_inc, self.ep_1, self.ep_2, self.d)

            # Cross-pol amplitudes
            f02_hv = L0_22HV(k, theta_s, phi_s, theta_inc,
                             phi_inc, self.ep_1, self.ep_2, self.d)
            f2_hv = L1_22HV(x, y, k, theta_s, phi_s, theta_inc,
                            phi_inc, self.ep_1, self.ep_2, self.d)

            # Interaction between layers
            # Co-pol amplitudes
            f12_hh = L1_12HH(x, y, k, theta_s, phi_s, theta_inc,
                        phi_inc, self.ep_1, self.ep_2, self.d) 

            f12_vv = L1_12VV(x, y, k, theta_s, phi_s, theta_inc,
                        phi_inc, self.ep_1, self.ep_2, self.d) 

            # Cross-pol amplitudes
            f12_hv = L1_12HV(x, y, k, theta_s, phi_s, theta_inc,
                        phi_inc, self.ep_1, self.ep_2, self.d)                                  

            return {
                'first layer': {'co-pol': (f01_hh, f01_vv, f1_hh, f1_vv),
                                'cross-pol': (f01_hv, f1_hv)},
                'second layer': {'co-pol': (f02_hh, f02_vv, f2_hh, f2_vv),
                                 'cross-pol': (f02_hv, f2_hv)},
                'interaction': {'co-pol': (f12_hh, f12_vv),
                                'cross-pol': f12_hv}
            }

        else:

            # Co-pol amplitudes
            f1_hh = alpha2_h(x, y, k, theta_inc, phi_inc,
                             theta_s, phi_s, self.ep_1)

            f1_vv = beta2_v(x, y, k, theta_inc, phi_inc,
                            theta_s, phi_s, self.ep_1)

            # Cross-pol amplitudes
            f1_hv = alpha2_v(x, y, k, theta_inc, phi_inc,
                             theta_s, phi_s, self.ep_1)

            return {
                'first layer': {'co-pol': (0, 0, f1_hh, f1_vv),
                                'cross-pol': (0, f1_hv)}
            }

    def _spm1_s_matrix(self, lambda_, theta_inc, phi_inc):
        """Returns SPM first order amplitudes pounded with 
        respective PSD (S matrix) for one or two layer random 
        rough surface.

        Parameters
        ----------
        lambda_ : ``float``          
            Incident wavelength.
        theta_inc : ``float``        
            Incident azimut angle in radians.
        phi_inc : ``float``        
            Incident polar angle in radians.

        Returns
        -------
        ``dict``         
            Cross and Co-pol scattering amplitudes.

        """
        # Unpack auxiliar vectors
        k_1, k_2 = self._auxiliar_vectors(lambda_, theta_inc, phi_inc)

        # First layer Gaussian Power Spectrum Density
        W_1 = w(self.acf, self.s_1, self.l_1, k_1, k_2)

        # Scattering Amplitudes
        amps = self._spm1_amplitudes(
            lambda_, theta_inc, phi_inc
            )
        f1_hh, f1_vv = amps['first layer']['co-pol']

        # First order S-Matrix coefficients
        # Co-pol
        S_hh = W_1 * (np.abs(f1_hh) ** 2)

        S_vv = W_1 * (np.abs(f1_vv) ** 2)

        S_hh_S_vv = W_1 * (f1_hh * np.conjugate(f1_vv))

        # Cross-pol
        S_hv = 0

        S_hh_S_hv = 0

        S_vv_S_hv = 0

        if self.two_layer:

            # Second layer Gaussian Power Spectrum Density
            W_2 = w(self.acf, self.s_2, self.l_2, k_1, k_2)

            # Unpack the rest of the amps
            f1_hv = amps['first layer']['cross-pol']

            f2_hh, f2_vv = amps['second layer']['co-pol']
            f2_hv = amps['second layer']['cross-pol']

            # First order S-Matrix coefficients
            # Co-pol
            S_hh += W_2 * (np.abs(f2_hh) ** 2) 

            S_vv += W_2 * (np.abs(f2_vv) ** 2)

            S_hh_S_vv += W_2 * (f2_hh * np.conjugate(f2_vv))

            # Cross-pol
            S_hv += W_1 * (np.abs(f1_hv) ** 2) + W_2 * (np.abs(f2_hv) ** 2)

            S_hh_S_hv += W_1 * (f1_hh * np.conj(f1_hv)) + \
                W_2 * (f2_hh * np.conj(f2_hv))

            S_vv_S_hv += W_1 * (f1_vv * np.conj(f1_hv)) + \
                W_2 * (f2_vv * np.conj(f2_hv))

        return {
            'co-pol': (S_hh, S_vv, S_hh_S_vv),
            'cross-pol': (S_hv, S_hh_S_hv, S_vv_S_hv)
        }

    def _spm2_s_matrix(self, lambda_, theta_inc, phi_inc, n=100, lim=1.5):
        """Returns SPM second order amplitudes pounded 
        with respective PSD for one or two layer random rough surface.

        Parameters
        ----------
        lambda_ : ``float``          
            Incident wavelength.
        theta_inc : ``float``         
            Incident azimut angle in radians.
        phi_inc : ``float``        
            Incident polar angle in radians.
        n : ``int``, default: 100         
            Number of points for integration variables.
        lim : ``float``, default: 1.5         
            Cut-off wavenumber for 2D integration, in 
            incident wavenumber units.

        Returns
        -------
        ``dict``         
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
        k_x, k_y =k * np.sin(theta_inc) * \
            np.cos(phi_inc), k * np.sin(theta_inc)*np.sin(phi_inc)

        k_sx, k_sy = k * np.sin(theta_s) * \
            np.cos(phi_s), k * np.sin(theta_s)*np.cos(phi_s)


        # Unpack amplitudes first layer
        # First term (evaluated on k')
        amps = self._spm2_amplitudes(lambda_, theta_inc, phi_inc, (kr_x, kr_y))
        f01_hh, f01_vv, f1_hh, f1_vv = amps['first layer']['co-pol']
        f01_hv, f1_hv = amps['first layer']['cross-pol']

        # Second term (evaluated on k - k_i - k')
        amps_st = self._spm2_amplitudes(
            lambda_, theta_inc, phi_inc,
            (k_sx + k_x - kr_x, k_sy + k_y - kr_y)
        )
        _, _, f1_hh_st, f1_vv_st = amps_st['first layer']['co-pol']
        _, f1_hv_st = amps_st['first layer']['cross-pol']

        # First layer Power Spectrum Density
        W_1 = w(self.acf, self.s_1, self.l_1, k_sx - kr_x, k_sy - kr_y) * \
            w(self.acf, self.s_1, self.l_1, kr_x - k_x, kr_y - k_y)


        # S matrix
        # Co-pol
        S_hh = W_1 * (2 * np.abs(f01_hh) ** 2 +
                      2 * np.real(f01_hh * np.conj(f1_hh + f1_hh_st)) +
                      f1_hh * np.conj(f1_hh + f1_hh_st))

        S_vv = W_1 * (2 * np.abs(f01_vv) ** 2 +
                      2 * np.real(f01_vv * np.conj(f1_vv + f1_vv_st)) +
                      f1_vv * np.conj(f1_vv + f1_vv_st))


        S_hh_S_vv = W_1 * (2 * f01_hh * np.conj(f01_vv) +
                           f01_hh * np.conj(f1_vv + f1_vv_st) +
                           np.conj(f01_vv) * (f1_hh + f1_hh_st) +
                           f1_hh * np.conj(f1_vv + f1_vv_st))

        # Cross-pol
        S_hv = W_1 * (2 * np.abs(f01_hv) ** 2 +
                      2 * np.real(f01_hv * np.conj(f1_hv + f1_hv_st)) +
                      f1_hv * np.conj(f1_hv + f1_hv_st))

        S_hh_S_hv = W_1 * (2 * f01_hh * np.conj(f01_hv) +
                           f01_hh * np.conj(f1_hv + f1_hv_st) +
                           np.conj(f01_hv) * (f1_hh + f1_hh_st) +
                           f1_hh * np.conj(f1_hv + f1_hv_st))

        S_vv_S_hv = W_1 * (2 * f01_vv * np.conj(f01_hv) +
                           f01_vv * np.conj(f1_hv + f1_hv_st) +
                           np.conj(f01_hv) * (f1_vv + f1_vv_st) +
                           f1_vv * np.conj(f1_hv + f1_hv_st))

        if self.two_layer:

            # Second layer amplitudes
            f02_hh, f02_vv, f2_hh, f2_vv = amps['second layer']['co-pol']
            f02_hv, f2_hv = amps['second layer']['cross-pol']

            _, _, f2_hh_st, f2_vv_st = amps_st['second layer']['co-pol']
            _, f2_hv_st = amps_st['second layer']['cross-pol']

            f12_hh, f12_vv = amps['interaction']['co-pol']
            f12_hv = amps['interaction']['cross-pol']

            f12_hh_st, f12_vv_st = amps_st['interaction']['co-pol']
            f12_hv_st = amps_st['interaction']['cross-pol']


            # Second layer Power Spectrum Density
            W_2 = w(self.acf, self.s_2, self.l_2, k_sx - kr_x, k_sy - kr_y) * \
                w(self.acf, self.s_2, self.l_2, kr_x - k_x, kr_y - k_y)

            # Interaction Power Spectrum Density
            W_12 = w(self.acf, self.s_1, self.l_1, k_sx - kr_x, k_sy - kr_y) * \
                w(self.acf, self.s_2, self.l_2, kr_x - k_x, kr_y - k_y)


            # Co-pol Elements
            S_hh += W_2 * (2 * np.abs(f02_hh) ** 2 +
                           2 * np.real(f02_hh * np.conj(f2_hh + f2_hh_st)) +
                           f2_hh * np.conj(f2_hh + f2_hh_st)) + \
                   W_12 * (f12_hh * np.conj(f12_hh + f12_hh_st))

            S_vv += W_2 * (2 * np.abs(f02_vv) ** 2 +
                           2 * np.real(f02_vv * np.conj(f2_vv + f2_vv_st)) +
                           f2_vv * np.conj(f2_vv + f2_vv_st)) + \
                   W_12 * (f12_vv * np.conj(f12_vv + f12_vv_st))

            S_hh_S_vv += W_2 * (2 * f02_hh * np.conj(f02_vv) +
                                f02_hh * np.conj(f2_vv + f2_vv_st) +
                                np.conj(f02_vv) * (f2_hh + f2_hh_st) +
                                f2_hh * np.conj(f1_vv + f2_vv_st)) + \
                        W_12 * (f12_hh * np.conj(f12_vv + f12_vv_st))

            # Cross-pol Elements
            S_hv += W_2 * (2 * np.abs(f02_hv) ** 2 +
                           2 * np.real(f02_hv * np.conj(f2_hv + f2_hv_st)) +
                           f2_hv * np.conj(f2_hv + f2_hv_st)) + \
                   W_12 * (f12_hv * np.conj(f12_hv + f12_hv_st))

            S_hh_S_hv += W_2 * (2 * f02_hh * np.conj(f02_hv) +
                                f02_hh * np.conj(f2_hv + f2_hv_st) +
                                np.conj(f02_hv) * (f2_hh + f2_hh_st) +
                                f2_hh * np.conj(f2_hv + f2_hv_st)) + \
                        W_12 * (f12_hh * np.conj(f12_hv) + f12_hh * np.conj(f12_hv_st))

            S_vv_S_hv += W_2 * (2 * f02_vv * np.conj(f02_hv) +
                                f02_vv * np.conj(f2_hv + f2_hv_st) +
                                np.conj(f02_hv) * (f2_vv + f2_vv_st) +
                                f2_vv * np.conj(f2_hv + f2_hv_st)) + \
                        W_12 * (f12_vv * np.conj(f12_hv + f12_hv_st))

        return {
            'co-pol': (S_hh, S_vv, S_hh_S_vv),
            'cross-pol': (S_hv, S_hh_S_hv, S_vv_S_hv)
        }

    def _spm2_integration(self, lambda_, theta_inc, phi_inc, n=100, lim=1.5):
        """Returns integrated SPM second order S Matrix coeffiecients
        for one or two layer random rough surface.

        Parameters
        ----------
        lambda_ : ``float``          
            Incident wavelength.
        theta_inc : ``float``         
            Incident azimut angle in radians.
        phi_inc : ``float``        
            Incident polar angle in radians.
        n : ``int``, default: 100         
            Number of points for integration variables.
        lim : ``float``, default: 1.5         
            Cut-off wavenumber for 2D integration, in 
            incident wavenumber units.

        Returns
        -------
        ``dict``         
            Cross and Co-pol integrated scattering amplitudes.

        """
        # Wave vector norm
        k = 2*np.pi/lambda_

        # Scattering amplitudes
        amps_dict = self._spm2_s_matrix(lambda_, theta_inc, phi_inc, n, lim)

        S_hh, S_vv, S_hh_S_vv = amps_dict['co-pol']
        S_hv, S_hh_S_hv, S_vv_S_hv = amps_dict['cross-pol']

        # Integration domain
        kr_x, kr_y = (
            np.linspace(-lim*k, lim*k, n), np.linspace(-lim*k, lim*k, n)
        )

        return {
            'co-pol': (
                simpson(simpson(S_hh.T, kr_y), kr_x),
                simpson(simpson(S_vv.T, kr_y), kr_x),
                simpson(simpson(S_hh_S_vv.T, kr_y), kr_x)
            ),
            'cross-pol': (
                simpson(simpson(S_hv.T, kr_y), kr_x),
                simpson(simpson(S_hh_S_hv.T, kr_y), kr_x),
                simpson(simpson(S_vv_S_hv.T, kr_y), kr_x)
            )
        }

    def _t_matrix_first_order(self, lambda_, theta_inc, phi_inc):
        """Returns T-Matrix for one or two layer random rough surface
        scattering in SPM approximation, up to first order.

        Parameters
        ----------
        lambda_ : ``float``        
            Incident wavelength.
        theta_inc : ``float``         
            Incident azimut angle in radians.
        phi_inc : ``float``        
            Incident polar angle in radians.

        Returns
        -------
        ``numpy.ndarray``      
            Cross and Co-pol integrated scattering amplitudes.

        """
        s_matrix_dict = self._spm1_s_matrix(lambda_, theta_inc, phi_inc)

        # Unpack amplitudes
        s_hh, s_vv, s_hh_vv = s_matrix_dict['co-pol']
        s_hv, s_hh_hv, s_vv_hv = s_matrix_dict['cross-pol']

        # T Matrix
        # Upper Triangle
        t_11 = s_hh + s_vv + 2 * np.real(s_hh_vv)

        t_12 = s_hh + s_vv - 2j * np.imag(s_hh_vv)

        t_13 = 2 * (s_hh_hv + s_vv_hv)

        t_22 = s_hh + s_vv - 2 * np.real(s_hh_vv)

        t_23 = 2 * (s_hh_hv - s_vv_hv)

        t_33 = 4 * s_hv

        # Lower Triangle
        t_21 = np.conj(t_12)

        t_31 = np.conj(t_13)

        t_32 = np.conj(t_23)

        # Add terms to first order T-Matrix
        t_matrix = .5 * np.array([(t_11, t_12, t_13),
                                  (t_21, t_22, t_23),
                                  (t_31, t_32, t_33)])

        return t_matrix

    def t_matrix(
        self, 
        lambda_, 
        theta_inc, 
        phi_inc, 
        second_order=True, 
        noise=False, 
        **int_kw
        ):
        """Returns T-Matrix for one or two layer random rough surface
        scattering in SPM approximation, up to second order.

        Parameters
        ----------
        lambda_ : ``float``        
            Incident wavelength.
        theta_inc : ``float``         
            Incident azimut angle in radians.
        phi_inc : ``float``        
            Incident polar angle in radians.
        second_order : ``bool``, default: True          
            If true calculations are made up to second 
            order in SPM approximation.
        **int_kw : `` ``
            All additional keyword arguments are pass to 
            t_matrix.SpmSurface._spm2_integration call.            

        Returns
        -------
        t_matrix : ``numpy.ndarray``      
            Cross and Co-pol integrated scattering amplitudes.

        """
        # Incident wave number
        k = 2 * np.pi / lambda_ 

        # First order calculation
        t_matrix = self._t_matrix_first_order(lambda_, theta_inc, phi_inc)

        # Add second order terms
        if second_order:
            amps_dict = self._spm2_integration(
                lambda_, theta_inc, phi_inc, **int_kw
                )

            # Unpack amplitudes
            s_hh, s_vv, s_hh_vv = amps_dict['co-pol']
            s_hv, s_hh_hv, s_vv_hv = amps_dict['cross-pol']

            # Second order terms
            # Upper Triangle
            t_11 = s_hh + s_vv + 2 * np.real(s_hh_vv)

            t_12 = s_hh + s_vv - 2j * np.imag(s_hh_vv)

            t_13 = 2 * (s_hh_hv + s_vv_hv)

            t_22 = s_hh + s_vv - 2 * np.real(s_hh_vv)

            t_23 = 2 * (s_hh_hv - s_vv_hv)

            t_33 = 4 * s_hv

            # Lower Triangle
            t_21 = np.conj(t_12)

            t_31 = np.conj(t_13)

            t_32 = np.conj(t_23)

            # Add terms to first order T-Matrix
            t_matrix += .5 * (np.array([(t_11, t_12, t_13),
                                        (t_21, t_22, t_23),
                                        (t_31, t_32, t_33)])
                                        )
        if noise:            
            t_matrix = cwishrnd(
                real_diagonal(t_matrix)
                )                                

        return 4 * np.pi * k**2 * np.cos(theta_inc)**2 * t_matrix

    def mueller_matrix(
        self, 
        lambda_, 
        theta_inc, 
        phi_inc, 
        second_order=True, 
        noise=False, 
        **int_kw
        ):
        """Returns Mueller Matrix for one or two layer random rough surface
        scattering in SPM approximation, up to second order.

        Parameters
        ----------
        lambda_ : ``float``        
            Incident wavelength.
        theta_inc : ``float``         
            Incident azimut angle in radians.
        phi_inc : ``float``        
            Incident polar angle in radians.
        second_order : ``bool``, default: True          
            If true calculations are made up to second 
            order in SPM approximation.
        **int_kw : `` ``
            All additional keyword arguments are pas to 
            t_matrix.SpmSurface._spm2_integration call.

        Returns
        -------
        m_matrix : ``numpy.ndarray``      
            Cross and Co-pol integrated scattering amplitudes.

        Reference
        ---------
        F.T. Ulaby, K. Sarabandi, A. Nashashibi, "Statistical properties off
        the mueller matrix of distributed targets", IEEE Proceedings F (Radar 
        and Signal Processing), Volume 139, Issue 2, pp. 136-146, April 1992.

        M. Migliaccio et al., "A multi-frequency polarimetric SAR processing 
        chain to observe oil fields in the Gulf of Mexico", IEEE Transaction 
        on Geoscience and Remote Sensing, Volume 49, Issue 12, pp. 4729-4737, 
        2011. 

        """
        # Incident wave number
        k = 2 * np.pi / lambda_ 

        # Coherence T-Matrix
        T = self.t_matrix(
            lambda_, 
            theta_inc, 
            phi_inc, 
            second_order=second_order,
            noise=noise,  
            **int_kw
            )
        
        # Upper Triangle
        m_11 = 1/2 * (T[0,0] + T[1,1] + T[2,2])

        m_12 = np.real(T[0, 1])
        
        m_13 = 1/2 * np.real(T[0, 2] + T[1, 2] + T[2, 0] - T[2, 1])

        m_14 = 1/2 * np.imag(T[0, 2] + T[1, 2] + T[2, 0] - T[2, 1])
        
        m_22 = 1/2 * (T[0,0] + T[1,1] - T[2,2]) 
        
        m_23 = 1/2 * np.real(T[0, 2] + T[1, 2] - T[2, 0] + T[2, 1])

        m_24 = 1/2 * np.imag(T[0, 2] + T[1, 2] - T[2, 0] + T[2, 1])
        
        m_33 = 1/2 * (T[0,0] - T[1,1] + T[2,2])

        m_34 = - np.imag(T[0, 1])

        m_44 = 1/2 * (+T[0,0] - T[1,1] - T[2,2])                

        # Lower Triangle
        m_21 = m_12

        m_31, m_32 = m_13, m_23

        m_41 = - m_14

        m_42, m_43 = -m_24, -m_34

        # Mueller Matrix
        m_matrix = 1/2 * (np.array([(m_11, m_12, m_13, m_14),
                                    (m_21, m_22, m_23, m_24),
                                    (m_31, m_32, m_33, m_34),
                                    (m_41, m_42, m_43, m_44)]))

        return np.real(m_matrix)

    def polarization_signature(
        self,
        lambda_,
        theta_inc,
        phi_inc,
        grid_size=(45, 90),
        second_order=True,
        wishard_noise=False,
        **int_kw
    ):
        """Returns Polarization Signature grid, for one or two layer random rough surface
        scattering in SPM approximation, up to second order.

        Parameters
        ----------
        lambda_ : ``float``        
            Incident wavelength.
        theta_inc : ``float``         
            Incident azimut angle in radians.
        phi_inc : ``float``        
            Incident polar angle in radians.
        second_order : ``bool``, default: True         
            If true calculations are made up to second 
            order in SPM approximation.
        grid_size : ``int tuple``, default: (90, 45)
            Shape of resulting array. 
            (a, b) where ::
                - a, ellipticity angle length.
                - b, orientation angle length.  
        wishard_noise :  ``bool``, default: False
            Add noise with wishard distribution,
            simulating 15 looks messure.          
        **int_kw :
            All additional keyword arguments are passed to 
            t_matrix.SpmSurface._spm2_integration call.    

        Returns
        -------
        m_matrix : ``numpy.ndarray``      
            Cross and Co-pol integrated scattering amplitudes.

        Reference
        ---------
        J. J. v. Zyl, H. A. Zebker and C. Elachi, "Imaging radar polarization 
        signatures: Theory and observation," in Radio Science, vol. 22, no. 04, 
        pp. 529-543, July-Aug. 1987.

        """
        # Incident wave vector
        k = 2*np.pi/lambda_

        # Mueller matrix
        m_matrix = self.mueller_matrix(
            lambda_, 
            theta_inc, 
            phi_inc, 
            second_order=second_order, 
            noise=wishard_noise, 
            **int_kw
        )

        # Grid Size
        n_chi, n_psi = grid_size

        # Orientation and ellipticity angles
        psi = np.linspace(0, 180, n_psi) * np.pi/180
        chi = np.linspace(-45, 45, n_chi) * np.pi/180

        PSI, CHI = np.meshgrid(psi, chi)

        # Rotation Vectors
        R = np.array([[np.ones(CHI.shape)], 
                        [np.cos(2*CHI) * np.cos(2*PSI)],
                        [np.cos(2*CHI) * np.sin(2*PSI)],
                        [np.sin(2*CHI)]])

        # Polarization signature
        M_dot_R = np.einsum('ij, jklm -> iklm', m_matrix, R)
        dot_product = np.einsum(
            'kilm, iklm -> lm',
             np.transpose(R, (1, 0, 2, 3)), 
             M_dot_R
            )

        sigma = 4 * np.pi/k**2 * dot_product  

        return sigma   

