"""Reordeno las funciones correspondientes a la dispersión de una onda
 planta en dos interfaces rugosas, bajo la aproximación Small Slope 
 Approx., escritas por Mariano en SSA_enero_2021.ipynb.

Para facilitar la vectorización cambié las salidas de varias funciones
 y reemplacé el producto de matrices, por el de Numpy.
 
"""

import numpy as np
from numpy import cos, sin, pi, log10, exp, sqrt, prod

# Vectores de onda


def kx(k, th, ph):
    return k * sin(th) * cos(ph)


def ky(k, th, ph):
    return k * sin(th) * sin(ph)


def alpha(k, x, y, ep):
    return sqrt(ep * k ** 2 - x ** 2 - y ** 2 + 0j)


# Producto matricial
def matrix_prod(A, B):

    r00 = A[0, 0] * B[0, 0] + A[0, 1] * B[1, 0]
    r01 = A[0, 0] * B[0, 1] + A[0, 1] * B[1, 1]
    r10 = A[1, 0] * B[0, 0] + A[1, 1] * B[1, 0]
    r11 = A[1, 0] * B[0, 1] + A[1, 1] * B[1, 1]

    C = np.array([(r00, r01), (r10, r11)])

    return C


def matrix_inverse(A):
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]

    C = 1 / det * np.array([(A[1, 1], -A[0, 1]), (-A[1, 0], A[0, 0])])

    return C


def auxiliar_vectors(lambda_inc, theta_inc, phi_inc, theta, phi):
    """Return K_1 and K_2 vectors for PSD"""

    k_1 = (
        2
        * np.pi
        / lambda_inc
        * (np.sin(theta) * np.cos(phi) - np.sin(theta_inc) * np.cos(phi_inc))
    )
    k_2 = (
        2
        * np.pi
        / lambda_inc
        * (np.sin(theta) * np.sin(phi) - np.sin(theta_inc) * np.sin(phi_inc))
    )

    return k_1, k_2


# Power Spectral Density
def power_spectral_density(k_1, k_2, rms_high, corr_length, acf_type="gaussian"):
    # Rename some variables
    s, l = rms_high, corr_length

    # Defines psd by autocorrelation function
    auto_corr = {
        "gaussian": np.exp(-0.25 * l ** 2 * (k_1 ** 2 + k_2 ** 2)) / (4 * np.pi),
        "exponential": 1
        / (2 * np.pi * (1 + (k_1 ** 2 + k_2 ** 2) * l ** 2) ** (3 / 2)),
        "power_law": np.exp(-np.sqrt(k_1 ** 2 + k_2 ** 2) * l) / (2 * np.pi),
    }

    return s ** 2 * l ** 2 * auto_corr[acf_type]


# Matrices Fh y Fv
def Fmas_v(ux, uy, qx, qy, ki, ep0, ep1, ep2, d):

    f = np.exp(2 * 1j * alpha(ki, ux, uy, ep1) * d)
    a21 = (ep2 * alpha(ki, qx, qy, ep1) - ep1 * alpha(ki, qx, qy, ep2)) / (
        ep2 * alpha(ki, qx, qy, ep1) + ep1 * alpha(ki, qx, qy, ep2)
    )
    a10 = (ep1 * alpha(ki, qx, qy, ep0) - ep0 * alpha(ki, qx, qy, ep1)) / (
        ep1 * alpha(ki, qx, qy, ep0) + ep0 * alpha(ki, qx, qy, ep1)
    )

    return (1 + f * a21) / (1 + f * a21 * a10)


def Fmenos_v(ux, uy, qx, qy, ki, ep0, ep1, ep2, d):

    f = np.exp(2 * 1j * alpha(ki, ux, uy, ep1) * d)
    a21 = (ep2 * alpha(ki, qx, qy, ep1) - ep1 * alpha(ki, qx, qy, ep2)) / (
        ep2 * alpha(ki, qx, qy, ep1) + ep1 * alpha(ki, qx, qy, ep2)
    )
    a10 = (ep1 * alpha(ki, qx, qy, ep0) - ep0 * alpha(ki, qx, qy, ep1)) / (
        ep1 * alpha(ki, qx, qy, ep0) + ep0 * alpha(ki, qx, qy, ep1)
    )

    return (1 - f * a21) / (1 + f * a21 * a10)


def Fmas_h(ux, uy, qx, qy, ki, ep0, ep1, ep2, d):

    f = np.exp(2 * 1j * alpha(ki, ux, uy, ep1) * d)
    b21 = (alpha(ki, qx, qy, ep1) - alpha(ki, qx, qy, ep2)) / (
        alpha(ki, qx, qy, ep1) + alpha(ki, qx, qy, ep2)
    )
    b10 = (alpha(ki, qx, qy, ep0) - alpha(ki, qx, qy, ep1)) / (
        alpha(ki, qx, qy, ep0) + alpha(ki, qx, qy, ep1)
    )

    return (1 + f * b21) / (1 + f * b21 * b10)


def Fmenos_h(ux, uy, qx, qy, ki, ep0, ep1, ep2, d):

    f = np.exp(2 * 1j * alpha(ki, ux, uy, ep1) * d)
    b21 = (alpha(ki, qx, qy, ep1) - alpha(ki, qx, qy, ep2)) / (
        alpha(ki, qx, qy, ep1) + alpha(ki, qx, qy, ep2)
    )
    b10 = (alpha(ki, qx, qy, ep0) - alpha(ki, qx, qy, ep1)) / (
        alpha(ki, qx, qy, ep0) + alpha(ki, qx, qy, ep1)
    )

    return (1 - f * b21) / (1 + f * b21 * b10)


# Productos necesarios para calcular amplitudes
def Qmas(x1, x2, y1, y2, ki, ep0, ep1):

    f = 0.5 * (alpha(ki, x1, x2, ep1) - alpha(ki, x1, x2, ep0)) / \
        alpha(ki, y1, y2, ep0)

    X = np.sqrt(x1 ** 2 + x2 ** 2)
    Y = np.sqrt(y1 ** 2 + y2 ** 2)

    x_dot_y = (x1 * y1 + x2 * y2) / (X * Y)
    x_X_y = (x1 * y2 - x2 * y1) / (X * Y)

    D = X ** 2 + alpha(ki, x1, x2, ep0) * alpha(ki, x1, x2, ep1)

    rh = (alpha(ki, y1, y2, ep0) - alpha(ki, y1, y2, ep1)) / (
        alpha(ki, y1, y2, ep0) + alpha(ki, y1, y2, ep1)
    )
    rv = (ep1 * alpha(ki, y1, y2, ep0) - ep0 * alpha(ki, y1, y2, ep1)) / (
        ep1 * alpha(ki, y1, y2, ep0) + ep0 * alpha(ki, y1, y2, ep1)
    )

    r00 = (
        f
        * (
            X * Y * (1 + rv)
            + (rv - 1) * alpha(ki, y1, y2, ep0) *
            alpha(ki, x1, x2, ep1) * x_dot_y
        )
        / D
    )
    r01 = -f * (1 + rh) * ki * np.sqrt(ep0) * \
        alpha(ki, x1, x2, ep1) * x_X_y / D
    r10 = f * (rv - 1) * alpha(ki, y1, y2, ep0) * x_X_y / (ki * np.sqrt(ep0))
    r11 = f * (1 + rh) * x_dot_y

    m = np.array([(r00, r01), (r10, r11)])

    return m


def T10p_U0p(ux, uy, qx, qy, ki, ep0, ep1, ep2, d):

    rv10 = (ep1 * alpha(ki, qx, qy, ep0) - ep0 * alpha(ki, qx, qy, ep1)) / (
        ep1 * alpha(ki, qx, qy, ep0) + ep0 * alpha(ki, qx, qy, ep1)
    )
    rh10 = (alpha(ki, qx, qy, ep0) - alpha(ki, qx, qy, ep1)) / (
        alpha(ki, qx, qy, ep0) + alpha(ki, qx, qy, ep1)
    )

    rv21 = (ep2 * alpha(ki, qx, qy, ep1) - ep1 * alpha(ki, qx, qy, ep2)) / (
        ep2 * alpha(ki, qx, qy, ep1) + ep1 * alpha(ki, qx, qy, ep2)
    )
    rh21 = (alpha(ki, qx, qy, ep1) - alpha(ki, qx, qy, ep2)) / (
        alpha(ki, qx, qy, ep1) + alpha(ki, qx, qy, ep2)
    )

    f = np.exp(1j * 2 * d * alpha(ki, ux, uy, ep1))

    z11 = (
        2
        * np.sqrt(ep0 * ep1)
        * alpha(ki, ux, uy, ep1)
        / (ep1 * alpha(ki, ux, uy, ep0) + ep0 * alpha(ki, ux, uy, ep1))
        * 1
        / (rv10 * rv21 * f + 1)
    )

    z22 = (
        2
        * alpha(ki, ux, uy, ep1)
        / (alpha(ki, ux, uy, ep0) + alpha(ki, ux, uy, ep1))
        * 1
        / (rh10 * rh21 * f + 1)
    )

    m = np.array([(z11, np.zeros(shape=z11.shape)),
                  (np.zeros(shape=z11.shape), z22)])

    return m


def Qmas_mas(ux, uy, qx, qy, ki, ep0, ep1, ep2, d):

    U = np.sqrt(ux ** 2 + uy ** 2)
    Q = np.sqrt(qx ** 2 + qy ** 2)

    a_u = ep0 * alpha(ki, ux, uy, ep1) + ep1 * alpha(ki, ux, uy, ep0)
    a_q = ep0 * alpha(ki, qx, qy, ep1) + ep1 * alpha(ki, qx, qy, ep0)

    b_u = alpha(ki, ux, uy, ep1) + alpha(ki, ux, uy, ep0)
    b_q = alpha(ki, qx, qy, ep1) + alpha(ki, qx, qy, ep0)

    u_dot_q = (ux * qx + uy * qy) / (U * Q)
    u_X_q = (ux * qy - uy * qx) / (U * Q)

    m00 = (
        (ep1 - ep0)
        / (a_q * a_u)
        * (
            ep1
            * U
            * Q
            * Fmas_v(ux, uy, qx, qy, ki, ep0, ep1, ep2, d)
            * Fmas_v(qx, qy, qx, qy, ki, ep0, ep1, ep2, d)
            - ep0
            * alpha(ki, ux, uy, ep1)
            * alpha(ki, qx, qy, ep1)
            * Fmenos_v(ux, uy, qx, qy, ki, ep0, ep1, ep2, d)
            * Fmenos_v(qx, qy, qx, qy, ki, ep0, ep1, ep2, d)
            * u_dot_q
        )
    )

    m01 = (
        (ep1 - ep0)
        / (a_u * b_q)
        * (
            -np.sqrt(ep0)
            * ki
            * alpha(ki, ux, uy, ep1)
            * Fmenos_v(ux, uy, qx, qy, ki, ep0, ep1, ep2, d)
            * Fmas_h(qx, qy, qx, qy, ki, ep0, ep1, ep2, d)
            * u_X_q
        )
    )

    m10 = (
        (ep1 - ep0)
        / (a_q * b_u)
        * (
            -np.sqrt(ep0)
            * ki
            * alpha(ki, qx, qy, ep1)
            * Fmas_h(ux, uy, qx, qy, ki, ep0, ep1, ep2, d)
            * Fmenos_v(qx, qy, qx, qy, ki, ep0, ep1, ep2, d)
            * u_X_q
        )
    )

    m11 = (
        (ep1 - ep0)
        / (b_u * b_q)
        * ki ** 2
        * Fmas_h(ux, uy, qx, qy, ki, ep0, ep1, ep2, d)
        * Fmas_h(qx, qy, qx, qy, ki, ep0, ep1, ep2, d)
        * u_dot_q
    )

    m = np.array([(m00, m01), (m10, m11)])

    return m


# Amplitudes de Scattering
def X1_u(ux, uy, qx, qy, ki, ep0, ep1, ep2, d):
    return 2 * 1j * Qmas_mas(ux, uy, qx, qy, ki, ep0, ep1, ep2, d)


def X1_b(ux, uy, qx, qy, ki, ep0, ep1, ep2, d):

    X1 = 2 * 1j * Qmas(ux, uy, qx, qy, ki, ep1, ep2)
    f_uq = np.exp(1j * d * (alpha(ki, ux, uy, ep1) + alpha(ki, qx, qy, ep1)))

    m1 = f_uq * matrix_prod(T10p_U0p(ux, uy, ux, uy, ki, ep0, ep1, ep2, d), X1)
    m = matrix_prod(m1, T10p_U0p(qx, qy, qx, qy, ki, ep0, ep1, ep2, d))

    return m


# Sección eficaz suma de superficies
def suma_O1(ki, ths, phs, thi, phi, ep1, ep2, s1, l1, s2, l2, d):
    ep0 = 1

    # Vector de onda incidente y reflejado
    ksx = kx(ki, ths, phs)
    ksy = ky(ki, ths, phs)

    kix = kx(ki, thi, phi)
    kiy = ky(ki, thi, phi)

    # Vectores auxiliares
    k_1, k_2 = auxiliar_vectors(2*np.pi/ki, thi, phi, ths, phs)

    # Power Spectral Density
    W_1 = power_spectral_density(k_1, k_2, s1, l1)
    W_2 = power_spectral_density(k_1, k_2, s2, l2)

    # Prefactor de la seccion eficaz
    f = ki ** 4 * np.cos(ths) ** 2 * np.cos(thi) / (2 * np.pi) ** 2

    # Amplitudes de scattering up y bottom
    X_u = X1_u(ksx, ksy, kix, kiy, ki, ep0, ep1, ep2, d)
    X_b = X1_b(ksx, ksy, kix, kiy, ki, ep0, ep1, ep2, d)

    # Cálculo sección eficaz
    out_ub = f * np.abs(X_u) ** 2 * W_1 + f * np.abs(
        X_b
    ) ** 2 * W_2

    return out_ub
