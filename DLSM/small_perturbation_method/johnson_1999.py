import numpy as np
from numpy import pi, sqrt, sin, cos, exp, log10


# onda incidente
def kix(ki, th, ph):
    return ki * cos(ph) * sin(th)


def kiy(ki, th, ph):
    return ki * sin(ph) * sin(th)


def kiz(ki, th):
    return ki * cos(th)


def kperp_i(ki, th, ph):
    return sqrt(kix(ki, th, ph) ** 2 + kiy(ki, th, ph) ** 2)


# onda dispersada
def ksx(ki, ths, phs):
    return ki * cos(phs) * sin(ths)


def ksy(ki, ths, phs):
    return ki * sin(phs) * sin(ths)


def kperp_s(ki, ths, phs):
    return sqrt(ksx(ki, ths, phs) ** 2 + ksy(ki, ths, phs) ** 2)


def ksz(ki, ths, phs):
    return sqrt(ki ** 2 - kperp_s(ki, ths, phs) ** 2)


# onda transmitida


def kiz1(ki, th, ph, ep):
    return sqrt(ep * ki ** 2 - kperp_i(ki, th, ph) ** 2)


def ktz(ki, ths, phs, ep):
    return sqrt(ep * ki ** 2 - kperp_s(ki, ths, phs) ** 2)


# Espectro de potencias de la superficie #

# exponencial
def w_exp(s, l, k1, k2):
    return s ** 2 * l ** 2 / (2 * np.pi * (1 + (k1 ** 2 + k2 ** 2) * l ** 2) ** (3 / 2))


# gaussiana
def w_gauss(s, l, k1, k2):
    return s ** 2 * l ** 2 / (4 * np.pi) * exp(-0.25 * l ** 2 * (k1 ** 2 + k2 ** 2))


# power law
def w_pl(s, l, k1, k2):
    return s ** 2 * l ** 2 / (2 * np.pi) * exp(-sqrt(k1 ** 2 + k2 ** 2) * l)


# Orden cero #

# MODO TE #
def alpha0_h(th, ep):
    return (cos(th) - sqrt(ep - sin(th) ** 2)) / (cos(th) + sqrt(ep - sin(th) ** 2))


def beta0_h(th, ep):
    return 0


# MODO TM #
def alpha0_v(th, ep):
    return 0


def beta0_v(th, ep):
    return (
        (ep - 1)
        * (sin(th) ** 2 - ep * (1 + sin(th) ** 2))
        / (ep * cos(th) + sqrt(ep - sin(th) ** 2)) ** 2
    )


# ----------------------------------------------------------------------------------------------#

# Orden uno #
def c_is(ki, th, ph, ths, phs):
    return (
        kix(ki, th, ph) * ksx(ki, ths, phs) + kiy(ki, th, ph) * ksy(ki, ths, phs)
    ) / (kperp_i(ki, th, ph) * kperp_s(ki, ths, phs))


def s_is(ki, th, ph, ths, phs):
    return (
        kix(ki, th, ph) * ksy(ki, ths, phs) - kiy(ki, th, ph) * ksx(ki, ths, phs)
    ) / (kperp_i(ki, th, ph) * kperp_s(ki, ths, phs))


# MODO TE #
def alpha1_h(ki, th, ph, ths, phs, ep):
    return (
        -2
        * 1j
        * kiz(ki, th)
        * (ki ** 2 - ep * ki ** 2)
        / (
            (ksz(ki, ths, phs) + ktz(ki, ths, phs, ep))
            * (kiz(ki, th) + kiz1(ki, th, ph, ep))
        )
        * c_is(ki, th, ph, ths, phs)
    )


def beta1_h(ki, th, ph, ths, phs, ep):
    return (
        -2
        * 1j
        * kiz(ki, th)
        * (ki ** 2 - ep * ki ** 2)
        / (
            (ep * ksz(ki, ths, phs) + ktz(ki, ths, phs, ep))
            * (kiz(ki, th) + kiz1(ki, th, ph, ep))
        )
        * (ktz(ki, ths, phs, ep) / ki)
        * s_is(ki, th, ph, ths, phs)
    )


def gamma1_h(ki, th, ph, ths, phs, ep):
    return alpha1_h(ki, th, ph, ths, phs, ep)


def delta1_h(ki, th, ph, ths, phs, ep):
    return -((ksz(ki, ths, phs) * ep * ki) / (ktz(ki, ths, phs, ep) * ki)) * beta1_h(
        ki, th, ph, ths, phs, ep
    )


# MODO TM #
def alpha1_v(ki, th, ph, ths, phs, ep):
    return (
        -2
        * 1j
        * kiz(ki, th)
        * (ki ** 2 - ep * ki ** 2)
        / (
            (ksz(ki, ths, phs) + ktz(ki, ths, phs, ep))
            * (ep * kiz(ki, th) + kiz1(ki, th, ph, ep))
        )
        * (kiz1(ki, th, ph, ep) / ki)
        * s_is(ki, th, ph, ths, phs)
    )


def beta1_v(ki, th, ph, ths, phs, ep):
    return (
        -2
        * 1j
        * kiz(ki, th)
        * (ki ** 2 - ep * ki ** 2)
        / (
            (ep * ksz(ki, ths, phs) + ktz(ki, ths, phs, ep))
            * (ep * kiz(ki, th) + kiz1(ki, th, ph, ep))
        )
        * (
            ep * kperp_i(ki, th, ph) * kperp_s(ki, ths, phs) / ki ** 2
            - kiz1(ki, th, ph, ep)
            * ktz(ki, ths, phs, ep)
            * c_is(ki, th, ph, ths, phs)
            / ki ** 2
        )
    )


def gamma1_v(ki, th, ph, ths, phs, ep):
    return alpha1_v(ki, th, ph, ths, phs, ep)


def delta1_v(ki, th, ph, ths, phs, ep):
    return (
        -2
        * 1j
        * kiz(ki, th)
        * (ki ** 2 - ep * ki ** 2)
        * sqrt(ep)
        / (
            (ep * ksz(ki, ths, phs) + ktz(ki, ths, phs, ep))
            * (ep * kiz(ki, th) + kiz1(ki, th, ph, ep))
        )
        * (
            kperp_i(ki, th, ph) * kperp_s(ki, ths, phs) / ki ** 2
            + kiz1(ki, th, ph, ep)
            * ksz(ki, ths, phs)
            * c_is(ki, th, ph, ths, phs)
            / ki ** 2
        )
    )


## Orden dos ##
# coeficientes auxiliares #
# los modos krx y kry luego van a ser integrados: términos de multiple scattering#


def c_si(ki, th, ph, ths, phs):
    return (
        kix(ki, th, ph) * ksx(ki, ths, phs) + kiy(ki, th, ph) * ksy(ki, ths, phs)
    ) / (kperp_i(ki, th, ph) * kperp_s(ki, ths, phs))


def c_ri(krx, kry, ki, th, ph, ths, phs):
    return (krx * kix(ki, th, ph) + kry * kiy(ki, th, ph)) / (
        kperp_i(ki, th, ph) * sqrt(krx ** 2 + kry ** 2)
    )


def c_rs(krx, kry, ki, th, ph, ths, phs):
    return (ksx(ki, ths, phs) * krx + ksy(ki, ths, phs) * kry) / (
        kperp_s(ki, ths, phs) * sqrt(krx ** 2 + kry ** 2)
    )


def s_si(krx, kry, ki, th, ph, ths, phs):
    return (
        kix(ki, th, ph) * ksy(ki, ths, phs) - kiy(ki, th, ph) * ksx(ki, ths, phs)
    ) / (kperp_i(ki, th, ph) * kperp_s(ki, ths, phs))


def s_ri(krx, kry, ki, th, ph, ths, phs):
    return (krx * kiy(ki, th, ph) - kry * kix(ki, th, ph)) / (
        kperp_i(ki, th, ph) * sqrt(krx ** 2 + kry ** 2)
    )


def s_rs(krx, kry, ki, th, ph, ths, phs):
    return (krx * ksy(ki, ths, phs) - kry * ksx(ki, ths, phs)) / (
        sqrt(krx ** 2 + kry ** 2) * kperp_s(ki, ths, phs)
    )


def kappa1(krx, kry, ki, ep):
    return sqrt(ki ** 2 - krx ** 2 - kry ** 2 + 0j) - sqrt(
        ep * ki ** 2 - krx ** 2 - kry ** 2 + 0j
    )


def kappa2(krx, kry, ki, ep):
    return (
        sqrt(ki ** 2 - krx ** 2 - kry ** 2 + 0j)
        * sqrt(ep * ki ** 2 - krx ** 2 - kry ** 2 + 0j)
        * (ki ** 2 - ep * ki ** 2)
        / (
            ki ** 2
            * (
                ep * sqrt(ki ** 2 - krx ** 2 - kry ** 2 + 0j)
                + sqrt(ep * ki ** 2 - krx ** 2 - kry ** 2 + 0j)
            )
        )
    )


def kappa3(krx, kry, ki, ep):
    return (
        sqrt(krx ** 2 + kry ** 2)
        * ki ** 2
        / (
            sqrt(krx ** 2 + kry ** 2) ** 2
            + sqrt(ki ** 2 - krx ** 2 - kry ** 2 + 0j)
            * sqrt(ep * ki ** 2 - krx ** 2 - kry ** 2 + 0j)
        )
    )


# MODO TE #


def alpha2_h(krx, kry, ki, th, ph, ths, phs, ep):
    return (
        -2
        * kiz(ki, th)
        * (ki ** 2 - ep * ki ** 2)
        / (
            (ksz(ki, ths, phs) + ktz(ki, ths, phs, ep))
            * (kiz(ki, th) + kiz1(ki, th, ph, ep))
        )
        * (
            c_rs(krx, kry, ki, th, ph, ths, phs)
            * c_ri(krx, kry, ki, th, ph, ths, phs)
            * (kappa1(krx, kry, ki, ep) + sqrt(ep * ki ** 2 - krx ** 2 - kry ** 2 + 0j))
            + s_rs(krx, kry, ki, th, ths, ths, phs)
            * s_ri(krx, kry, ki, th, ph, ths, phs)
            * (ktz(ki, ths, phs, ep) + kappa2(krx, kry, ki, ep))
            + 0.5
            * c_si(ki, th, ph, ths, phs)
            * (kiz1(ki, th, ph, ep) - ktz(ki, ths, phs, ep))
        )
    )


def beta2_h(krx, kry, ki, th, ph, ths, phs, ep):
    return (
        -2
        * kiz(ki, th)
        * (ki ** 2 - ep * ki ** 2)
        / (
            (ep * ksz(ki, ths, phs) + ktz(ki, ths, phs, ep))
            * (kiz(ki, th) + kiz1(ki, th, ph, ep))
        )
        * (
            s_rs(krx, kry, ki, th, ths, ths, phs)
            * c_ri(krx, kry, ki, th, ph, ths, phs)
            * (ep * ki + ktz(ki, ths, phs, ep) * kappa1(krx, kry, ki, ep) / ki)
            - c_rs(krx, kry, ki, th, ph, ths, phs)
            * s_ri(krx, kry, ki, th, ph, ths, phs)
            * (ep * ki + ktz(ki, ths, phs, ep) * kappa2(krx, kry, ki, ep) / ki)
            + s_ri(krx, kry, ki, th, ph, ths, phs)
            * ep
            * kperp_s(ki, ths, phs)
            * kappa3(krx, kry, ki, ep)
            / ki
            + 0.5
            * s_si(krx, kry, ki, th, ph, ths, phs)
            * (ep * ki - ktz(ki, ths, phs, ep) * kiz1(ki, th, ph, ep) / ki)
        )
    )


def gamma2_h(krx, kry, ki, th, ph, ths, phs, ep):
    return (
        -2
        * kiz(ki, th)
        * (ki ** 2 - ep * ki ** 2)
        / (
            (ksz(ki, ths, phs) + ktz(ki, ths, phs, ep))
            * (kiz(ki, th) + kiz1(ki, th, ph, ep))
        )
        * (
            c_rs(krx, kry, ki, th, ph, ths, phs)
            * c_ri(krx, kry, ki, th, ph, ths, phs)
            * (kappa1(krx, kry, ki, ep) - ksz(ki, ths, phs))
            + s_rs(krx, kry, ki, th, ths, ths, phs)
            * s_ri(krx, kry, ki, th, ph, ths, phs)
            * (-ksz(ki, ths, phs) + kappa2(krx, kry, ki, ep))
            + 0.5
            * c_si(ki, th, ph, ths, phs)
            * (kiz1(ki, th, ph, ep) + ksz(ki, ths, phs))
        )
    )


def delta2_h(krx, kry, ki, th, ph, ths, phs, ep):
    return (
        -2
        * kiz(ki, th)
        * (ki ** 2 - ep * ki ** 2)
        * sqrt(ep)
        / (
            (ep * ksz(ki, ths, phs) + ktz(ki, ths, phs, ep))
            * (kiz(ki, th) + kiz1(ki, th, ph, ep))
        )
        * (
            s_rs(krx, kry, ki, th, ths, ths, phs)
            * c_ri(krx, kry, ki, th, ph, ths, phs)
            * (ki - ksz(ki, ths, phs) * kappa1(krx, kry, ki, ep) / ki)
            - c_rs(krx, kry, ki, th, ph, ths, phs)
            * s_ri(krx, kry, ki, th, ph, ths, phs)
            * (ki - ksz(ki, ths, phs) * kappa2(krx, kry, ki, ep) / ki)
            + s_ri(krx, kry, ki, th, ph, ths, phs)
            * kperp_s(ki, ths, phs)
            * kappa3(krx, kry, ki, ep)
            / ki
            + 0.5
            * s_si(krx, kry, ki, th, ph, ths, phs)
            * (ki - ksz(ki, ths, phs) * kiz1(ki, th, ph, ep) / ki)
        )
    )


# MODO TM #


def alpha2_v(krx, kry, ki, th, ph, ths, phs, ep):
    return (
        -2
        * kiz(ki, th)
        * (ki ** 2 - ep * ki ** 2)
        / (
            (ksz(ki, ths, phs) + ktz(ki, ths, phs, ep))
            * (ep * kiz(ki, th) + kiz1(ki, th, ph, ep))
        )
        * (
            -c_rs(krx, kry, ki, th, ph, ths, phs)
            * s_ri(krx, kry, ki, th, ph, ths, phs)
            * (kiz1(ki, th, ph, ep) / ki)
            * (kappa1(krx, kry, ki, ep) + ktz(ki, ths, phs, ep))
            + s_rs(krx, kry, ki, th, ths, ths, phs)
            * c_ri(krx, kry, ki, th, ph, ths, phs)
            * (kiz1(ki, th, ph, ep) / ki)
            * (ktz(ki, ths, phs, ep) + kappa2(krx, kry, ki, ep))
            - s_rs(krx, kry, ki, th, ths, ths, phs)
            * (ep * kperp_i(ki, th, ph) / ki)
            * kappa3(krx, kry, ki, ep)
            - 0.5
            * s_si(krx, kry, ki, th, ph, ths, phs)
            * (ep * ki - ktz(ki, ths, phs, ep) * kiz1(ki, th, ph, ep) / ki)
        )
    )


def beta2_v(krx, kry, ki, th, ph, ths, phs, ep):
    return (
        -2
        * kiz(ki, th)
        * (ki ** 2 - ep * ki ** 2)
        / (
            (ep * ksz(ki, ths, phs) + ktz(ki, ths, phs, ep))
            * (ep * kiz(ki, th) + kiz1(ki, th, ph, ep))
        )
        * (
            -s_rs(krx, kry, ki, th, ths, ths, phs)
            * s_ri(krx, kry, ki, th, ph, ths, phs)
            * (kiz1(ki, th, ph, ep) / ki)
            * (ep * ki + ktz(ki, ths, phs, ep) * kappa1(krx, kry, ki, ep) / ki)
            - c_rs(krx, kry, ki, th, ph, ths, phs)
            * c_ri(krx, kry, ki, th, ph, ths, phs)
            * (kiz1(ki, th, ph, ep) / ki)
            * (ep * ki + ktz(ki, ths, phs, ep) * kappa2(krx, kry, ki, ep) / ki)
            + c_rs(krx, kry, ki, th, ph, ths, phs)
            * (ep * kperp_i(ki, th, ph) * ktz(ki, ths, phs, ep) / ki ** 2)
            * kappa3(krx, kry, ki, ep)
            + (ep * kperp_s(ki, ths, phs) * kappa3(krx, kry, ki, ep) / ki ** 2)
            * (
                kiz1(ki, th, ph, ep) * c_ri(krx, kry, ki, th, ph, ths, phs)
                + sqrt(ki ** 2 - krx ** 2 - kry ** 2 + 0j)
                * kperp_i(ki, th, ph)
                * kappa1(krx, kry, ki, ep)
                / ki ** 2
            )
            + 0.5
            * c_si(ki, th, ph, ths, phs)
            * (ep * kiz1(ki, th, ph, ep) - ep * ktz(ki, ths, phs, ep))
        )
    )


def gamma2_v(krx, kry, ki, th, ph, ths, phs, ep):
    return (
        -2
        * kiz(ki, th)
        * (ki ** 2 - ep * ki ** 2)
        / (
            (ksz(ki, ths, phs) + ktz(ki, ths, phs, ep))
            * (ep * kiz(ki, th) + kiz1(ki, th, ph, ep))
        )
        * (
            -c_rs(krx, kry, ki, th, ph, ths, phs)
            * s_ri(krx, kry, ki, th, ph, ths, phs)
            * (kiz1(ki, th, ph, ep) / ki)
            * (kappa1(krx, kry, ki, ep) - ksz(ki, ths, phs))
            + s_rs(krx, kry, ki, th, ths, ths, phs)
            * c_ri(krx, kry, ki, th, ph, ths, phs)
            * (kiz1(ki, th, ph, ep) / ki)
            * (-ksz(ki, ths, phs) + kappa2(krx, kry, ki, ep))
            + -s_rs(krx, kry, ki, th, ths, ths, phs)
            * (ep * kperp_i(ki, th, ph) / ki)
            * kappa3(krx, kry, ki, ep)
            - 0.5
            * s_si(krx, kry, ki, th, ph, ths, phs)
            * (ep * ki + ksz(ki, ths, phs) * kiz1(ki, th, ph, ep) / ki)
        )
    )


def delta2_v(krx, kry, ki, th, ph, ths, phs, ep):
    return (
        -2
        * kiz(ki, th)
        * (ki ** 2 - ep * ki ** 2)
        * sqrt(ep)
        / (
            (ep * ksz(ki, ths, phs) + ktz(ki, ths, phs, ep))
            * (ep * kiz(ki, th) + kiz1(ki, th, ph, ep))
        )
        * (
            s_rs(krx, kry, ki, th, ths, ths, phs)
            * s_ri(krx, kry, ki, th, ph, ths, phs)
            * (kiz1(ki, th, ph, ep) / ki)
            * (ki - ksz(ki, ths, phs) * kappa1(krx, kry, ki, ep) / ki)
            - c_rs(krx, kry, ki, th, ph, ths, phs)
            * c_ri(krx, kry, ki, th, ph, ths, phs)
            * (kiz1(ki, th, ph, ep) / ki)
            * (ki - ksz(ki, ths, phs) * kappa2(krx, kry, ki, th, ph, ths, phs, ep) / ki)
            - c_rs(krx, kry, ki, th, ph, ths, phs)
            * (ep * kperp_i(ki, th, ph) * ksz(ki, ths, phs) / ki ** 2)
            * kappa3(krx, kry, ki, th, ph, ths, phs, ep)
            + (
                kperp_s(ki, ths, phs)
                * kappa3(krx, kry, ki, th, ph, ths, phs, ep)
                / ki ** 2
            )
            * (
                kiz1(ki, th, ph, ep) * c_ri(krx, kry, ki, th, ph, ths, phs)
                + sqrt(ki ** 2 - krx ** 2 - kry ** 2)
                * kperp_i(ki, th, ph)
                * kappa1(krx, kry, ki, ep)
                / ki ** 2
            )
            + 0.5
            * c_si(ki, th, ph, ths, phs)
            * (
                kiz1(ki, th, ph, ep)
                + ep * ksz(ki, ths, phs) * kiz1(ki, th, ph, ep) / ki
            )
        )
    )


## sección eficaz a primer orden
def HH_O1(ki, th, ph, ths, phs, ep, s, l, k1, k2):
    return (
        4
        * np.pi
        * ki ** 2
        * cos(th) ** 2
        * w_gauss(s, l, k1, k2)
        * (abs(alpha1_h(ki, th, ph, ths, phs, ep)) ** 2)
    )


def VV_O1(ki, th, ph, ths, phs, ep, s, l, k1, k2):
    return (
        4
        * np.pi
        * ki ** 2
        * cos(th) ** 2
        * w_gauss(s, l, k1, k2)
        * (abs(beta1_v(ki, th, ph, ths, phs, ep)) ** 2)
    )


def VH_O1(ki, th, ph, ths, phs, ep, s, l, k1, k2):
    return (
        4
        * np.pi
        * ki ** 2
        * cos(th) ** 2
        * w_gauss(s, l, k1, k2)
        * (abs(beta1_h(ki, th, ph, ths, phs, ep)) ** 2)
    )


def HV_O1(ki, th, ph, ths, phs, ep, s, l, k1, k2):
    return (
        4
        * np.pi
        * ki ** 2
        * cos(th) ** 2
        * w_gauss(s, l, k1, k2)
        * (abs(alpha1_v(ki, th, ph, ths, phs, ep)) ** 2)
    )


## Segundo Orden
## hh
def aux2_hh(krx, kry, ki, th, ph, ths, phs, ep, s, l):
    return (
        w_gauss(s, l, krx - kix(ki, th, ph), kry - kiy(ki, th, ph))
        * w_gauss(s, l, ksx(ki, ths, phs) - krx, ksy(ki, ths, phs) - kry)
        * (
            abs(alpha2_h(krx, kry, ki, th, ph, ths, phs, ep)) ** 2
            + alpha2_h(krx, kry, ki, th, ph, ths, phs, ep)
            * np.conj(
                alpha2_h(
                    kix(ki, th, ph) + ksx(ki, ths, phs) - krx,
                    kiy(ki, th, ph) + ksy(ki, ths, ph) - kry,
                    ki,
                    th,
                    ph,
                    ths,
                    phs,
                    ep,
                )
            )
        )
    )


## vv
def aux2_vv(krx, kry, ki, th, ph, ths, phs, ep, s, l):
    return (
        w_gauss(s, l, krx - kix(ki, th, ph), kry - kiy(ki, th, ph))
        * w_gauss(s, l, ksx(ki, ths, phs) - krx, ksy(ki, ths, phs) - kry)
        * (
            abs(beta2_v(krx, kry, ki, th, ph, ths, phs, ep)) ** 2
            + beta2_v(krx, kry, ki, th, ph, ths, phs, ep)
            * np.conj(
                beta2_v(
                    kix(ki, th, ph) + ksx(ki, ths, phs) - krx,
                    kiy(ki, th, ph) + ksx(ki, ths, phs) - kry,
                    ki,
                    th,
                    ph,
                    ths,
                    phs,
                    ep,
                )
            )
        )
    )


## vh
def aux2_vh(krx, kry, ki, th, ph, ths, phs, ep, s, l):
    return (
        w_gauss(s, l, krx - kix(ki, th, ph), kry - kiy(ki, th, ph))
        * w_gauss(s, l, ksx(ki, ths, phs) - krx, ksy(ki, ths, phs) - kry)
        * (
            abs(beta2_h(krx, kry, ki, th, ph, ths, phs, ep)) ** 2
            + beta2_h(krx, kry, ki, th, ph, ths, phs, ep)
            * np.conj(
                beta2_h(
                    kix(ki, th, ph) + ksx(ki, ths, phs) - krx,
                    kiy(ki, th, ph) + ksy(ki, ths, phs) - kry,
                    ki,
                    th,
                    ph,
                    ths,
                    phs,
                    ep,
                )
            )
        )
    )


## HV
def aux2_hv(krx, kry, ki, th, ph, ths, phs, ep, s, l):
    return (
        w_gauss(s, l, krx - kix(ki, th, ph), kry - kiy)
        * w_gauss(s, l, ksx(ki, ths, phs) - krx, ksy(ki, ths, phs) - kry)
        * (
            abs(alpha2_v(krx, kry, ki, th, ph, ths, phs, ep)) ** 2
            + alpha2_v(krx, kry, ki, th, ph, ths, phs, ep)
            * np.conj(
                alpha2_v(
                    kix(ki, th, ph) + ksx(ki, ths, phs) - krx,
                    kiy(ki, th, ph) + ksy(ki, ths, phs) - kry,
                    ki,
                    th,
                    ph,
                    ths,
                    phs,
                    ep,
                )
            )
        )
    )

## HH*VV
def aux2_hh_vv(krx, kry, ki, th, ph, ths, phs, ep, s, l):
    return (
        w_gauss(s, l, krx - kix(ki, th, ph), kry - kiy)
        * w_gauss(s, l, ksx(ki, ths, phs) - krx, ksy(ki, ths, phs) - kry)
        * (
            alpha2_h(krx, kry, ki, th, ph, ths, phs, ep) *
            np.conj(beta2_v(krx, kry, ki, th, ph, ths, phs, ep))
            + alpha2_h(krx, kry, ki, th, ph, ths, phs, ep)
            * np.conj(
                beta2_v(
                    kix(ki, th, ph) + ksx(ki, ths, phs) - krx,
                    kiy(ki, th, ph) + ksy(ki, ths, phs) - kry,
                    ki,
                    th,
                    ph,
                    ths,
                    phs,
                    ep,
                )
            )
        )
    ) 

## HV*VV
def aux2_hv_vv(krx, kry, ki, th, ph, ths, phs, ep, s, l):
    return (
        w_gauss(s, l, krx - kix(ki, th, ph), kry - kiy)
        * w_gauss(s, l, ksx(ki, ths, phs) - krx, ksy(ki, ths, phs) - kry)
        * (
            beta2_v(krx, kry, ki, th, ph, ths, phs, ep) *
            np.conj(alpha2_v(krx, kry, ki, th, ph, ths, phs, ep))
            + beta2_v(krx, kry, ki, th, ph, ths, phs, ep)
            * np.conj(
                alpha2_v(
                    kix(ki, th, ph) + ksx(ki, ths, phs) - krx,
                    kiy(ki, th, ph) + ksy(ki, ths, phs) - kry,
                    ki,
                    th,
                    ph,
                    ths,
                    phs,
                    ep,
                )
            )
        )
    ) 

## HV*HH
def aux2_hv_hh(krx, kry, ki, th, ph, ths, phs, ep, s, l):
    return (
        w_gauss(s, l, krx - kix(ki, th, ph), kry - kiy)
        * w_gauss(s, l, ksx(ki, ths, phs) - krx, ksy(ki, ths, phs) - kry)
        * (
            alpha2_h(krx, kry, ki, th, ph, ths, phs, ep) *
            np.conj(alpha2_v(krx, kry, ki, th, ph, ths, phs, ep))
            + alpha2_h(krx, kry, ki, th, ph, ths, phs, ep)
            * np.conj(
                alpha2_v(
                    kix(ki, th, ph) + ksx(ki, ths, phs) - krx,
                    kiy(ki, th, ph) + ksy(ki, ths, phs) - kry,
                    ki,
                    th,
                    ph,
                    ths,
                    phs,
                    ep,
                )
            )
        )
    )    