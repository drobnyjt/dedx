import numpy as np
import matplotlib.pyplot as plt

Q = 1.602E-19
PI = 3.14159
AMU = 1.66E-27
ANGSTROM = 1E-10
MICRON = 1E-6
CM = 1E-2
EPS0 = 8.85E-12
A0 = 0.52918E-10
K = 1.11265E-10
ME = 9.11E-31
SQRTPI = 1.77245385
SQRT2PI = 2.506628274631
C = 300000000.
BETHE_BLOCH_PREFACTOR = 4.*PI*(Q*Q/(4.*PI*EPS0))*(Q*Q/(4.*PI*EPS0))/ME/C/C
LINDHARD_SCHARFF_PREFACTOR = 1.212*ANGSTROM*ANGSTROM*Q

def W(E0, E, Ma):
    K = E + 2*Ma*C**2
    B = E0**2 - 2*ME*C**2*E0 - 2*E*K
    A = E + 2*ME*C**2 + K - 2*E0
    G = A**2 - 4*E*K
    D = A*B + 2*E*K*(E + K)
    F = B**2 - 4*E**2*K**2

    W_plus = (-D + np.sqrt(D**2 - F*G))/G
    W_minus = (-D - np.sqrt(D**2 - F*G))/G

    return W_plus, W_minus

def S_BV(Za, Zb, Ma, n, E, CK=1.):
    v = np.sqrt(2*E/Ma)
    beta = v/C

    if Zb < 13:
        I0 = 12 + 7/Zb
    else:
        I0 = 9.76 + 58.5*Zb**(-1.19)

    I = Zb*I0*Q

    if Zb < 3:
        B = 100.*Za/Zb
    else:
        B = 5.

    prefactor = BETHE_BLOCH_PREFACTOR*Zb*Za*Za/beta/beta
    eb = 2.*ME*v*v/I
    S_BB_BV = prefactor*np.log(eb + 1. + B/eb)*n
    S_BB = prefactor*np.log(eb)*n
    S_LS = CK*LINDHARD_SCHARFF_PREFACTOR*(Za**(7./6.)*Zb)/(Za**(2./3.) + Zb**(2./3.))**(3./2.)*np.sqrt(E/Ma*AMU/Q)*n
    S = 1./(1./S_LS + 1./S_BB_BV)
    return S_BB, S_LS, S

def S_MV(Za, Ma, n, E, E0s, alpha0s):
    beta = np.sqrt(1. - (1. + E/Ma/C**2)**(-2.))

    prefactor = Za**2/(PI*A0*ME*C**2*beta**2)

    sum = 0.
    for E0, alpha0 in zip(E0s, alpha0s):
        E0 *= Q
        alpha0 *= Q**2
        W_plus, W_minus = W(E0, E, Ma)
        if np.isnan(W_minus) or np.isnan(W_plus):
            continue
        #print(W_plus, W_minus)
        #print((2.*Ma*C**2 + W_plus - E0)/(2.*Ma*C**2 + W_minus - E0))
        term_1 = (2.*Ma*C**2 - ME*C**2)*np.log((2.*Ma*C**2 + W_plus - E0)/(2.*Ma*C**2 + W_minus - E0))
        #print((W_plus - E0)/(W_minus - E0))
        if (W_plus - E0)/(W_minus - E0) > 0:
            term_2 = ME*C**2*np.log((W_plus - E0)/(W_minus - E0))
        else:
            term_2 = 0.
        sum += alpha0/(ME*C**2)*(term_1 + term_2)

    return prefactor*sum

def main():
    Ma = 1*AMU
    Za = 1
    Zb = 29
    Mb = 63.54*AMU
    n = 6.026E28
    energies = np.logspace(-3, 5, 500)*1E6*Q
    lw = 3

    S_MV_list = []
    S_BV_list = []
    S_LS_list = []
    S_BB_list = []
    for energy in energies:
        S_MV_ = S_MV(Za, Ma, n, energy, [1520, 150.7, 111, 110, 15.1], [158, 164, 260, 517, 385])
        S_MV_list .append(S_MV_/Q*1E-10)
        S_BB_, S_LS_, S_BV_ = S_BV(Za, Zb, Ma, n, energy, CK=1.)
        S_BV_list.append(S_BV_/Q*1E-10)
        S_LS_list.append(S_LS_/Q*1E-10)
        S_BB_list.append(S_BB_/Q*1E-10)

    plt.semilogx(energies/1E6/Q, S_MV_list, '-.', linewidth=lw)

    data = np.genfromtxt('H_Al.dat')
    E_PSTAR = data[:,0]
    S_PSTAR = data[:,1]*270.*1E6*1E-10
    S_BV_list = np.array(S_BV_list)
    S_LS_list = np.array(S_LS_list)
    S_BB_list = np.array(S_BB_list)
    S_MV_list = np.array(S_MV_list)

    plt.semilogx(E_PSTAR, S_PSTAR, linewidth=lw)
    plt.semilogx(energies/1E6/Q, S_BV_list, '--', linewidth=lw)
    plt.semilogx(energies/1E6/Q, S_LS_list, ':', linewidth=lw)
    plt.semilogx(energies/1E6/Q, S_BB_list, '.', linewidth=lw)

    #plt.semilogx(energies/1E6/Q, 1./(1./S_LS_list + 1./S_MV_list), '*')


    plt.axis([0., 1E5, 0., 1.2*np.max(S_BV_list)])
    plt.ylabel('Stopping Power [eV/A]')
    plt.xlabel('Incident Energy [MeV]')

    plt.legend(['Medvedev-Volkov', 'PSTAR', 'Biersack-Varelas', 'Lindhard-Scharff', 'Pure Bethe'])


    plt.show()

if __name__ == '__main__':
    main()
