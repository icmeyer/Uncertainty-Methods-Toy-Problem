import numpy as np
from scipy.constants import physical_constants
from scipy.special import erf, erfc

import matplotlib.pyplot as plt

k = physical_constants['Boltzmann constant in eV/K'][0] 

PI =  np.pi
SQRT_PI = np.sqrt(PI)
SQRT_PI_INV = 1 / SQRT_PI


def calculate_F(a):
    F = np.empty([5,])
    F[0] = 0.5*erfc(a);
    F[1] = 0.5*SQRT_PI_INV*np.exp(-a*a);
    F[2] = 0.5*F[0] + a*F[1];
    F[3] = F[1]*(1.0 + a*a);
    F[4] = 0.75*F[0] + F[1]*a*(1.5 + a*a);
    return F


def sigma_T(energy, xs, T, A, y):
    # Evaluate Constants
    alpha = A/(k*T)
    x_vector = np.sqrt(alpha * energy)
    x_square = x_vector**2
    a_vector = x_vector - y 
    y_sq = y*y
    y_inv = 1/y
    y_inv_sq = y_inv /y

    S_k = (xs[1:] - xs[:-1])/(x_square[1:] - x_square[:-1])

    # Build F vector
    # F is indexed by [point, order]
    Fs = []
    for i in range(len(xs)):
        Fs.append(calculate_F(a_vector[i]))

    Fs = np.array(Fs)
    F_diffs = Fs[1:, :] - Fs[:-1, :]

    # print('Fs')
    # print(Fs)
    # print('Fs k + 1')
    # print(Fs[1:, :])
    # print('Fs k ')
    # print(Fs[:-1, :])
    # print('F diff')
    # print(F_diffs)
    # print(F_diffs[:, 2])

    integral_sum = 0 
    for i in range(len(xs) - 1):
        Ak = y_inv_sq*F_diffs[i,2] + 2.0*y_inv*F_diffs[i,1] + F_diffs[i,0];
        Bk = y_inv_sq*F_diffs[i,4] + 4.0*y_inv*F_diffs[i,3] + \
             6.0*F_diffs[i,2] + 4.0*y*F_diffs[i,1] + y_sq*F_diffs[i,0]
        slope  = S_k[i]
        integral_sum += Ak*(xs[i] - slope*x_vector[i]**2) + slope*Bk;
        
    # Not sure about this minus sign
    sigma = -integral_sum
    return sigma


def broaden(energy, xs, T, A, new_energies):
    broadened = np.zeros_like(new_energies)
    for i in range(len(new_energies)):
        alpha = A/(k*T)
        y = np.sqrt(alpha*new_energies[i])
        broadened[i] = sigma_T(energy, xs, T, A, y) 
    return broadened


if __name__=='__main__':
    # n_points = int(1e2)
    n_points = 1000
    min_E = 1e-6
    max_E = 25
    energy = np.linspace(min_E, max_E, n_points)
    new_energies = np.linspace(min_E, max_E, 100)
    # new_energies = np.linspace(4, 8, 100)
    xs = np.abs(3/((energy - 6)**2 + 0.0001)) + 10

    T = 1e4
    A = 238
    
    broadened = broaden(energy, xs, T, A, new_energies)
    print(broadened[0:20])
    sigma =  sigma_T(energy, xs, T, A, E_y = 6)
    
    plt.loglog(energy, xs, label='xs')
    plt.loglog(new_energies, broadened, label='broadened')
    plt.legend()
    plt.show()
