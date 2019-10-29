import numpy as np
import time
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


class broadened_xs_generator:
    """
    Class to contain 0K cross section data and produce xs at new temperatures

    Parameters
    ----------
    energy
    cross section
    temperature
    atomic weight ratio
    """
    def __init__(self, energy, xs, T, A):
        self.energy_grid = energy # Original energy grid
        self.orig_xs = xs # Original cross section
        self.T = T # Change in temperature
        self.alpha = A/(k*T)
        self.x_vector = np.sqrt(self.alpha * energy)
        self.x_square = self.x_vector**2

    def single_xs_value(self, energy, xs, x_local, x_local_sq, y):
        # Evaluate Constants
        a_vector = x_local - y 
        y_sq = y*y
        y_inv = 1/y
        y_inv_sq = y_inv /y
    
        S_k = (xs[1:] - xs[:-1])/(x_local_sq[1:] - x_local_sq[:-1])
    
        # Build F vector
        # F is indexed by [point, order]
        Fs = []
        for i in range(len(xs)):
            Fs.append(calculate_F(a_vector[i]))
    
        Fs = np.array(Fs)
        F_diffs = Fs[1:, :] - Fs[:-1, :]
    
        integral_sum = 0 
        for i in range(len(xs) - 1):
            Ak = y_inv_sq*F_diffs[i,2] + 2.0*y_inv*F_diffs[i,1] + F_diffs[i,0];
            Bk = y_inv_sq*F_diffs[i,4] + 4.0*y_inv*F_diffs[i,3] + \
                 6.0*F_diffs[i,2] + 4.0*y*F_diffs[i,1] + y_sq*F_diffs[i,0]
            slope  = S_k[i]
            integral_sum += Ak*(xs[i] - slope*x_local[i]**2) + slope*Bk;
            
        # Not sure about this minus sign
        sigma = -integral_sum
        return sigma


    def broadened_xs(self, new_energies):
        broadened = np.zeros_like(new_energies)
        for i in range(len(new_energies)):
            y = np.sqrt(self.alpha*new_energies[i])

            # Determine window for xs
            a_vector = self.x_vector - y
            low_index = np.searchsorted(a_vector, -4, side='left')
            if low_index != 0:
                low_index -= 1

            a_top = -4
            high_index = low_index
            while (a_top <= 4) and (high_index < len(self.orig_xs)-1):
                high_index += 1
                a_top = a_vector[high_index]
            high_index += 1

            # print(a_vector[low_index:high_index])
            i_energies = self.energy_grid[low_index:high_index]
            i_xs = self.orig_xs[low_index:high_index]
            i_x = self.x_vector[low_index:high_index]
            i_x_sq = self.x_square[low_index:high_index]
            value = self.single_xs_value(i_energies, i_xs, i_x, i_x_sq, y)
            inverse_value = self.single_xs_value(i_energies, i_xs, i_x, i_x_sq, -y)
            broadened[i] = value - inverse_value
        return broadened

    def broaden_use_trap(self, new_energies):
        broadened = np.zeros_like(new_energies)
        for i in range(len(new_energies)):
            y = np.sqrt(self.alpha*new_energies[i])
            broadened[i] = self.sigma_T_trap(y)
        return broadened
    
    def sigma_T_trap(self, y):
        # Evaluate Constants
        a_vector = self.x_vector - y
        b_vector = self.x_vector + y
        y_inv_sq = 1/y**2
    
        integrand = self.x_square * self.orig_xs * \
                    (np.exp(-a_vector**2)-np.exp(-b_vector**2))
        integral = np.trapz(integrand, x=self.x_vector)
        sigma = y_inv_sq*SQRT_PI_INV*integral
        return sigma
    
if __name__=='__main__':
    # Test capability and compare against direct integration
    min_E = 1e-6
    max_E = 25
    n_points = 100
    # energy = np.linspace(min_E, max_E, n_points)
    energy = np.logspace(np.log10(min_E), np.log10(max_E), n_points)
    xs = np.abs(3/((energy - 6)**2 + 0.0001)) + 10
    # new_energies = np.logspace(-4, 1.1, 1000)
    new_energies = energy
    # new_energies = np.linspace(min_E, max_E, 1000)
    T=1e5
    A=238

    start = time.perf_counter()

    xs_object = broadened_xs_generator(energy, xs, T, A)
    broadened = xs_object.broadened_xs(new_energies)

    end = time.perf_counter()
    new_time = end - start

    start = time.perf_counter()

    broadened_trap = xs_object.broaden_use_trap(new_energies)

    end = time.perf_counter()
    old_time = end - start
    
    fig = plt.figure()
    
    ax1 = fig.add_subplot(211)
    ax1.loglog(energy, xs, label='xs')
    ax1.loglog(new_energies, broadened, label='sigma1')
    ax1.loglog(new_energies, broadened_trap, label='trapezoidal')
    ax1.set_title('SIGMA1 time: {:f} s \nDirect time: {:f} s\n Number of points: {:d}'.format(new_time, old_time, n_points))
    ax1.legend()

    ax2 = fig.add_subplot(212, sharex=ax1)
    rel_error = np.abs(broadened - broadened_trap)/broadened_trap
    ax2.loglog(new_energies, rel_error, label='rel error')
    ax2.legend()

    # Test direct on different amounts of background xs
    fig2 = plt.figure()
    ax1 = fig2.add_subplot(111)

    ranges = 10
    import matplotlib.pylab as pl
    colors =pl.cm.jet(np.linspace(0,1,2*ranges))
    for i in range(ranges):
        min_order = 0-i
        max_order = 1+i
        energy = np.logspace(min_order, max_order, 10000)
        xs = np.abs(3/((energy - 6)**2 + 0.0001)) + 10
        xs_object = broadened_xs_generator(energy, xs, T, A)
        broadened_trap = xs_object.broaden_use_trap(energy)
        ax1.loglog(energy, xs, label='0K: Range {:d}-{:d}'.format(min_order, max_order), color=colors[2*i])
        ax1.loglog(energy, broadened_trap, label='{:f}K: Range {:d}-{:d}'.format(T, 0-i, 1+i), color=colors[2*i+1])
    ax1.legend()


    # fig.save('direct_integration_comparison.pdf')
    plt.show()
