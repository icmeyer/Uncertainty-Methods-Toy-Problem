import ctypes
from ctypes import (c_int, c_double, POINTER)
import numpy as np
from numpy.ctypeslib import ndpointer
import matplotlib.pyplot as plt

_doppler = ctypes.CDLL('./broaden_module_ext/libdoppler.so')
# _doppler = ctypes.CDLL('./libdoppler.so')

_doppler.broaden_c.restypes = None
_doppler.broaden_c.argtypes = (c_int, ndpointer(c_double), ndpointer(c_double),
                               ndpointer(c_double), c_double, c_int)

def broaden(energy, xs, T, A):
    n = len(xs)
    broadened = np.empty(n, dtype=np.float64)
    xs = np.array(xs, dtype=np.float64)
    energy =  np.array(energy, dtype=np.float64)
    c_result = _doppler.broaden_c(c_int(n), broadened, energy, xs, 
                                   c_double(T), c_int(A))

    return broadened

def broaden_extend(energy, xs, T, A):
    fractional_extent = 0.10
    energy_ext, xs_ext, orig_inds = extend(energy, xs, fractional_extent)
    n = len(xs_ext)
    broadened_ext = np.empty(n, dtype=np.float64)
    xs_ext = np.array(xs_ext, dtype=np.float64)
    energy =  np.array(energy_ext, dtype=np.float64)
    # print('broadening')
    c_result = _doppler.broaden_c(c_int(n), broadened_ext, energy_ext, xs_ext, 
                                   c_double(T), c_int(A))
    broadened = broadened_ext[orig_inds[0]:orig_inds[1]]

    return broadened

def extend(energy, xs, fractional_extent):
    """
    Extend the cross section using the first derivative approximation 
    at each end

    """
    E_low = energy[0]
    E_high = energy[-1]
    xs_low = xs[0]
    xs_high = xs[-1]
    delta_E = energy[1] - energy[0]
    E_range = E_high - E_low

    new_E_low = E_low - fractional_extent*E_range
    new_E_high = E_high + fractional_extent*E_range
    
    high_slope = (xs[-3] - 4*xs[-2] + 3*xs[-1])/(2*delta_E)
    low_slope = (-3*xs[0] + 4*xs[1] - xs[2])/(2*delta_E)

    low_xs_additions = []
    low_E_additions = []
    cur_E = E_low
    while (cur_E>new_E_low) & (cur_E>0):
        cur_E = cur_E - delta_E
        low_E_additions.insert(0, cur_E)
        low_xs_additions.insert(0, xs_low + (cur_E - E_low)*low_slope)

    high_xs_additions = []
    high_E_additions = []
    cur_E = E_high
    while cur_E < new_E_high:
        cur_E = cur_E + delta_E
        high_E_additions.append(cur_E)
        high_xs_additions.append(xs_high + (cur_E - E_high)*high_slope)

    orig_start_index = len(low_E_additions)
    orig_end_index = orig_start_index + len(energy)
    return_energy = np.concatenate([low_E_additions, energy, high_E_additions])
    return_xs = np.concatenate([low_xs_additions, xs, high_xs_additions])

    # Test by plotting
    # plt.plot(energy, xs)
    # plt.plot(low_E_additions, low_xs_additions, 'g')
    # plt.plot(high_E_additions, high_xs_additions, 'g')
    # plt.plot(return_energy[orig_start_index:orig_end_index],return_xs[orig_start_index:orig_end_index]+.1)
    # plt.show()

    return return_energy, return_xs, [orig_start_index, orig_end_index]
    


if __name__=='__main__':
    # Import example xs
    example = np.loadtxt('example_xs', delimiter=',')
    energy = example[:,0]
    xs = example[:,1]
    # energy = np.linspace(20, 80, 20)
    # xs = np.sin(energy/5)
    T = 900
    A = 238
    broadened =  broaden_extend(energy, xs, T, A)
    # broadened =  broaden(energy, xs, T, A)

    plt.plot(energy, xs)
    plt.plot(energy, broadened)
    plt.show()
