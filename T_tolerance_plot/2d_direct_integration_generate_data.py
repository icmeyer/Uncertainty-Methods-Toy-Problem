# Building a union grid which will evaluate all functions to
# a given tolerance
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import copy
import time
import pickle

from openmc.data.function import Tabulated1D


# Need some functions from parent directory
import os, sys
problem_dir = os.path.dirname(os.getcwd())
sys.path.insert(0, problem_dir)

# import openmc.data.grid as grid
from openmc_functions import grid
from slbw import * #exact_Σγ, exact_Real_SLBW, zeroK_exact_Re_SLBW_χψ_Σγ, exact_dΣγ_dΓ, z_space_Σγ, analytic_Σγ, analytic_dΣγ_dΓ, multipole_Σ, multipole_dΣ_dΓ, exact_poles_and_residues , exact_poles_and_residues_differentials_dΠ_dΓ
from data import a_U238, ρ0_U238, ρ0, μ_E0_U238, μ_Γn_U238, μ_Γγ_U238, μ_Γ_U238, cov_Γ_U238 
from vector_fit import VF_algorithm, rational_function

from doppler_object import broadened_xs_generator
start = time.perf_counter()

# Data
A = 238
a = 0 
Γ = μ_Γ_U238
b = 2*np.pi*Γ[1]*Γ[2]/(ρ0**2*Γ[0]**0.5*(Γ[1]+ Γ[2]))
r = a - 1j*b
Π = exact_poles_and_residues(μ_Γ_U238)

# Functions for tolerance checking
def one_input_exact_Σγ_0K(E):
    return exact_Σγ(E, μ_Γ_U238)

# "Hyperparameters"
# refined value in parantheses
# Number of 0K energy groups (1e5)
N_g = int(1e5)
# Tolerance for 0K energy grid (1e-10)
tol_0K = 1e-8
# Number of groups at T (1e3)
N_sub = int(1e3)
# Tolerance at T (1e-4)
tol_for_plot = 1e-4
# Number of temparature (40)
N_Ts = 40

# Find 0K grid
E_g_start = np.logspace(-9, 3, N_g)
# subset_low = np.searchsorted( E_g_start, 1e-4)
# subset_high = np.searchsorted(E_g_start, 1e2)
E_subset = np.logspace(-4, 2, N_sub)

E_g_0K, exact_0K = grid.linearize(E_g_start, one_input_exact_Σγ_0K, tolerance=tol_0K)

# plt.loglog(E_g_0K, exact_0K)
# plt.show()
print('Shape of 0K grid', E_g_0K.shape)


def one_input_generator_direct_integration(T):
    broadening_object = broadened_xs_generator(E_g_0K, exact_0K, T, A)
    def one_input_direct_integration(E):
        return broadening_object.broaden_use_trap([E])
    return one_input_direct_integration

def one_input_generator_sigma1(T):
    broadening_object = broadened_xs_generator(E_g_0K, exact_0K, T, A)
    def one_input_sigma1(E):
        return broadening_object.broadened_xs([E])
    return one_input_sigma1

def one_input_generator_psichi(T):
    def one_input_psichi(E):
        return TK_approx_SLBW_Faddeeva_Σγ(E, Γ, r ,T)
    return one_input_psichi
    
def one_input_generator_mp(T):
    def one_input_mp(E):
        z = E**0.5
        return approx_multipole_Doppler_Σ(z, Π, T) 
    return one_input_mp

def arr_conv(array):
    return np.asarray(array).flatten()

Ts = np.logspace(0, 6, N_Ts)
E_grids = []
Tab1D_dict = {}
# Test at 10K
for T in Ts:
    # di_E_grid, xs_di = grid.linearize(E_subset, one_input_generator_direct_integration(T),
    #                                   tolerance=tol_for_plot)
    di_E_grid, xs_di = grid.linearize(E_subset, one_input_generator_sigma1(T),
                                      tolerance=tol_for_plot)
    pc_E_grid, xs_pc = grid.linearize(E_subset, one_input_generator_psichi(T),
                                      tolerance=tol_for_plot)
    mp_E_grid, xs_mp = grid.linearize(E_subset, one_input_generator_mp(T),
                                      tolerance=tol_for_plot)
    E_grids.extend([di_E_grid.copy(), pc_E_grid.copy(), mp_E_grid.copy()])
    print('T = {:f}'.format(T))
    print('shape of di_grid: ', di_E_grid.shape)
    print('shape of pc_grid: ', pc_E_grid.shape)
    print('shape of mp_grid: ', mp_E_grid.shape)
    Tab1D_dict[T] = {'di': Tabulated1D(arr_conv(di_E_grid.copy()),
                                       arr_conv(xs_di.copy())),
                     'pc': Tabulated1D(arr_conv(pc_E_grid.copy()),
                                       arr_conv(xs_pc.copy())),
                     'mp': Tabulated1D(arr_conv(mp_E_grid.copy()),
                                       arr_conv(xs_mp.copy())) }

                  

union_grid = reduce(np.union1d, E_grids)
print('shape of union_grid: ', union_grid.shape)

# Evaluate errors on union grid
slbw_abs_error_vecs = []
slbw_rel_error_vecs = []
multipole_abs_error_vecs = []
multipole_rel_error_vecs = []
for T in Ts:
    di_xs = Tab1D_dict[T]['di'](union_grid)
    pc_xs = Tab1D_dict[T]['pc'](union_grid)
    mp_xs = Tab1D_dict[T]['mp'](union_grid)
    
    slbw_abs_err = np.abs(pc_xs - di_xs)
    slbw_rel_err = (slbw_abs_err)/di_xs
    multipole_abs_err = np.abs(mp_xs - di_xs)
    multipole_rel_err = (multipole_abs_err)/di_xs
    
    slbw_abs_error_vecs.append(copy.deepcopy(slbw_abs_err))
    slbw_rel_error_vecs.append(copy.deepcopy(slbw_rel_err))
    multipole_abs_error_vecs.append(copy.deepcopy(multipole_abs_err))
    multipole_rel_error_vecs.append(copy.deepcopy(multipole_rel_err))
    
    # fig = plt.figure()
    # ax1 = fig.add_subplot(211)
    # ax1.loglog(union_grid, di_xs, label='di')
    # ax1.loglog(union_grid, pc_xs, label='pc')
    # ax1.loglog(union_grid, mp_xs, label='mp')
    # ax1.legend()
    # 
    # ax2 = fig.add_subplot(212, sharex=ax1)
    # ax2.loglog(union_grid, slbw_rel_err, label='pc')
    # ax2.loglog(union_grid, multipole_rel_err, label='mp')
    # ax2.legend()
    # ax2.set_title('Error to Direct Integration')
    # plt.show()
end = time.perf_counter()
total_time = (end-start)/60

# Store data in a dictionay for writing file
file_dict = {'name': 'Direct Integration Comparison',
             'tol_0K': tol_0K, 
             'tol_at_T' : tol_for_plot,
             'union_grid': union_grid,
             'Ts': Ts,
             'mp_plot_grid': np.array(multipole_rel_error_vecs),
             'slbw_plot_grid': np.array(slbw_rel_error_vecs),
             'time': total_time}


# Create unique data file
name_maker = 'di_data%s'
index = 0
while os.path.exists(name_maker % index):
    index += 1
filename = name_maker%index

with open(filename, 'wb') as f:
    pickle.dump(file_dict, f, pickle.HIGHEST_PROTOCOL)
