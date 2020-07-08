# Building a union grid which will evaluate all functions to
# a given tolerance
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt

import openmc.data.grid as grid
from openmc.data.function import Tabulated1D


# Need some functions from parent directory
import os, sys
problem_dir = os.path.dirname(os.getcwd())
sys.path.insert(0, problem_dir)

from slbw import * #exact_Σγ, exact_Real_SLBW, zeroK_exact_Re_SLBW_χψ_Σγ, exact_dΣγ_dΓ, z_space_Σγ, analytic_Σγ, analytic_dΣγ_dΓ, multipole_Σ, multipole_dΣ_dΓ, exact_poles_and_residues , exact_poles_and_residues_differentials_dΠ_dΓ
from data import a_U238, ρ0_U238, ρ0, μ_E0_U238, μ_Γn_U238, μ_Γγ_U238, μ_Γ_U238, cov_Γ_U238 
from vector_fit import VF_algorithm, rational_function

from doppler_object import broadened_xs_generator
from njoy_helper import njoy_broadened

def arr_conv(array):
    return np.asarray(array).flatten()


# Data
A = 238
a = 0 
Γ = μ_Γ_U238
b = 2*np.pi*Γ[1]*Γ[2]/(ρ0**2*Γ[0]**0.5*(Γ[1]+ Γ[2]))
r = a - 1j*b
Π = exact_poles_and_residues(μ_Γ_U238)

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
# NJOY
njoy_comparison = False
njoy_tol = 0.001

# Functions for tolerance checking
def one_input_exact_Σγ_0K(E):
    return exact_Σγ(E, μ_Γ_U238)


# Find 0K grid
N_g = 100000
E_g_start = np.logspace(-9, 3, N_g)
# subset_low = np.searchsorted( E_g_start, 1e-4)
# subset_high = np.searchsorted(E_g_start, 1e2)
N_sub = 1000
E_subset = np.logspace(-4, 2, N_sub)

E_g_0K, exact_0K = grid.linearize(E_g_start, one_input_exact_Σγ_0K, tolerance=tol_0K)

# plt.loglog(E_g_0K, exact_0K)
# plt.show()
print('Shape of 0K grid', E_g_0K.shape)


def one_input_generator_direct_integration(T):
    def one_input_direct_integration(E):
        broadening_object = broadened_xs_generator(E_g_0K, exact_0K, T, A)
        return broadening_object.broaden_use_trap([E])
    return one_input_direct_integration

def one_input_generator_psichi(T):
    def one_input_psichi(E):
        return TK_approx_SLBW_Faddeeva_Σγ(E, Γ, r ,T)
    return one_input_psichi
    
def one_input_generator_mp(T):
    def one_input_mp(E):
        z = E**0.5
        return approx_multipole_Doppler_Σ(z, Π, T) 
    return one_input_mp

# Test at 10K
T = 300 # K
di_E_grid, xs_di = grid.linearize(E_subset, one_input_generator_direct_integration(T),
                                  tolerance=tol_for_plot)
pc_E_grid, xs_pc = grid.linearize(E_subset, one_input_generator_psichi(T),
                                  tolerance=tol_for_plot)
mp_E_grid, xs_mp = grid.linearize(E_subset, one_input_generator_mp(T),
                                  tolerance=tol_for_plot)

print('shape of di_grid: ', di_E_grid.shape)
print('shape of pc_grid: ', pc_E_grid.shape)
print('shape of mp_grid: ', mp_E_grid.shape)

union_grid = reduce(np.union1d, (di_E_grid, pc_E_grid, mp_E_grid))
print('shape of union_grid: ', union_grid.shape)

# Evaluation on union grid
xs_di_union  = Tabulated1D(arr_conv(di_E_grid.copy()),
                           arr_conv(xs_di.copy()))(union_grid)
xs_pc_union  = Tabulated1D(arr_conv(pc_E_grid.copy()),
                           arr_conv(xs_pc.copy()))(union_grid)
xs_mp_union  = Tabulated1D(arr_conv(mp_E_grid.copy()), 
                           arr_conv(xs_mp.copy()))(union_grid)

# Calculate error
pc_err = np.abs((xs_pc_union - xs_di_union)/(xs_di_union))
mp_err = np.abs((xs_mp_union - xs_di_union)/(xs_di_union))

if njoy_comparison:
    MT = 102
    njoy_xs = njoy_broadened('./endf/x998.ndf-edit', T, MT, union_grid, error=njoy_tol)
    njoy_err = np.abs((njoy_xs - xs_di_union)/(xs_di_union))

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.loglog(E_g_0K, exact_0K, label='0K')
ax1.loglog(di_E_grid, xs_di, label='di')
ax1.loglog(pc_E_grid, xs_pc, label='pc')
ax1.loglog(mp_E_grid, xs_mp, label='mp')
ax1.set_title('T = {:.2e} K \n'
              '0K Tol: {:.1e}\n'
              'DI/PC/MP Tol: {:.1e} \n'
              'NJOY Tol: {:.1e}'.format(T, tol_0K, tol_for_plot, njoy_tol))

ax2 = fig.add_subplot(212, sharex=ax1)
ax2.loglog(union_grid, pc_err, label='psi chi')
ax2.loglog(union_grid, mp_err, label='mp')
ax2.set_title('Error to Direct Integration')

# ax1.set_xscale('linear')
# ax2.set_xscale('linear')

if njoy_comparison:
    ax1.loglog(union_grid, njoy_xs, label='njoy')
    ax2.loglog(union_grid, njoy_err, label='njoy')

ax1.legend()
ax2.legend()

plt.show()
