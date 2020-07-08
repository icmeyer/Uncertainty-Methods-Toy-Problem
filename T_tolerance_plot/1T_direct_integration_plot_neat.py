# Building a union grid which will evaluate all functions to
# a given tolerance
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
import matplotlib

import openmc.data.grid as grid
from openmc.data.function import Tabulated1D

import copy


# Need some functions from parent directory
import os, sys
problem_dir = os.path.dirname(os.getcwd())
sys.path.insert(0, problem_dir)

from slbw import * #exact_Σγ, exact_Real_SLBW, zeroK_exact_Re_SLBW_χψ_Σγ, exact_dΣγ_dΓ, z_space_Σγ, analytic_Σγ, analytic_dΣγ_dΓ, multipole_Σ, multipole_dΣ_dΓ, exact_poles_and_residues , exact_poles_and_residues_differentials_dΠ_dΓ
from data import a_U238, ρ0_U238, ρ0, μ_E0_U238, μ_Γn_U238, μ_Γγ_U238, μ_Γ_U238, cov_Γ_U238 
from data import cov_Γ_U238 as cov_Γ_U238_orig
from vector_fit import VF_algorithm, rational_function

from doppler_object import broadened_xs_generator
from njoy_helper import njoy_broadened

def arr_conv(array):
    return np.asarray(array).flatten()

#Change font
from matplotlib import rc

# Fig size
scaling = 4
figsize = [2*scaling, 2*scaling]
matplotlib.rcParams['figure.figsize'] = figsize

## for Palatino and other serif fonts use:
nice_font = True
if nice_font:
    rc('font',**{'family':'serif','serif':['Palatino']})
    plt.rcParams.update({'font.size': 16})
    rc('text', usetex=True)
font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 13}
matplotlib.rc('font', **font)

# Data
A = 238
a = 0 
Γ = μ_Γ_U238
b = 2*np.pi*Γ[1]*Γ[2]/(ρ0**2*Γ[0]**0.5*(Γ[1]+ Γ[2]))
r = a - 1j*b
Π = exact_poles_and_residues(μ_Γ_U238)
data_scaling = 1
cov_Γ_U238 = data_scaling * cov_Γ_U238_orig

# "Hyperparameters"
# refined value in parantheses
# Number of 0K energy groups (1e5)
N_g = int(1e5)
# Tolerance for 0K energy grid (1e-10)
tol_0K = 1e-11
# Number of groups at T (1e3)
N_sub = int(1e3)
# Tolerance at T (1e-4)
tol_for_plot = 1e-5

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
plot_info = ''
plot_info += 'shape of 0K grid: {:f}\n'.format(E_g_0K.shape[0])

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
T_list = [300, 1e5, 1e7] # K
T_num = len(T_list)

# Generate axes for subplots
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(nrows=4, ncols=2, 
                                                        sharex=True, sharey='col')
axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]
# Plotting
# get colors
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
styles = ['-', '--', '-.', ':']
# xs_labels = ['T=0K Cross Section', 'Piecewise Integration \n of Solbrig Kernel', '$\sigma(\Pi,T)$','$\Psi/\xi$']
xs_labels = ['T=0K Cross Section', 'Piecewise Integration \n of Solbrig Kernel','$\Psi / \chi$', '$\sigma(\Pi,T)$']
# xs_labels = ['1', '2', '3', '4']
error_labels = ['$ \Psi / \chi$', '$\sigma(\Pi,T)$']


width = 2

for plot_num, T in enumerate(T_list):
    print('Processing T = {:f}'.format(T))
    plot_info += '\nFor T={:f}\n'.format(T)
    di_E_grid, xs_di = grid.linearize(E_subset, one_input_generator_direct_integration(T),
                                      tolerance=tol_for_plot)
    pc_E_grid, xs_pc = grid.linearize(E_subset, one_input_generator_psichi(T),
                                      tolerance=tol_for_plot)
    mp_E_grid, xs_mp = grid.linearize(E_subset, one_input_generator_mp(T),
                                      tolerance=tol_for_plot)
    
    plot_info += 'shape of di_grid: {:f}\n'.format(di_E_grid.shape[0])
    plot_info += 'shape of pc_grid: {:f}\n'.format(pc_E_grid.shape[0])
    plot_info += 'shape of mp_grid: {:f}\n'.format(mp_E_grid.shape[0])
    
    union_grid = reduce(np.union1d, (di_E_grid, pc_E_grid, mp_E_grid))
    plot_info += 'shape of union_grid: {:f}\n'.format(union_grid.shape[0])
    
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
    
    
    ax_left = axes[2*plot_num]
    ax_right = axes[2*plot_num+1]

    ax_left.loglog(E_g_0K, exact_0K, label='0K', linestyle=styles[0], color=colors[0], lw=width)
    ax_left.loglog(di_E_grid, xs_di, label='di', linestyle=styles[1], color=colors[1], lw=width)
    ax_left.loglog(pc_E_grid, xs_pc, label='pc', linestyle=styles[2], color=colors[2], lw=width)
    ax_left.loglog(mp_E_grid, xs_mp, label='mp', linestyle=styles[3], color=colors[3], lw=width)


    temp_str = 'T={0:1.1e} K\n'.format(T)
    if plot_num == 0:
        ax_left.set_title('Cross Section')
        ax_right.set_title('Error to Piecewise Integration')
        ax_left.set_ylabel(temp_str + 'Cross Section [barns]')
        ax_right.set_ylabel('Absolute Relative Error')
    elif plot_num==(T_num - 1):
        ax_left.set_ylabel(temp_str + '.')
        ax_left.set_xlabel('Energy [eV]')
        ax_right.set_xlabel('Energy [eV]')
    else:
        ax_left.set_ylabel(temp_str + '.')
    plot_info +=   ('T = {:1.2e} K \n'
                    '0K Tol: {:.1e}\n'
                    'DI/PC/MP Tol: {:.1e} \n'
                    'NJOY Tol: {:.1e} \n'
                    'Covariance Scaling: {:f}'.format(T, tol_0K, tol_for_plot, njoy_tol, data_scaling))
    
    ax_right.loglog(union_grid, pc_err, label='psi chi', color = colors[2], 
               linestyle=styles[2], lw=width)
    ax_right.loglog(union_grid, mp_err, label='mp',      color = colors[3], 
               linestyle=styles[3], lw=width)
    
    # ax_left.set_xscale('linear')
    # ax_right.set_xscale('linear')
    
    if njoy_comparison:
        ax_left.loglog(union_grid, njoy_xs, label='njoy')
        ax_right.loglog(union_grid, njoy_err, label='njoy')



    ax_left.set_xlim(1e-2, 1e2)
    ax_left.set_ylim(1e-4, 1e6)

# Custom legends
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color=colors[0], lw=2, linestyle=styles[0]),
                Line2D([0], [0], color=colors[1], lw=2, linestyle=styles[1]),
                Line2D([0], [0], color=colors[2], lw=2, linestyle=styles[2]),
                Line2D([0], [0], color=colors[3], lw=2, linestyle=styles[3])]
ax7.legend(custom_lines, xs_labels, loc='center')
ax7.axis('off')

custom_lines = [Line2D([0], [0], color=colors[2], lw=2, linestyle=styles[2]),
                Line2D([0], [0], color=colors[3], lw=2, linestyle=styles[3])]
ax8.legend(custom_lines, error_labels, loc='center')
ax8.axis('off')

# Fix ticks/add grid
locmajx = matplotlib.ticker.LogLocator(base=10.0,numticks=200) 
locmajy = matplotlib.ticker.LogLocator(base=10.0,numticks=200) 
locmin = matplotlib.ticker.LogLocator(base=10.0,subs=(0.33,0.66), numticks=100)
for ax in axes:
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(axis='both', direction='out')
    ax.tick_params(axis='both', which='minor')
    ax.xaxis.set_major_locator(locmajx)
    ax.xaxis.set_minor_locator(locmin)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.yaxis.set_major_locator(locmajy)
    ax.yaxis.set_minor_locator(locmin)
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.grid()
    ax.set_yticks(ax.get_yticks())

# Set every other ytick label to blank
for ax in axes:
    ylabels = ax.get_yticklabels()
    for label in ylabels[::2]:
        label.set_visible(False)

plt.tight_layout()
# plt.show()

# File management
import os
base_directory = 'figs'
index = 0 
if not os.path.isdir(base_directory):
    os.mkdir(directory)

while os.path.exists(base_directory+'/'+str(index)):
    index += 1
directory = base_directory + '/' + str(index)
os.mkdir(directory)

fig.savefig( directory+'/'+'comparisons.pdf')

text_file = open(directory+'/'+'plot_info', "w")
n = text_file.write(plot_info)
text_file.close()

