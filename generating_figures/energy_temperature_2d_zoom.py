import time
start = time.perf_counter()
import numpy as np
import scipy
import scipy.special as faddeeva
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# Need some functions from parent directory
import os, sys
problem_dir = os.path.dirname(os.getcwd())
sys.path.insert(0, problem_dir)

from slbw import * #exact_Σγ, exact_Real_SLBW, zeroK_exact_Re_SLBW_χψ_Σγ, exact_dΣγ_dΓ, z_space_Σγ, analytic_Σγ, analytic_dΣγ_dΓ, multipole_Σ, multipole_dΣ_dΓ, exact_poles_and_residues , exact_poles_and_residues_differentials_dΠ_dΓ
from data import a_U238, ρ0_U238, ρ0, μ_E0_U238, μ_Γn_U238, μ_Γγ_U238, μ_Γ_U238, cov_Γ_U238 
from vector_fit import VF_algorithm, rational_function

crude_for_test = False
if crude_for_test:
    N_g_orig = 100 # 0K cross section points
    N_Ts = 10
    N_g = 10 # Points at which to evaluate cross section at temperature
else:
    N_g_orig = 10000 # 0K cross section points
    N_Ts = 40
    N_g = 1000 # Points at which to evaluate cross section at temperature

A = 238
E_g_orig = np.logspace(-2, 3, N_g_orig) # Energy groups (here log-spaced) for flux ψ
z_g_orig = E_g_orig**0.5 # Energy groups (here log-spaced) for flux ψ


a = 0 
Γ = μ_Γ_U238
b = 2*np.pi*Γ[1]*Γ[2]/(ρ0**2*Γ[0]**0.5*(Γ[1]+ Γ[2]))
r = a - 1j*b
Π = exact_poles_and_residues(μ_Γ_U238)

# Broaden the 0 K cross section
from doppler_object import broadened_xs_generator
import copy

exact_Σγ_0K = np.array([exact_Σγ(E_g_orig[g], μ_Γ_U238) for g in range(E_g_orig.size)])
# # Build 2D Tolerance Plot
slbw_max_abs_error = []
slbw_max_rel_error = []
slbw_abs_error_vecs = []
slbw_rel_error_vecs = []
multipole_abs_error_vecs = []
multipole_rel_error_vecs = []
# Ts = np.linspace(1e-3, 1e5, N_Ts)
Ts = np.logspace(-3, 6, N_Ts)

# E_g = np.linspace(1e-6, 1e4, 1000)
E_g = np.logspace(0, 2, N_g)
# Add more points at resonance
N_extra = 1000
E_extra = np.linspace(2.0, 12, N_extra)
E_g = np.union1d(E_extra, E_g)
z_g = E_g**0.5
subset_low = np.searchsorted(E_g, 1e-5)
subset_high = np.searchsorted(E_g, 1e3)

# Ts = np.append(Ts, 300e6)
i=0
for T in Ts:
    i+=1
    print('At temperature: {:f} K  ({:d}/{:d})'.format(T, i, len(Ts)))
    slbw_xs = np.array([TK_approx_SLBW_Faddeeva_Σγ(E_g[g], Γ, r ,T) for g in range(E_g.size)])
    multipole_xs = np.array([approx_multipole_Doppler_Σ(z_g[g], Π, T) for g in range(z_g.size)])
    broadening_object = broadened_xs_generator(E_g_orig, exact_Σγ_0K, T, A)
    # sigma1_xs = broadening_object.broaden_use_trap(E_g)
    sigma1_xs = broadening_object.broadened_xs(E_g)
    # sigma1_xs = broaden(E_g, exact_Σγ_0K, T, A, E_g)
    
    slbw_abs_err = np.abs(slbw_xs - sigma1_xs)
    slbw_rel_err = (slbw_abs_err)/sigma1_xs
    multipole_abs_err = np.abs(multipole_xs - sigma1_xs)
    multipole_rel_err = (multipole_abs_err)/sigma1_xs
    
    slbw_abs_error_vecs.append(copy.deepcopy(slbw_abs_err))
    slbw_rel_error_vecs.append(copy.deepcopy(slbw_rel_err))
    multipole_abs_error_vecs.append(copy.deepcopy(multipole_abs_err))
    multipole_rel_error_vecs.append(copy.deepcopy(multipole_rel_err))
    
    # # Plot xs
    # plt.loglog(E_g_orig, exact_Σγ_0K, label='0K Exact')
    # plt.loglog(E_g, sigma1_xs, label='Sigma 1')
    # plt.loglog(E_g , slbw_xs, label='SLBW')
    # plt.loglog(E_g , multipole_xs, label='Multipole')
    # plt.legend()
    # plt.show()
  
    # # Plot error individual error plots
    # plt.loglog(E_g , slbw_abs_err, '-.', label='Abs. Err. of SLBW') 
    # plt.loglog(E_g , slbw_rel_err, '-.', label='Rel. Err. of SLBW') 
    # plt.loglog(E_g , multipole_abs_err, '-.', label='Abs. Err. of Multipole') 
    # plt.loglog(E_g , multipole_rel_err, '-.', label='Rel. Err. of Multipole')
    # plt.legend()
    # plt.show()
    
#     # Plot multipole only and mark maximums
#     plt.loglog(E_g , multipole_abs_err, '-.', label='Abs. Err. of Multipole') 
#     plt.loglog(E_g , multipole_rel_err, '-.', label='Rel. Err. of Multipole')
#     plt.axvline(x=E_g[np.argmax(multipole_abs_err)], color='chartreuse', label='Absolute Max')
#     plt.axvline(x=E_g[np.argmax(multipole_rel_err)], color='gold', label='Relative Max')
#     plt.legend()
#     plt.show()
    
    
### PLOTTING 
# Options
nice_font = True
choose_levels = True
levels = [1e-3, 1e-2, 1e-1]
styles = ['solid', 'dashed', 'dashdot']

from matplotlib import rc
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D
scaling = 1.8
ratio = [5, 4]
figsize = [ratio[0]*scaling, ratio[1]*scaling]
mpl.rcParams['figure.figsize'] = figsize
if nice_font:
    rc('font',**{'family':'serif','serif':['Palatino']})
    plt.rcParams.update({'font.size': 16})
    rc('text', usetex=True)

# Find values for colorbar range
mp_plot_grid = np.array(multipole_rel_error_vecs)[:,subset_low:subset_high]
slbw_plot_grid = np.array(slbw_rel_error_vecs)[:,subset_low:subset_high]
max_color = np.max([mp_plot_grid.max(), slbw_plot_grid.max()])
min_color = np.min([mp_plot_grid.min(), slbw_plot_grid.min()])

# Initialize figure
fig = plt.figure()

# Plot Multipole Error
plot_grid = mp_plot_grid
ax1 = fig.add_subplot(111)
grid1 = ax1.pcolormesh(E_g[subset_low:subset_high], Ts, plot_grid, norm=colors.LogNorm(vmin=min_color, vmax=max_color), cmap='plasma')
if choose_levels:
    CS = ax1.contour(E_g[subset_low:subset_high], Ts, plot_grid, levels, linestyles=styles, colors='k')
    # ax1.clabel(CS, levels, inline=1, fmt='%1.1e', fontsize=10)
else: 
    CS = ax1.contour(E_g[subset_low:subset_high], Ts, plot_grid, colors='k')
    ax1.clabel(CS, fmt='%1.1e', fontsize=10)
ax1.set_xscale('log')
ax1.set_xlabel('E (eV)')
ax1.set_ylabel('T (Kelvin)')
ax1.set_title('Multipole Relative Error')
ax1.set_yscale('log')
# ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1E'))
# ax1.grid()

# # Plot SLBW Error
# plot_grid = slbw_plot_grid
# ax2 = fig.add_subplot(122)
# grid2 = ax2.pcolormesh(E_g[subset_low:subset_high], Ts, plot_grid, norm=colors.LogNorm(vmin=min_color, vmax=max_color), cmap='plasma')
# if choose_levels:
#     CS = ax2.contour(E_g[subset_low:subset_high], Ts, plot_grid, levels, linestyles=styles, colors='k')
#     # ax2.clabel(CS, levels, inline=1, fmt='%1.1e', fontsize=10)
# else: 
#     CS = ax2.contour(E_g[subset_low:subset_high], Ts, plot_grid, colors='k')
#     ax2.clabel(CS, fmt='%1.1e', fontsize=10)
# ax2.set_xscale('log')
# ax2.set_xlabel('E (eV)')
# ax2.set_ylabel('T (Kelvin)')
# ax2.set_title('SLBW Relative Error')
# ax2.set_yscale('log')
# # ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1E'))
# # ax2.grid()

fig.tight_layout(w_pad=1.0)

# Add color bar with axes [left, bottom, width, height]
fig.subplots_adjust(right=0.80)
cbar_ax = fig.add_axes([0.85, 0.30, 0.05, 0.65])
norm=colors.LogNorm(vmin=min_color, vmax=max_color)
print('Attempted min/max for LogNorm color bar: {:E}/{:E}'.format(min_color, max_color))
# Make my own ticks
# Find powers of 10
min_pow = np.ceil(np.log10(min_color))
max_pow = np.floor(np.log10(max_color))
tick_locations = np.logspace(min_pow, max_pow, (max_pow - min_pow)+1)
colorbar = mpl.colorbar.ColorbarBase(cbar_ax, cmap='plasma', norm=norm, ticks=tick_locations)
# colorbar = fig.colorbar(grid2, cax=cbar_ax)

# Add contour legend [left, bottom, width, height]
contour_ax = fig.add_axes([0.90, 0.10, 0.05, 0.1])
legend_elements = [Line2D([0],[0],linestyle=styles[0], label=levels[0], color='k'),
                   Line2D([0],[0],linestyle=styles[1], label=levels[1], color='k'),
                   Line2D([0],[0],linestyle=styles[2], label=levels[2], color='k')]
contour_ax.legend(handles=legend_elements, loc='center', title='Tolerance\n Contours')
contour_ax.set_axis_off()


end = time.perf_counter()
total_time = (end-start)/60
title = 'Energy Range: {:e} - {:e} eV \n N_Points 0K: {:d} \n N_Points Broadened: {:d} \n N Temperatures: {:d} \n Time: {:f} min'.format(E_g_orig[0], E_g_orig[-1], N_g_orig, N_g, N_Ts, total_time)
filename = 'Multipole_Only - Energy Range: {:e} - {:e} eV  N_Points 0K: {:d}  N_Points Broadened: {:d}  N Temperatures: {:d}  Time: {:f} min'.format(E_g_orig[0], E_g_orig[-1], N_g_orig, N_g, N_Ts, total_time)
# fig.suptitle(title)

fig.savefig('./figs/'+filename+".pdf")
plt.show()

