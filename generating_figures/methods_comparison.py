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

# Build energy grid
# E_g = np.linspace(1e-6, 1e4, 1000)
N_g = 10000
E_g = np.logspace(-6, 4, N_g)
z_g = E_g**0.5
subset_low = np.searchsorted(E_g, 1e-4)
subset_high = np.searchsorted(E_g, 1e2)

A = 238
a = 0 
Γ = μ_Γ_U238
b = 2*np.pi*Γ[1]*Γ[2]/(ρ0**2*Γ[0]**0.5*(Γ[1]+ Γ[2]))
r = a - 1j*b
Π = exact_poles_and_residues(μ_Γ_U238)

# Broaden the 0 K cross section
from doppler_object import broadened_xs_generator
exact_Σγ_0K = np.array([exact_Σγ(E_g[g], μ_Γ_U238) for g in range(E_g.size)])


T = 1e6
slbw_xs = np.array([TK_approx_SLBW_Faddeeva_Σγ(E_g[g], Γ, r ,T) for g in range(E_g.size)])
multipole_xs = np.array([approx_multipole_Doppler_Σ(z_g[g], Π, T) for g in range(z_g.size)])
broadening_object = broadened_xs_generator(E_g, exact_Σγ_0K, T, A)
# sigma1_xs = broadening_object.broaden_use_trap(E_g)
sigma1_xs = broadening_object.broadened_xs(E_g)
# sigma1_xs = broaden(E_g, exact_Σγ_0K, T, A, E_g)



### PLOTTING
# Options
nice_font = True
use_subset = True

if use_subset:
    E_g = E_g[subset_low:subset_high]
    exact_Σγ_0K =  exact_Σγ_0K[subset_low:subset_high]
    sigma1_xs =    sigma1_xs[subset_low:subset_high]
    slbw_xs =      slbw_xs[subset_low:subset_high]
    multipole_xs = multipole_xs[subset_low:subset_high]

slbw_abs_err = np.abs(slbw_xs - sigma1_xs)
slbw_rel_err = (slbw_abs_err)/sigma1_xs
multipole_abs_err = np.abs(multipole_xs - sigma1_xs)
multipole_rel_err = (multipole_abs_err)/sigma1_xs

from matplotlib import rc
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D
scaling = 0.9
ratio = [8, 10]
figsize = [ratio[0]*scaling, ratio[1]*scaling]
mpl.rcParams['figure.figsize'] = figsize
if nice_font:
    rc('font',**{'family':'serif','serif':['Palatino']})
    plt.rcParams.update({'font.size': 16})
    rc('text', usetex=True)

fig = plt.figure()

# Plot xs
ax1 = fig.add_subplot(211)
ax1.loglog(E_g, exact_Σγ_0K, linewidth=2, label='0 K Exact')
ax1.loglog(E_g,  sigma1_xs, ':', linewidth=2, label='SIGMA1')
ax1.loglog(E_g , slbw_xs, 'g--',linewidth=2, label='$ \Psi / \chi$')
ax1.loglog(E_g , multipole_xs,'r-.', linewidth=2,label='Multipole')
ax1.set_xlabel('Energy (eV)')
ax1.set_ylabel('Cross Section (barns)')
ax1.set_title('Comparison at T={:.1e}'.format(T))
ax1.grid()
ax1.legend()

# Plot error individual error plots
ax2 = fig.add_subplot(212, sharex=ax1)
# ax2.loglog(E_g , slbw_abs_err, linewidth=2, label='Abs. Err. of SLBW') 
ax2.loglog(E_g , slbw_rel_err, 'g--', linewidth=2, label='$\Psi/\chi$') 
# ax2.loglog(E_g , multipole_abs_err, linewidth=2, label='Abs. Err. of Multipole') 
ax2.loglog(E_g , multipole_rel_err, 'r-.', linewidth=2, label='Multipole')
ax2.grid()
ax2.set_title('Relative Error to SIGMA1')
ax2.set_xlabel('Energy (eV)')
ax2.legend()


fig.tight_layout(h_pad=1.5)
plt.show()

end = time.perf_counter()
total_time = (end-start)/60
print(total_time)
filename = 'T={:f}'.format(T)

fig.savefig('./figs/'+filename+".pdf")
plt.show()

# # Plot multipole only and mark maximums
# plt.loglog(E_g , multipole_abs_err, '-.', label='Abs. Err. of Multipole') 
# plt.loglog(E_g , multipole_rel_err, '-.', label='Rel. Err. of Multipole')
# plt.axvline(x=E_g[np.argmax(multipole_abs_err)], color='chartreuse', label='Absolute Max')
# plt.axvline(x=E_g[np.argmax(multipole_rel_err)], color='gold', label='Relative Max')
# plt.legend()
# plt.show()

