# Want to compare the distribution of reconstructed cross sections from:
#   - resonance parameters
#   - multipole parameters
# By starting with resonance parameters and their uncertainty and then 
# converting that to multipole parameters and their uncertianty
# 
# General outline:
# Define resonance parameters and corresponding uncertainty
# Convert to RPs to MPs
# Convert Cov(RPs) to Cov(MPs)
# Plot histograms of xs at given energy by sampling resonance parameters
# Overlay a normal curve on data

# Regular python libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
np.random.seed(2)

# Need some functions from parent directory
import os, sys
problem_dir = os.path.dirname(os.getcwd())
sys.path.insert(0, problem_dir)

# Local functions
from slbw import  exact_Σγ, exact_dΣγ_dΓ, z_space_Σγ, analytic_Σγ, \
                  analytic_dΣγ_dΓ, multipole_Σ, multipole_dΣ_dΓ,   \
                  exact_poles_and_residues, \
                  exact_poles_and_residues_differentials_dΠ_dΓ, mp_sens_rp, \
                  multipole_dΣ_dΠ, multipoles_set_Π
from data import a_U238, ρ0_U238, ρ0, μ_E0_U238, μ_Γn_U238, μ_Γγ_U238, \
                 μ_Γ_U238, cov_Γ_U238
from data import cov_Γ_U238 as cov_Γ_U238_orig
from vector_fit import VF_algorithm, rational_function

# Helper functions
def sandwich(sens, cov):
    # print('sensitivity size:', sens.shape)
    # print('cov size:', sens.shape)
    return np.matmul(sens.T, np.matmul(cov, sens))

def separate_complex(vector):
    return np.hstack([np.real(vector), np.imag(vector)])

def join_complex(vector):
    vector = np.asarray(vector)
    n = len(vector)
    half = int(n/2)
    return vector[:half]+1j*vector[half:]

def correlation(cov):
    corr = np.zeros_like(cov)
    n = cov.shape[0]
    for i in range(n):
        for j in range(n):
            corr[i,j]=cov[i,j]/cov[i,i]**(0.5)/cov[j,j]**(0.5)
    return corr

def normal(mu, sigma, x):
    return (1/((2*np.pi)**0.5 * sigma))*np.exp(-0.5*(1/sigma*(x-mu))**2)

### Problem parameters ###
# E_0 = 3.674280 # [eV] Energy for histogram plots
E_0 = 6.674280 # [eV] Energy for histogram plots
# E_0 = 6.674280 - μ_Γγ_U238 # [eV] Energy for histogram plots
z_0 = E_0**0.5
N_g = 1000 # Number of energy groups

# Covariance data scaling
cov_scaling = 1e-3
cov_Γ_U238 = cov_scaling * cov_Γ_U238_orig

# Energy grid structure
n_samples = 10000 # Number of samples for histogram
E_max = 10**3 # Maximum energy of the energy groups
E_min = 10**-1 # Minimum energy
E_g = np.logspace(np.log10(E_min),np.log10(E_max),N_g) 
z_g = E_g**0.5 # Energy groups (here log-spaced) for flux ψ

### Generate Sample Sets


# Sampling cross section values using σ(Γ, E_0) and Cov_Γ
Γ_samples = np.random.multivariate_normal(μ_Γ_U238, cov_Γ_U238, n_samples)
Σγ_hist = np.array([exact_Σγ(E_0, Γ_sample) for Γ_sample in Γ_samples])
mean_Σγ = np.sum(Σγ_hist)/n_samples
strd_dev_Σγ = (np.dot((Σγ_hist-mean_Σγ).T,(Σγ_hist-mean_Σγ))/(n_samples-1))**0.5

# Exact Multipole Representation
μ_Π_U238 = exact_poles_and_residues(μ_Γ_U238)

# Evaluate Multipole Sensitivities to Resonances Parameters
dΠ_dΓ = mp_sens_rp(μ_Γ_U238)

# Use sandwich rule to evaluate MP covariance (4x4 matrix for 2 pole/residue pairs)
cov_Π_U238 =  sandwich(separate_complex(dΠ_dΓ), cov_Γ_U238)

# Sampling cross section values using σ(Π, E_0) and Cov_Π
# Sensitivity method
mean_for_sampling = np.hstack([np.real(μ_Π_U238.flatten()), 
                               np.imag(μ_Π_U238.flatten())])
Π_samples = np.random.multivariate_normal(mean_for_sampling, cov_Π_U238, n_samples)
Σγ_hist_Π = np.array([multipole_Σ(z_0, join_complex(Π_sample).reshape((2,2)))\
            for Π_sample in Π_samples])
mean_Σγ_Π = np.sum(Σγ_hist_Π)/n_samples
strd_dev_Σγ_Π = (np.dot((Σγ_hist_Π - mean_Σγ_Π).T,(Σγ_hist_Π - mean_Σγ_Π))/\
                (n_samples-1))**0.5

# Monte Carlo method
Π_samples_mc = np.array([exact_poles_and_residues(Γ_sample) for Γ_sample in Γ_samples])
Σγ_hist_Π_mc = np.array([multipole_Σ(z_0, Π_sample.reshape((2,2))) for Π_sample in Π_samples_mc])
mean_Σγ_Π_mc = np.sum(Σγ_hist_Π_mc)/n_samples
strd_dev_Σγ_Π_mc = (np.dot((Σγ_hist_Π_mc-mean_Σγ_Π).T,(Σγ_hist_Π_mc-mean_Σγ_Π))/\
                   (n_samples-1))**0.5

# Generate samples from sensitivity propagation

### PLOTTING ###
# Should probably be a separate file at this point but lazy

# File management
# Saving 4 files: mc_xs, histogram, covariance, info
import os
base_directory = 'figs'
index = 0 
if not os.path.isdir(base_directory):
    os.mkdir(directory)

while os.path.exists(base_directory+'/'+str(index)):
    index += 1
directory = base_directory + '/' + str(index)
os.mkdir(directory)


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

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
styles = ['-', '--', '-.', ':']

# Plotting Paramaters
fig_alpha = 1.0 # Change opacity of lines
width = 2 # line width
num_bins = 200

# Generate 1st order sensitivity samples
# SLBW
Σγ_mean = exact_Σγ(E_0, μ_Γ_U238)
Σγ_sigma_at_E0 = (np.dot(exact_dΣγ_dΓ(E_0, μ_Γ_U238), \
                  np.dot(cov_Γ_U238,exact_dΣγ_dΓ(E_0, μ_Γ_U238))))**0.5
# Σγ_local_propagation = normal(Σγ_mean, Σγ_sigma_at_E0, bins)
Σγ_local_propagation_samples = np.random.normal(Σγ_mean, Σγ_sigma_at_E0, n_samples)
# Multipole
Π_Σγ_mean = multipole_Σ(z_0, μ_Π_U238)
mp_sens = multipole_dΣ_dΠ(z_0, μ_Π_U238)
Π_Σγ_sigma_at_E0 = sandwich(mp_sens, cov_Π_U238)**0.5
# Π_Σγ_local_propagation = normal(Π_Σγ_mean, Π_Σγ_sigma_at_E0, bins)
Π_Σγ_local_propagation_samples = np.random.normal(Π_Σγ_mean, Π_Σγ_sigma_at_E0, n_samples)


# Get bins using all data for plotting
bins = np.histogram(np.hstack((Π_Σγ_local_propagation_samples, Σγ_local_propagation_samples,
                               Σγ_hist_Π, Σγ_hist)), bins = num_bins)[1]
                               

fig = plt.figure()
ax1 = fig.add_subplot(211)

# Plot continuous Gaussian dist for 1st order sensitivity line
# ax1.plot(bins, Π_Σγ_local_propagation, '--b', label='1st order snesitivity analysis - Π')
# ax1.plot(bins, Σγ_local_propagation, '--r', label='1st order sensitivity analysis - Γ')

# Plot histograms
labels = ['$\sigma(\Gamma, E)$ sampling $\Gamma$ from Cov$(\Gamma)$',
          '$\sigma(\Pi, E)$ sampling $\Pi$ from conversion',
          '$\sigma(\sigma(\Gamma, E)$ sampling from 1st Order Gaussian',
          '$\sigma(\sigma(\Pi, E)$ sampling from 1st Order Gaussian']

n, _, _ = ax1.hist(Σγ_hist, bins, density=1, 
                            label=labels[0], alpha=fig_alpha,
                            histtype='step', linestyle=styles[0], lw=width)
n, _, _ = ax1.hist(Σγ_hist_Π, bins, density=1,
                            label=labels[1], alpha=fig_alpha,
                            histtype='step', linestyle=styles[1], lw=width)
n, _, _ = ax1.hist(Σγ_local_propagation_samples, bins, density=1,
                            label=labels[2], alpha=fig_alpha,
                            histtype='step', linestyle=styles[3], lw=width)
n, _, _ = ax1.hist(Π_Σγ_local_propagation_samples, bins, density=1,
                            label=labels[3], alpha=fig_alpha,
                            histtype='step', linestyle=styles[2], lw=width)
# n, _, _ = ax1.hist(Σγ_hist_Π_mc, num_bins, density=1,
#                             label='σ(Π, E_0) using Π from Γ sample conversion', alpha=fig_alpha)

# Plot means as vertical lines
# ax1.axvline(mean_Σγ_Π, color='r', label='MP Sensitivity Sample Mean')
# ax1.axvline(mean_Σγ, color='k', label='Resonances Sample Mean')
# ax1.axvline(mean_Σγ_Π_mc, color='g', label='Exact conversion of resonance samples mean')

# Set scale
ax1.set_yscale('log')

# Label

# Custom legend
ax_legend = fig.add_subplot(212)
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color=colors[0], lw=2, linestyle=styles[0]),
                Line2D([0], [0], color=colors[1], lw=2, linestyle=styles[1]),
                Line2D([0], [0], color=colors[2], lw=2, linestyle=styles[2]),
                Line2D([0], [0], color=colors[3], lw=2, linestyle=styles[3])]
ax_legend.legend(custom_lines, labels, loc='upper center')
ax_legend.axis('off')

ax1.set_xlabel('$\Sigma_{\gamma}(E)$ for E = %s eV [barns]'%(E_0) )
ax1.set_ylabel('Probability Density Histogram (bins sum to one)')
plot_info_string = ('----------- Γ --------------- \n'
              'Mean Σγ(E_0) = {:f} \n'
              'Samples Mean Σγ(E_0) = {:f} \n'
              'Senstivity sigma_at_E0 = {:f} \n'
              'Samples strd_dev_Σγ = {:f} \n\n'
              '----------- Π --------------- \n'
              'Mean Σγ(E_0) = {:f} \n'
              'Samples Mean Σγ(E_0) = {:f} \n'
              'Senstivity sigma_at_E0 = {:f} \n'
              'Samples strd_dev_Σγ = {:f} \n\n'
              'Uncertainty scaling = {:f} \n'
              .format(Σγ_mean, mean_Σγ, Σγ_sigma_at_E0, strd_dev_Σγ,
                      Π_Σγ_mean, mean_Σγ_Π, Π_Σγ_sigma_at_E0, strd_dev_Σγ_Π,
                      cov_scaling))

# Add ticks on right hand side
# ax1.yaxis.set_major_locator(plt.LogLocator(base=10, numticks=15))
# ax1.yaxis.set_major_locator(plt.MultipleLocator(5e-5))
locs = ax1.get_yticks()
print(locs)
labels = ax1.get_yticklabels()
# ax1.yaxis.tick_right(locs = locs, labels = labels)
ax1.yaxis.tick_right()
ax1.yaxis.set_ticks_position('both')
ax1.tick_params(axis='y', which='both', labelleft='on', labelright='on')

# Add grid
plt.rc('grid', linestyle="dashed", color='black')
ax1.grid()

# ax1.yaxis.set_yticks(locs)
# ax1.yaxis.set_yticklabels(labels)

# plt.show()
# ax1.set_title(title_text)
# fig.tight_layout()
# fig.subplots_adjust(top=0.6)

# ax2.set_xlabel('Σγ(E_0) for E_0 = %s'%(E_0) )
# ax2.set_ylabel('Probability density Histogram of Σγ(E_0)')
# ax2.set_title('Σmeanγ(E_0) = %s, meanΣγ(E_0) = %s, Σγ_sigma_at_E0 = %s, strd_dev_Σγ = %s'%(Σγ_mean, Π_Σγ_mean, Π_Σγ_sigma_at_E0, strd_dev_Σγ))

# Plot multipole covariance and correlation
fig2 = plt.figure()
ax4 = fig2.add_subplot(121)
im1 = ax4.imshow(cov_Π_U238)
ax4.set_title('$\Pi$ Covariance')
ax4.figure.colorbar(im1)
ax5 = fig2.add_subplot(122)
im2 = ax5.imshow(correlation(cov_Π_U238))
ax5.figure.colorbar(im2)
ax5.set_title('$\Pi$ Correlation (Blanks probably zero division)')
fig.tight_layout()

# Plot sampled cross sections
energies = np.logspace(-3,3,10000)
fig3 = plt.figure()
ax4 = fig3.add_subplot(111)
mp_xs = [multipole_Σ(energy**0.5, μ_Π_U238) for energy in energies]
ax4.plot(energies, mp_xs, label='mean')
for i in range(100):
    Π_for_plotting = join_complex(Π_samples[i]).reshape((2,2))
    ax4.plot(energies, [multipole_Σ(energy**0.5, Π_for_plotting) for energy in energies], alpha=0.1)
ax4.set_xscale('log')
ax4.set_yscale('log')
ax4.legend()

plt.tight_layout()
fig.savefig( directory+'/'+'histogram.pdf')
fig2.savefig(directory+'/'+'cov_conversion.pdf')
fig3.savefig(directory+'/'+'mc_xs.pdf')

text_file = open(directory+'/'+'plot_info', "w")
n = text_file.write(plot_info_string)
text_file.close()

# plt.show()
