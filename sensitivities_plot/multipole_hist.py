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
E_0 = 3.674280 # [eV] Energy for histogram plots
z_0 = E_0**0.5
N_g = 10000 # Number of energy groups

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

# Plotting Paramaters
fig_alpha = 0.4 # Change opaqueness
num_bins = 1000

fig = plt.figure()
ax1 = fig.add_subplot(111)

n, bins, patches = ax1.hist(Σγ_hist, num_bins, density=1, 
                            label='σ(Γ, E_0) samples using Cov(Γ)', alpha=fig_alpha)

# add a 1st order sensitivity line - SLBW
Σγ_mean = exact_Σγ(E_0, μ_Γ_U238)
Σγ_sigma_at_E0 = (np.dot(exact_dΣγ_dΓ(E_0, μ_Γ_U238), \
                  np.dot(cov_Γ_U238,exact_dΣγ_dΓ(E_0, μ_Γ_U238))))**0.5
Σγ_local_propagation = normal(Σγ_mean, Σγ_sigma_at_E0, bins)

ax1.plot(bins, Σγ_local_propagation, '--r', label='1st order sensitivity analysis - Γ')

n, bins, patches = ax1.hist(Σγ_hist_Π, num_bins, density=1,
                            label='σ(Π, E_0) using Cov_Π from Sensitivity', alpha=fig_alpha)

# add a 1st order sensitivity line - Multipole
Π_Σγ_mean = multipole_Σ(z_0, μ_Π_U238)
mp_sens = multipole_dΣ_dΠ(z_0, μ_Π_U238)
Π_Σγ_sigma_at_E0 = sandwich(mp_sens, cov_Π_U238)**0.5
Π_Σγ_local_propagation = normal(Π_Σγ_mean, Π_Σγ_sigma_at_E0, bins)

ax1.plot(bins, Π_Σγ_local_propagation, '--b', label='1st order snesitivity analysis - Π')

n, bins, patches = ax1.hist(Σγ_hist_Π_mc, num_bins, density=1,
                            label='σ(Π, E_0) using Π from Γ sample conversion', alpha=fig_alpha)

# Labels
ax1.legend()
ax1.set_xlabel('Σγ(E_0) for E_0 = %s'%(E_0) )
ax1.set_ylabel('Probability density Histogram of Σγ(E_0)')
title_text = ('----------- Γ --------------- \n'
              'Mean Σγ(E_0) = {:f} \n'
              'Samples Mean Σγ(E_0) = {:f} \n'
              'Senstivity sigma_at_E0 = {:f} \n'
              'Samples strd_dev_Σγ = {:f} \n\n'
              '----------- Π --------------- \n'
              'Mean Σγ(E_0) = {:f} \n'
              'Samples Mean Σγ(E_0) = {:f} \n'
              'Senstivity sigma_at_E0 = {:f} \n'
              'Samples strd_dev_Σγ = {:f} \n\n'
              .format(Σγ_mean, mean_Σγ, Σγ_sigma_at_E0, strd_dev_Σγ,
                      Π_Σγ_mean, mean_Σγ_Π, Π_Σγ_sigma_at_E0, strd_dev_Σγ_Π,))
ax1.set_title(title_text)
# fig.tight_layout()
fig.subplots_adjust(top=0.6)

# ax2.set_xlabel('Σγ(E_0) for E_0 = %s'%(E_0) )
# ax2.set_ylabel('Probability density Histogram of Σγ(E_0)')
# ax2.set_title('Σmeanγ(E_0) = %s, meanΣγ(E_0) = %s, Σγ_sigma_at_E0 = %s, strd_dev_Σγ = %s'%(Σγ_mean, Π_Σγ_mean, Π_Σγ_sigma_at_E0, strd_dev_Σγ))

# Plot multipole covariance and correlation
fig2 = plt.figure()
ax4 = fig2.add_subplot(121)
im1 = ax4.imshow(cov_Π_U238)
ax4.set_title('Π Covariance')
ax4.figure.colorbar(im1)
ax5 = fig2.add_subplot(122)
im2 = ax5.imshow(correlation(cov_Π_U238))
ax5.figure.colorbar(im2)
ax5.set_title('Π Correlation (Blanks probably zero division)')
fig.tight_layout()

# # Plot sampled cross sections
# energies = np.logspace(-3,3,1000)
# fig2 = plt.figure()
# ax4 = fig2.add_subplot(111)
# mp_xs = [multipole_Σ(energy**0.5, μ_Π_U238) for energy in energies]
# ax4.plot(energies, mp_xs, label='mean')
# for i in range(100):
#     Π_for_plotting = join_complex(Π_samples[i]).reshape((2,2))
#     ax4.plot(energies, [multipole_Σ(energy**0.5, Π_for_plotting) for energy in energies], alpha=0.1)
# ax4.set_xscale('log')
# ax4.set_yscale('log')
# ax4.legend()

plt.show()
