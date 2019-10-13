import numpy as np
from data import ρ0

import scipy
import scipy.special

#def evaluate_Σγ(E, Γ): ## The most simple SLBW caputre resonance
#    """
#    Evaluate SLBW capture cross section
#
#    Parameters
#    ----------
#    Γ :  Resonance Parameters [Eλ resonance energy, Γn neutron width, Γγ gamma width]
#
#    Returns
#    -------
#    float : capture cross section
#    """
#    return (np.pi*Γ[1]*Γ[2]/(ρ0**2*E**0.5*Γ[0]**0.5))/((E-Γ[0])**2+((Γ[1]*(E/Γ[0])**(1/2)+ Γ[2])/2)**2)


def exact_Σγ(E, Γ): ## The most simple SLBW caputre resonance, with BS approximation
    """
        Evaluate SLBW capture cross section (with BS approximation)
        
        Parameters
        ----------
        Γ :  Resonance Parameters [Eλ resonance energy, Γn neutron width, Γγ gamma width]
        
        Returns
        -------
        float : capture cross section
        """
    return (np.pi*Γ[1]*Γ[2]/(ρ0**2*Γ[0]**0.5*E**0.5))/((Γ[0]-E)**2+(Γ[1]+ Γ[2])**2/4)


def exact_Real_SLBW(E, Γ , α): ## The OK Real[•] version of E-space Breit-Wigner profile
    """
        Evaluate SLBW capture cross section (with BS approximation)
        
        Parameters
        ----------
        Γ :  Resonance Parameters [Eλ resonance energy, Γn neutron width, Γγ gamma width]
        
        Returns
        -------
        float : capture cross section
        """
    ελ = Γ[0] - 1j*(Γ[1]+ Γ[2])/2
    return 1/(E**0.5)*np.real(α/(ελ - E))


def w_faddeeva(z): #this is only for integral representation purpose use only
    if np.imag(z) > 0 :
        return scipy.special.wofz(z)
    else :
        return - scipy.special.wofz(-z) # - np.conj(scipy.special.wofz(np.conj(z)))


def χ0(x):  ## the antisymmetric χ function at 0 K
    return x/(1 + x**2)

def χT_Faddeeva_approx(x,τ):  ## the antisymmetric χ function approximated at T K
    z = (x + 1j)/(2*(τ**0.5))
    return (np.pi/(4*τ))**0.5*np.imag(w_faddeeva(z))

def ψ0(x):  ## the symmetric ψ function at 0 K
    return 1/(1 + x**2)

def ψT_Faddeeva_approx(x,τ):  ## the symmetric ψ function approximated at T K
    z = (x + 1j)/(2*(τ**0.5))
    return (np.pi/(4*τ))**0.5*np.real(w_faddeeva(z))


def zeroK_exact_SLBW_χψ_Σγ(E, Γ, a , b): ## The OK Real[•] version of E-space Breit-Wigner profile
    """
        Evaluate SLBW capture cross section (with BS approximation)
        
        Parameters
        ----------
        Γ :  Resonance Parameters [Eλ resonance energy, Γn neutron width, Γγ gamma width]
        
        Returns
        -------
        float : capture cross section
        """
    Γtot = (Γ[1]+ Γ[2])
    x =  (Γ[0] - E)/(Γtot/2)
    return (a/(Γtot/2))/(E**0.5)*χ0(x) + (b/(Γtot/2))/(E**0.5)*ψ0(x)


def TK_approx_SLBW_χψ_Σγ(E, Γ, a, b, T): ## The OK Real[•] version of E-space Breit-Wigner profile
    """
        Evaluate SLBW capture cross section (with BS approximation)
        
        Parameters
        ----------
        Γ :  Resonance Parameters [Eλ resonance energy, Γn neutron width, Γγ gamma width]
        
        Returns
        -------
        float : capture cross section
        """
    kB = 8.617333262145*10**(-5)
    A = 238
    β = (kB * T /A)**0.5
    τ = 4*E*(β/(Γ[1]+ Γ[2]))**2
    Γtot = (Γ[1]+ Γ[2])
    x =  (Γ[0] - E)/(Γtot/2)
    z = (1 -1j*x)/(2*τ**0.5)
    return (a/(Γtot/2))/(E**0.5)*χT_Faddeeva_approx(x,τ) + (b/(Γtot/2))/(E**0.5)*ψT_Faddeeva_approx(x,τ)



def TK_approx_SLBW_Faddeeva_Σγ(E, Γ, α, T ): ## The OK Real[•] version of E-space Breit-Wigner profile
    """
        Evaluate SLBW capture cross section (with BS approximation)
        
        Parameters
        ----------
        Γ :  Resonance Parameters [Eλ resonance energy, Γn neutron width, Γγ gamma width]
        
        Returns
        -------
        float : capture cross section
        """
    kB = 8.617333262145*10**(-5)
    A = 238
    β = (kB * T /A)**0.5
    τ = 4*E*(β/(Γ[1]+ Γ[2]))**2

    ελ = Γ[0] - 1j*(Γ[1]+ Γ[2])/2
    x =  (Γ[0] - E)/((Γ[1]+ Γ[2])/2)
    Z0 = (np.conj(ελ) - E)/(2*β*E**0.5)
    return (1/E**0.5)*np.imag( (np.conj(α)*np.pi**0.5)/(2*β*E**0.5) * w_faddeeva(Z0) )

# SLBW Derivative Equations
def exact_dΣγ_dΓ(E, Γ):
    """
        Derivative of capture cross section with respect to resonance
        parameters
        
        Parameters
        ----------
        Γ :  Resonance Parameters [Eλ resonance energy, Γn neutron width, Γγ gamma width]
        
        Returns
        -------
        np.array : [dΣγ_dEλ , dΣγ_dΓn , dΣγ_dΓγ]
        """
    dΣγ_dEλ = exact_Σγ(E, Γ)*( (-1)/(2*Γ[0]) - 2*(Γ[0]-E)*exact_Σγ(E, Γ)*(ρ0**2*E**0.5*Γ[0]**(0.5))/(np.pi*Γ[1]*Γ[2]))
    dΣγ_dΓn = exact_Σγ(E, Γ)*( 1/(Γ[1]) - 1/2*(Γ[1] + Γ[2])*exact_Σγ(E, Γ)*(ρ0**2*E**0.5*Γ[0]**(0.5))/(np.pi*Γ[1]*Γ[2]))
    dΣγ_dΓγ = exact_Σγ(E, Γ)*( 1/(Γ[2]) - 1/2*(Γ[1] + Γ[2])*exact_Σγ(E, Γ)*(ρ0**2*E**0.5*Γ[0]**(0.5))/(np.pi*Γ[1]*Γ[2]))
    return np.array([dΣγ_dEλ , dΣγ_dΓn , dΣγ_dΓγ])


# z-space Multipole representation of BS approximation of SLBW

def exact_poles_and_residues(Γ):
    """
        Calcultes the exact poles and residues for the SLBW capture cross section (with BS approximation)
        
        Parameters
        ----------
        Γ :  Resonance Parameters [Eλ resonance energy, Γn neutron width, Γγ gamma width]
        
        Returns
        -------
        Π :  Multipole Parameters [{pole, residue}], for poles in the lower half of the complex plane (in the {E,+} sheet of the Rieman surface).
        """
    ελ = Γ[0] - 1j*(Γ[1]+Γ[2])/2
    r1 = 1j*np.pi*Γ[1]*Γ[2]/(ρ0**2*Γ[0]**0.5*(Γ[1]+ Γ[2]))
    p1 = np.sqrt(ελ)
    return np.array([ [ p1, r1 ] , [-p1 , r1] ])

def exact_poles_and_residues_differentials_dΠ_dΓ(Γ):
    """
        Calcultes the exact poles and residues for the SLBW capture cross section (with BS approximation)
        
        Parameters
        ----------
        Γ :  Resonance Parameters [Eλ resonance energy, Γn neutron width, Γγ gamma width]
        
        Returns
        -------
        dΠ_dΓ :  Multipole Parameters [{pole, residue}], for poles in the lower half of the complex plane (in the {E,+} sheet of the Rieman surface).
        """
    ελ = Γ[0] - 1j*(Γ[1]+Γ[2])/2
    r1 = 1j*np.pi*Γ[1]*Γ[2]/(ρ0**2*Γ[0]**0.5*(Γ[1]+ Γ[2]))
    p1 = np.sqrt(ελ)

    dp1_dEλ = 1/(2*p1)
    dr1_dEλ = - r1/(2*Γ[0])

    dp1_dΓn = -1j/(4*p1)
    dr1_dΓn = r1*( 1/Γ[1] - 1/(Γ[1]+Γ[2]) )

    dp1_dΓγ = -1j/(4*p1)
    dr1_dΓγ = r1*( 1/Γ[2] - 1/(Γ[1]+Γ[2]) )

    dΠ_dΓ = np.array([ [[dp1_dEλ, dr1_dEλ] , [-dp1_dEλ , dr1_dEλ]] , [[dp1_dΓn, dr1_dΓn] , [-dp1_dΓn , dr1_dΓn]] , [[dp1_dΓγ, dr1_dΓγ] , [-dp1_dΓγ , dr1_dΓγ]] ])
    
#    dp_dΓ = np.array([ [ dp1_dEλ , -dp1_dEλ] , [dp1_dΓn , -dp1_dΓn] , [dp1_dΓγ,-dp1_dΓγ] ])
#    dr_dΓ = np.array([ [ dr1_dEλ , dr1_dEλ] , [dr1_dΓn , dr1_dΓn] , [dr1_dΓγ, dr1_dΓγ] ])

    dp_dΓ = np.array([ [ dp1_dEλ , dp1_dΓn , dp1_dΓγ] , [ -dp1_dEλ , -dp1_dΓn , -dp1_dΓγ] ])
    dr_dΓ = np.array([ [ dr1_dEλ , dr1_dΓn , dr1_dΓγ] , [ dr1_dEλ , dr1_dΓn , dr1_dΓγ] ])
    return dp_dΓ , dr_dΓ


def multipole_Σ(z, Π): ## The most simple SLBW caputre resonance, with BS approximation
    """
        Evaluate SLBW capture cross section (with BS approximation)
        
        Parameters
        ----------
        z : square-root-ofenergy (eV) (on {E,+} sheet of Riemann surface)
        Π :  Multipole Parameters [{pole, residue}], for poles in the lower half of the complex plane (in the {E,+} sheet of the Rieman surface).
        
        Returns
        -------
        float : capture cross section
        """
    return  (1/z**2) * np.real(sum( Π[j][1]/(z-Π[j][0]) for j in range(Π.shape[0]))) #+ np.conj(Π[j][1])/(z-np.conj(Π[j][0])) for j in range(Π.shape[0]) ) )


def approx_multipole_Doppler_Σ(z, Π, T): ## The most simple SLBW caputre resonance, with BS approximation
    """
        Evaluate SLBW capture cross section (with BS approximation)
        
        Parameters
        ----------
        z : square-root-ofenergy (eV) (on {E,+} sheet of Riemann surface)
        Π :  Multipole Parameters [{pole, residue}], for poles in the lower half of the complex plane (in the {E,+} sheet of the Rieman surface).
        
        Returns
        -------
        float : capture cross section
        """
    kB = 8.617333262145*10**(-5)
    A = 238
    β = (kB * T /A)**0.5
    return  (1/z**2) * np.real( (np.pi**0.5/(1j*β)) * sum( Π[j][1] * w_faddeeva( ((z-Π[j][0])/β) ) for j in range(Π.shape[0])))




def multipole_dΣ_dΓ(z, Π, dp_dΓ, dr_dΓ): ## The most simple SLBW caputre resonance, with BS approximation
    """
        Evaluate SLBW (with BS approximation) capture cross section differential with respect to resonance parameters Γ
        
        Parameters
        ----------
        z : square-root-ofenergy (eV) (on {E,+} sheet of Riemann surface)
        Π :  Multipole Parameters [{pole, residue}], for poles in the lower half of the complex plane (in the {E,+} sheet of the Rieman surface).
        dΠ_dΓ : Jacobians of poles and residues with respect to all the resonance parameters Γ [{d_pole_dΓ, d_residue_dΓ}]
        Returns
        -------
        array (size of resonance parameters) : differential capture cross sections in multipole representation
        """
    return  np.array([ (1/z**2) * np.real(sum( dr_dΓ[j][i]/(z-Π[j][0]) + Π[j][1]*dp_dΓ[j][i]/(z-Π[j][0])**2 for j in range(Π.shape[0]))) for i in range(dp_dΓ.shape[1])])


def multipole_dΣ_dΠ(z, Π): ## The most simple SLBW caputre resonance, with BS approximation
    """
        Evaluate SLBW (with BS approximation) capture cross section differential with respect to multipoles (represented as a set of twice as much real parameters (the real and imaginary part of each multipole complex parameter))
        
        Parameters
        ----------
        z : square-root-ofenergy (eV) (on {E,+} sheet of Riemann surface)
        Π :  Multipole Parameters [{pole, residue}], for poles in the lower half of the complex plane (in the {E,+} sheet of the Rieman surface).

        Returns
        -------
        array (size of resonance multipoles times two) : differential capture cross sections (functions of z) [ dΣ_dRe[p] , dΣ_dIm[p] , dΣ_dRe[r] , dΣ_dIm[r] ]
        """
    dΣ_dRe_p = np.array([ (1/z**2) * np.real( Π[j][1]/(z-Π[j][0])**2 ) for j in range(Π.shape[0]) ])
    dΣ_dIm_p = np.array([ (1/z**2) * np.real( 1j*Π[j][1]/(z-Π[j][0])**2 ) for j in range(Π.shape[0]) ])
    dΣ_dRe_r = np.array([ (1/z**2) * np.real( 1.0/(z-Π[j][0]) ) for j in range(Π.shape[0]) ])
    dΣ_dIm_r = np.array([ (1/z**2) * np.real( 1j/(z-Π[j][0])) for j in range(Π.shape[0]) ])
    
    return  np.concatenate( ( dΣ_dRe_p, dΣ_dIm_p,  dΣ_dRe_r, dΣ_dIm_r ) )


def multipoles_real_vector_Π(Π):
    """
        Take a set of multipoles (p_j, r_j), and convert then into a vector of real parameters multipole: [Re[p_j] ; Im[p_j] ; Re[r_j] ; Im[r_j]]
        
        inputs
        ----------
        Π :  Multipole Parameters [{pole, residue}], for poles in the lower half of the complex plane (in the {E,+} sheet of the Rieman surface).
        """
    N_p = Π.shape[0]
    Π_real_vector = np.zeros(4*N_p)
    for j in range(Π.shape[0]):
        Π_real_vector[j] = np.real(Π[j][0])
        Π_real_vector[N_p + j] = np.imag(Π[j][0])
        Π_real_vector[2*N_p + j] = np.real(Π[j][1])
        Π_real_vector[3*N_p + j] = np.imag(Π[j][1])
    return Π_real_vector

def multipoles_set_Π(Π_real_vector):
    """
        Take a set of multipoles (p_j, r_j), and convert then into a vector of real parameters multipole: [Re[p_j] ; Im[p_j] ; Re[r_j] ; Im[r_j]]
        
        inputs
        ----------
        Π :  Multipole Parameters [{pole, residue}], for poles in the lower half of the complex plane (in the {E,+} sheet of the Rieman surface).
        """
    N_p = np.int(Π_real_vector.shape[0]/4)
    Π = complex(0.0)*np.zeros([N_p , 2])
    for j in range(Π.shape[0]):
        Π[j][0] = Π_real_vector[j] + 1j*Π_real_vector[N_p + j]
        Π[j][1] = Π_real_vector[2*N_p + j] + 1j*Π_real_vector[3*N_p + j]
    return Π


def z_space_Σγ(z, Γ): ## The most simple SLBW caputre resonance, with BS approximation
    """
        Evaluate SLBW capture cross section (with BS approximation)
        
        Parameters
        ----------
        E : energy (eV)
        Γ :  Resonance Parameters [mean, neutron width, gamma width]
        
        Returns
        -------
        float : capture cross section
        """
    return (np.pi*Γ[1]*Γ[2]/(ρ0**2*Γ[0]**0.5*z))/((Γ[0]-z**2)**2+(Γ[1]+ Γ[2])**2/4)



def analytic_Σγ(z, Γ): ## The most simple SLBW caputre resonance, with BS approximation
    """
        Evaluate SLBW capture cross section (with BS approximation)
        
        Parameters
        ----------
        z : square-root-ofenergy (eV) (on {E,+} sheet of Riemann surface)
        Γ :  Resonance Parameters [mean, neutron width, gamma width]
        
        Returns
        -------
        float : capture cross section
        """
    ελ = Γ[0] - 1j*(Γ[1]+Γ[2])/2
    b = (2*np.pi*Γ[1]*Γ[2])/(ρ0**2*Γ[0]**0.5*(Γ[1]+ Γ[2]))
    return  1/z * ( (-1j*b)/(ελ - z**2) + np.conj((-1j*b))/(np.conj(ελ) - z**2) )/2

def analytic_dΣγ_dΓ(z, Γ):
    """
        Derivative of SLBW (with BS approximation) capture cross section with respect to resonance parameters Γ
        
        Parameters
        ----------
        z : square-root-ofenergy (eV) (on {E,+} sheet of Riemann surface)
        Γ :  Resonance Parameters [mean, neutron width, gamma width]
        
        Returns
        -------
        np.array : [dΣγ_dEλ , dΣγ_dΓn , dΣγ_dΓγ]
        """
    ελ = Γ[0] - 1j*(Γ[1]+Γ[2])/2
    dΣγ_dEλ = analytic_Σγ(z, Γ)*( (-1)/(2*Γ[0]) - (1j/(ελ - z**2)**2  + np.conj(1j)/(np.conj(ελ) - z**2)**2)/( 1j/(ελ - z**2) + np.conj(1j)/(np.conj(ελ) - z**2) ) )
    dΣγ_dΓn = analytic_Σγ(z, Γ)*( 1/Γ[1] - 1/(Γ[1]+Γ[2]) - 1/2*(1/(ελ - z**2)**2  + 1/(np.conj(ελ) - z**2)**2)/( 1j/(ελ - z**2) + np.conj(1j)/(np.conj(ελ) - z**2) ) )
    dΣγ_dΓγ = analytic_Σγ(z, Γ)*( 1/Γ[2] - 1/(Γ[1]+Γ[2]) - 1/2*(1/(ελ - z**2)**2  + 1/(np.conj(ελ) - z**2)**2)/( 1j/(ελ - z**2) + np.conj(1j)/(np.conj(ελ) - z**2) ) )
    return np.array([dΣγ_dEλ , dΣγ_dΓn , dΣγ_dΓγ])


