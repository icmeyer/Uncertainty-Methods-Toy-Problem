import numpy as np
from data import ρ0

#def evaluate_Σγ(E, Γ): ## The most simple SLBW caputre resonance
#    """
#    Evaluate SLBW capture cross section
#
#    Parameters
#    ----------
#    E : energy (eV)
#    Γ :  Resonance Parameters [mean, neutron width, gamma width]
#
#    Returns
#    -------
#    float : capture cross section
#    """
#    return (np.pi*Γ[1]*Γ[2]/(ρ0**2*E**0.5*Γ[0]**0.5))/((E-Γ[0])**2+((Γ[1]*(E/Γ[0])**(1/2)+ Γ[2])/2)**2)


def evaluate_Σγ(E, Γ): ## The most simple SLBW caputre resonance, with BS approximation
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
    return (np.pi*Γ[1]*Γ[2]/(ρ0**2*Γ[0]**0.5*E**0.5))/((Γ[0]-E)**2+(Γ[1]+ Γ[2])**2)


## SLBW Derivative Equations
#def dΣγ_dΓ(E, Γ):
#    """
#    Derivative of capture cross section with respect to resonance
#    parameters
#
#    Parameters
#    ----------
#    E : Energy (eV)
#    Γ :  Resonance Parameters [mean, neutron width, gamma width]
#
#    Returns
#    -------
#    np.array : [dΣγ_dEλ , dΣγ_dΓn , dΣγ_dΓγ]
#    """
#    dΣγ_dEλ = evaluate_Σγ(E, Γ)*( (-1)/(2*Γ[0]) + evaluate_Σγ(E, Γ)*(ρ0**2*E**0.5*Γ[0]**(0.5))/(np.pi*Γ[1]*Γ[2])*(2*(E-Γ[0]) + ( Γ[2] + Γ[1]*E**0.5*Γ[0]**(-0.5))*(Γ[2]*E**0.5)/(4*Γ[0]**1.5)) )
#    dΣγ_dΓn = evaluate_Σγ(E, Γ)*( 1/(Γ[1]) - evaluate_Σγ(E, Γ)*(ρ0**2*E**0.5*Γ[0]**(0.5))/(np.pi*Γ[1]*Γ[2])*(0.5*E**0.5*Γ[0]**(-0.5)*( Γ[2] + Γ[1]*E**0.5*Γ[0]**(-0.5))) )
#    dΣγ_dΓγ = evaluate_Σγ(E, Γ)*( 1/(Γ[2]) - evaluate_Σγ(E, Γ)*(ρ0**2*E**0.5*Γ[0]**(0.5))/(np.pi*Γ[1]*Γ[2])*(0.5*( Γ[2] + Γ[1]*E**0.5*Γ[0]**(-0.5))) )
#    return np.array([dΣγ_dEλ , dΣγ_dΓn , dΣγ_dΓγ])


# SLBW Derivative Equations
def dΣγ_dΓ(E, Γ):
    """
        Derivative of capture cross section with respect to resonance
        parameters
        
        Parameters
        ----------
        E : Energy (eV)
        Γ :  Resonance Parameters [resonance energy, neutron width, gamma width]
        
        Returns
        -------
        np.array : [dΣγ_dEλ , dΣγ_dΓn , dΣγ_dΓγ]
        """
    dΣγ_dEλ = evaluate_Σγ(E, Γ)*( (-1)/(2*Γ[0]) + 2*(E-Γ[0])*evaluate_Σγ(E, Γ)*(ρ0**2*E**0.5*Γ[0]**(0.5))/(np.pi*Γ[1]*Γ[2]))
    dΣγ_dΓn = evaluate_Σγ(E, Γ)*( 1/(Γ[1]) - 2*(Γ[1] + Γ[2])*evaluate_Σγ(E, Γ)*(ρ0**2*E**0.5*Γ[0]**(0.5))/(np.pi*Γ[1]*Γ[2]))
    dΣγ_dΓγ = evaluate_Σγ(E, Γ)*( 1/(Γ[2]) - 2*(Γ[1] + Γ[2])*evaluate_Σγ(E, Γ)*(ρ0**2*E**0.5*Γ[0]**(0.5))/(np.pi*Γ[1]*Γ[2]))
    return np.array([dΣγ_dEλ , dΣγ_dΓn , dΣγ_dΓγ])


# z-space Multipole representation of BS approximation of SLBW

def exact_poles_and_residues(Γ):
    """
        Calculta the exact poles and residues for the SLBW capture cross section (with BS approximation)
        
        Parameters
        ----------
        Γ :  Resonance Parameters [mean, neutron width, gamma width]
        
        Returns
        -------
        Π :  Multipole Parameters [{pole, residue}], for poles in the lower half of the complex plane (in the {E,+} sheet of the Rieman surface).
        """
    ελ = Γ[0] + 1j*(Γ[1]+Γ[2])
    r1 = -1j*np.pi*Γ[1]*Γ[2]/(2*ρ0**2*Γ[0]**0.5*(Γ[1]+ Γ[2]))
    p1 = np.sqrt(ελ)
    return np.array([ [ p1, r1 ] , [-p1 , r1] ])


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
    return (np.pi*Γ[1]*Γ[2]/(ρ0**2*Γ[0]**0.5*z))/((Γ[0]-z**2)**2+(Γ[1]+ Γ[2])**2)



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
    ελ = Γ[0] + 1j*(Γ[1]+Γ[2])
    return (np.pi*Γ[1]*Γ[2]/(ρ0**2*Γ[0]**0.5*z*(Γ[1]+ Γ[2])))* ( 1j/(ελ - z**2) + np.conj(1j)/(np.conj(ελ) - z**2) )/2

def analytic_dΣγ_dΓ(z, Γ):
    """
        Derivative of capture cross section with respect to resonance
        parameters
        
        Parameters
        ----------
        z : square-root-ofenergy (eV) (on {E,+} sheet of Riemann surface)
        Γ :  Resonance Parameters [mean, neutron width, gamma width]
        
        Returns
        -------
        np.array : [dΣγ_dEλ , dΣγ_dΓn , dΣγ_dΓγ]
        """
    ελ = Γ[0] + 1j*(Γ[1]+Γ[2])
    dΣγ_dEλ = analytic_Σγ(z, Γ)*( (-1)/(2*Γ[0]) + (-1j/(ελ - z**2)**2  + np.conj(-1j)/(np.conj(ελ) - z**2)**2)/( 1j/(ελ - z**2) + np.conj(1j)/(np.conj(ελ) - z**2) ) )
    dΣγ_dΓn = analytic_Σγ(z, Γ)*( 1/Γ[1] - (-1j/(ελ - z**2)**2  + 1j/(np.conj(ελ) - z**2)**2)/( 1j/(ελ - z**2) - 1j/(np.conj(ελ) - z**2) ) )
    dΣγ_dΓγ = analytic_Σγ(z, Γ)*( 1/Γ[2] - (-1j/(ελ - z**2)**2  + 1j/(np.conj(ελ) - z**2)**2)/( 1j/(ελ - z**2) - 1j/(np.conj(ελ) - z**2) ) )
    return np.array([dΣγ_dEλ , dΣγ_dΓn , dΣγ_dΓγ])


