import numpy as np
from data import ρ0

def evaluate_Σγ(E, Γ): ## The most simple SLBW caputre resonance
    """
    Evaluate SLBW capture cross section

    Parameters
    ----------
    E : energy (eV)
    Γ :  Resonance Parameters [mean, neutron width, gamma width]

    Returns
    -------
    float : capture cross section
    """
    return (np.pi*Γ[1]*Γ[2]/(ρ0**2*E**0.5*Γ[0]**0.5))/((E-Γ[0])**2+((Γ[1]*(E/Γ[0])**(1/2)+ Γ[2])/2)**2)

# SLBW Derivative Equations
def dΣγ_dΓ(E, Γ):
    """
    Derivative of capture cross section with respect to resonance
    parameters

    Parameters
    ----------
    E : Energy (eV)
    Γ :  Resonance Parameters [mean, neutron width, gamma width]

    Returns
    -------
    np.array : [dΣγ_dEλ , dΣγ_dΓn , dΣγ_dΓγ]
    """
    dΣγ_dEλ = evaluate_Σγ(E, Γ)*( (-1)/(2*Γ[0]) + evaluate_Σγ(E, Γ)*(ρ0**2*E**0.5*Γ[0]**(0.5))/(np.pi*Γ[1]*Γ[2])*(2*(E-Γ[0]) + ( Γ[2] + Γ[1]*E**0.5*Γ[0]**(-0.5))*(Γ[2]*E**0.5)/(4*Γ[0]**1.5)) )
    dΣγ_dΓn = evaluate_Σγ(E, Γ)*( 1/(Γ[1]) - evaluate_Σγ(E, Γ)*(ρ0**2*E**0.5*Γ[0]**(0.5))/(np.pi*Γ[1]*Γ[2])*(0.5*E**0.5*Γ[0]**(-0.5)*( Γ[2] + Γ[1]*E**0.5*Γ[0]**(-0.5))) )
    dΣγ_dΓγ = evaluate_Σγ(E, Γ)*( 1/(Γ[2]) - evaluate_Σγ(E, Γ)*(ρ0**2*E**0.5*Γ[0]**(0.5))/(np.pi*Γ[1]*Γ[2])*(0.5*( Γ[2] + Γ[1]*E**0.5*Γ[0]**(-0.5))) )
    return np.array([dΣγ_dEλ , dΣγ_dΓn , dΣγ_dΓγ])
    

