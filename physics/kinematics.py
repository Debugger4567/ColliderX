import numpy as np

def invariant_mass(E, p):
    """Calculate invariant mass from energy E and momentum vector p"""
    return np.sqrt(E**2 - np.dot(p, p))