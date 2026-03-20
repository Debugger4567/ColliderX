import numpy as np

def sample_relativistic_bw(
        m0: float,
        gamma: float, 
        rng: np.random.Generator,
        m_min: float | None = None,
        m_max: float | None = None,
):
    """
    Relativistic Breit–Wigner sampling (MeV units):
      P(m) ∝ (m² Γ) / [ (m² − m0²)² + (m Γ)² ]

    Parameters
    ----------
    m0 : float         Pole mass (MeV)
    gamma : float      Width Γ (MeV)
    m_min, m_max :     Optional hard bounds (MeV)

    Returns
    -------
    float              Sampled mass (MeV)
    """
    # Use standard tricK: sample m^2 via Cauchy-like variable 
    while True:
        # sameple s = m^2
        y=rng.standard_cauchy()
        s = m0*m0 + m0*gamma*y

        if s <=0:
            continue

        m = np.sqrt(s)

        if m_min is not None and m < m_min:
            continue
        if m_max is not None and m > m_max:
            continue

        return m