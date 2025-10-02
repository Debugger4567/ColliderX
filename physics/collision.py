from .particles import Particle
from .decays import decay_particle

def simulate_decay_chain(particle_name, decay_table, depth=0, max_depth=5):
    """
    Simulate recursive decay chain.
    Returns a list of FourVectors (final stable particles).
    """
    if depth > max_depth:
        return []

    daughters = decay_table.get(particle_name, [])
    if not daughters:
        # Stable particle
        return [Particle(particle_name).fourvec]

    parent_mass = Particle.lookup_mass(particle_name)
    daughter_masses = [Particle.lookup_mass(d) for d in daughters]

    products = decay_particle(parent_mass, daughter_masses)

    final_states = []
    for daughter_name, fv in zip(daughters, products):
        if daughter_name in decay_table:
            final_states.extend(simulate_decay_chain(daughter_name, decay_table, depth+1, max_depth))
        else:
            final_states.append(fv)
    return final_states
