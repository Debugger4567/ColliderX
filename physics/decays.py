import math, random, numpy as np
from .kinematics import FourVector

def two_body_decay(parent_mass, m1, m2):
    """
    2-body isotropic decay. Returns [FourVector, FourVector].
    """
    p = math.sqrt(
        max(
            0,
            (parent_mass**2 - (m1 + m2) ** 2) *
            (parent_mass**2 - (m1 - m2) ** 2),
        )
    ) / (2 * parent_mass)

    costheta = random.uniform(-1, 1)
    sintheta = math.sqrt(1 - costheta**2)
    phi = random.uniform(0, 2 * math.pi)

    px, py, pz = p * sintheta * math.cos(phi), p * sintheta * math.sin(phi), p * costheta

    E1, E2 = math.sqrt(m1**2 + p**2), math.sqrt(m2**2 + p**2)

    return [
        FourVector(E1, px, py, pz),
        FourVector(E2, -px, -py, -pz),
    ]


def three_body_decay(parent_mass, m1, m2, m3, n_events=1):
    """
    3-body Raubold–Lynch algorithm. Returns np.array of events.
    Each event = [FourVector1, FourVector2, FourVector3].
    """
    events = []
    for _ in range(n_events):
        m12_min, m12_max = m1 + m2, parent_mass - m3
        m12 = math.sqrt(random.uniform(m12_min**2, m12_max**2))

        # first stage: parent -> X + p3
        E3 = (parent_mass**2 + m3**2 - m12**2) / (2 * parent_mass)
        p3 = math.sqrt(max(0, E3**2 - m3**2))
        p3_vec = FourVector(E3, 0, 0, p3)
        X = FourVector(parent_mass - E3, 0, 0, -p3)

        # second stage: X -> p1 + p2
        p = math.sqrt(max(0, (m12**2 - (m1 + m2) ** 2) * (m12**2 - (m1 - m2) ** 2))) / (2 * m12)

        costheta = random.uniform(-1, 1)
        sintheta = math.sqrt(1 - costheta**2)
        phi = random.uniform(0, 2 * math.pi)

        px, py, pz = p * sintheta * math.cos(phi), p * sintheta * math.sin(phi), p * costheta
        E1, E2 = math.sqrt(m1**2 + p**2), math.sqrt(m2**2 + p**2)

        p1, p2 = FourVector(E1, px, py, pz), FourVector(E2, -px, -py, -pz)

        # boost p1, p2 into parent frame
        bx, by, bz = X.px / X.E, X.py / X.E, X.pz / X.E
        p1, p2 = p1.boost(bx, by, bz), p2.boost(bx, by, bz)

        events.append([p1, p2, p3_vec])

    return np.array(events, dtype=object)


def decay_particle(parent_mass, daughter_masses):
    """
    Dispatcher: chooses 2- or 3-body decay.
    Always returns list of FourVectors.
    """
    if len(daughter_masses) == 2:
        return two_body_decay(parent_mass, *daughter_masses)
    elif len(daughter_masses) == 3:
        return list(three_body_decay(parent_mass, *daughter_masses, n_events=1)[0])
    else:
        raise NotImplementedError(f"{len(daughter_masses)}-body decay not supported")
