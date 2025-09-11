import numpy as np

def check_charge(before, after):
    Q_before = np.sum(before['Charge'])
    Q_after = np.sum(after['Charge'])
    return Q_before == Q_after

def check_baryon(before, after):
    B_before = np.sum(before['Baryon_Number'])
    B_after = np.sum(after['Baryon_Number'])
    return B_before == B_after