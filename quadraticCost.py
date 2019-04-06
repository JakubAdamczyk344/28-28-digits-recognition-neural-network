import numpy as np
import miscellaneous as misc

def countCost(a, y):
        return 0.5*np.linalg.norm(a-y)**2

def countDelta(z, a, y):
        return (a-y) * misc.sigmoidPrime(z)