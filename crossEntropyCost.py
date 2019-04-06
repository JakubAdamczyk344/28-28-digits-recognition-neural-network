import numpy as np

def countCost(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

def countDelta(z, a, y):
        return (a-y)