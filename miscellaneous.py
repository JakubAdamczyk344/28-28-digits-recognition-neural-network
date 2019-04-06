import numpy as np

def sigmoid(x):
        return 1.0/(1.0 + np.exp(-x))

def sigmoidPrime(x):
        return sigmoid(x)*(1-sigmoid(x))