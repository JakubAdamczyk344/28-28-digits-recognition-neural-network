import numpy as np

def sigmoid(x):
        return 1.0/(1.0 + np.exp(-x))

def sigmoidPrime(x):
        return sigmoid(x)*(1-sigmoid(x))

def defaultWeightsInitializer(hiddenLayers,numOfInputs,numOfOutputs):
        return  [np.random.randn(hiddenLayers[0],numOfInputs)] + [np.random.randn(x,y) for x,y in zip(hiddenLayers[1:],hiddenLayers[:-1])] + [np.random.randn(numOfOutputs,hiddenLayers[-1])]

def squeezedWeightsInitializer(hiddenLayers,numOfInputs,numOfOutputs):
        return  [np.random.randn(hiddenLayers[0],numOfInputs)/np.sqrt(numOfInputs)] + [np.random.randn(x,y)/np.sqrt(y) for x,y in zip(hiddenLayers[1:],hiddenLayers[:-1])] + [np.random.randn(numOfOutputs,hiddenLayers[-1])/np.sqrt(hiddenLayers)]