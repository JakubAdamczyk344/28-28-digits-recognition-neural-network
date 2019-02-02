#Neural network have changeable number of hidden layers and number of neurons in each hidden layer
# These vars are provided as a vector, each value stores number of neurons in each layer
#There will be 784 inputs (one for each pixel) and 10 outputs (one for each digit)
#Activation function will be sigmoid
#Weight and biases will be tuned using backpropagation algorithm

import numpy as np
import mnistLoader
import random

class NeuralNetwork:

    numOfInputs = 784
    numOfOutputs = 10
    batchSize = 10
    numOfEpochs = 1
    learningRate = 0.1

    def __init__(self, hiddenLayers):
        self.numberOfLayers = len(hiddenLayers) + 2
        self.trainigData, self.validationData, self.testData = mnistLoader.loadDataWrapper()

        #Initializing biases and weights for each neuron in hidden layers
        self.biases = [np.random.rand(x, 1) for x in hiddenLayers] + [np.random.rand(self.numOfOutputs,1)]
        self.weights = [np.random.rand(hiddenLayers[0],self.numOfInputs)] + [np.random.rand(x,y) for x,y in zip(hiddenLayers[1:],hiddenLayers[:-1])] + [np.random.rand(self.numOfOutputs,hiddenLayers[-1])]

    def sigmoid(self, x):
        return 1.0/(1.0 + np.exp(-x))

    def sigmoidPrime(self, x):
        return self.sigmoid(x)*(1-self.sigmoid(x))

    def costDerivative(self, outputActivations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (outputActivations-y)

    def feedForward(self, inputs):
        output = inputs
        for x in range(len(self.weights)):
            output = self.sigmoid(np.dot(self.weights[x], output) + self.biases[x])
        return output

    def setMiniBatches(self, trainingData):
        random.shuffle(trainingData)
        self.miniBatches = [trainingData[k:k+self.batchSize] for k in range(0, len(trainingData), self.batchSize)]

    def evaluateCost(self, miniBatch):
        outputs = []
        expectedOutputs = []
        for inputData, expectedOutput in miniBatch:
            outputs.append(self.feedForward(inputData))
            expectedOutputs.append(expectedOutput)
        results = zip(outputs,expectedOutputs)
        sum = 0
        for output, expectedOutput in results:
            sum += np.power(np.linalg.norm(expectedOutput - output),2)
            #sum += np.linalg.norm(expectedOutput - output)
        return sum/(2*self.batchSize)

    def backprop(self, inputData, expectedOutput):
        """Return a tuple ``(nablaB, nablaW)`` representing the
        gradient for the cost function C_x.  ``nablaB`` and
        ``nablaW`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nablaB = [np.zeros(b.shape) for b in self.biases]
        nablaW = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = inputData
        activations = [inputData] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.costDerivative(activations[-1], expectedOutput) * self.sigmoidPrime(zs[-1])
        nablaB[-1] = delta
        nablaW[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.numberOfLayers):
            z = zs[-l]
            sp = self.sigmoidPrime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nablaB[-l] = delta
            nablaW[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nablaB, nablaW)

    def updateMiniBatch(self, miniBatch):
        nablaB = [np.zeros(b.shape) for b in self.biases]
        nablaW = [np.zeros(w.shape) for w in self.weights]
        for inputData, expectedOutput in miniBatch:
            deltaNablaB, deltaNablaW = self.backprop(inputData, expectedOutput)
            nablaB = [nb+dnb for nb, dnb in zip(nablaB, deltaNablaB)]
            nablaW = [nw+dnw for nw, dnw in zip(nablaW, deltaNablaW)]
        self.weights = [w-(self.learningRate/len(miniBatch))*nw
                        for w, nw in zip(self.weights, nablaW)]
        self.biases = [b-(self.learningRate/len(miniBatch))*nb
                       for b, nb in zip(self.biases, nablaB)]

    def processMiniBatch(self, miniBatch):
        self.updateMiniBatch(miniBatch)
        print(self.evaluateCost(miniBatch))
        
    def teachNetwork(self):
        for epoch in range(self.numOfEpochs):
            self.setMiniBatches(self.trainigData)
            for miniBatch in self.miniBatches:
                self.processMiniBatch(miniBatch)

network = NeuralNetwork([6,6,6])
network.teachNetwork()