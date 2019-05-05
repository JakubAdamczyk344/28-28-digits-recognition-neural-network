#Neural network have changeable number of hidden layers and number of neurons in each hidden layer
# These vars are provided as a vector, each value stores number of neurons in each layer
#There will be 784 inputs (one for each pixel) and 10 outputs (one for each digit)
#Activation function will be sigmoid
#Weight and biases will be tuned using backpropagation algorithm

import numpy as np
import mnistLoader
import random
import matplotlib.pyplot as plt
import miscellaneous as misc
import quadraticCost as qc
import crossEntropyCost as cec
from miscellaneous import defaultWeightsInitializer as defInit
from miscellaneous import squeezedWeightsInitializer as sqzInit

class NeuralNetwork:

    numOfInputs = 784
    numOfOutputs = 10
    batchSize = 10
    numOfEpochs = 10
    learningRate = 0.1

    def __init__(self, hiddenLayers, cost=cec, weightsInit=sqzInit, lmbda=0.0):
        self.cost = cost
        self.numberOfLayers = len(hiddenLayers) + 2
        self.trainigData, self.validationData, self.testData = mnistLoader.loadDataWrapper()
        #Regularization factor
        self.lmbda = lmbda

        #Initializing biases and weights for each neuron in hidden layers
        self.biases = [np.random.randn(x, 1) for x in hiddenLayers] + [np.random.randn(self.numOfOutputs,1)]
        self.weights = weightsInit(hiddenLayers,self.numOfInputs,self.numOfOutputs)

    def feedForward(self, inputs):
        output = inputs
        for x in range(len(self.weights)):
            output = misc.sigmoid(np.dot(self.weights[x], output) + self.biases[x])
        return output

    def setMiniBatches(self, trainingData):
        random.shuffle(trainingData)
        self.miniBatches = [trainingData[k:k+self.batchSize] for k in range(0, len(trainingData), self.batchSize)]

    def testNetwork(self, testData):
        """Neural network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        testResults = [(np.argmax(self.feedForward(inputData)), expectedOutput) for (inputData, expectedOutput) in testData]
        return sum(int(inputData == expectedOutput) for (inputData, expectedOutput) in testResults)

    def evaluateCost(self, miniBatch):
        outputs = []
        expectedOutputs = []
        for inputData, expectedOutput in miniBatch:
            outputs.append(self.feedForward(inputData))
            expectedOutputs.append(expectedOutput)
        results = zip(outputs,expectedOutputs)
        cost = 0
        for output, expectedOutput in results:
            cost += (self.cost).countCost(output,expectedOutput)
        cost /= len(miniBatch)
        return cost

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
            activation = misc.sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = (self.cost).countDelta(zs[-1], activations[-1], expectedOutput)
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
            sp = misc.sigmoidPrime(z)
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
        self.weights = [(1-self.learningRate*(self.lmbda/len(self.trainigData)))*w-(self.learningRate/len(miniBatch))*nw
                        for w, nw in zip(self.weights, nablaW)]
        self.biases = [b-(self.learningRate/len(miniBatch))*nb
                        for b, nb in zip(self.biases, nablaB)]
        
    def teachNetwork(self):
        numberOfBatch = []
        batchError = []
        numberOfEpoch = []
        percentOfCorrectAnswers = []
        batchNumber = 0
        epochNumber = 0
        for epoch in range(self.numOfEpochs):
            self.setMiniBatches(self.trainigData)
            for miniBatch in self.miniBatches:
                batchNumber += 1
                numberOfBatch.append(batchNumber)
                self.updateMiniBatch(miniBatch)
                batchError.append(self.evaluateCost(miniBatch))
            epochNumber += 1
            numberOfEpoch.append(epochNumber)
            percentOfCorrectAnswers.append(self.testNetwork(self.testData))
        
        fig1 = plt.figure(1) 
        plt.plot(numberOfBatch,batchError)
        fig1.show()
        fig2 = plt.figure(2)
        plt.plot(numberOfEpoch,percentOfCorrectAnswers)
        fig2.show()
        input()


network = NeuralNetwork([30],cec,sqzInit,5)
network.teachNetwork()