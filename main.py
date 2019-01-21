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
    numOfEpochs = 100

    def __init__(self, hiddenLayers):

        self.trainigData, self.validationData, self.testData = mnistLoader.loadDataWrapper

        #Initializing biases and weights for each neuron in hidden layers
        self.biases = [np.random.rand(x, 1) for x in hiddenLayers] + [np.random.rand(self.numOfOutputs,1)]
        self.weights = [np.random.rand(hiddenLayers[0],self.numOfInputs)] + [np.random.rand(x,y) for x,y in zip(hiddenLayers[1:],hiddenLayers[:-1])] + [np.random.rand(self.numOfOutputs,hiddenLayers[-1])]

    def sigmoid(self, x):
        return 1.0/(1.0 + np.exp(-x))

    def feedForward(self, inputs):
        output = inputs
        for x in range(len(self.weights)):
            output = self.sigmoid(np.dot(self.weights[x], output) + self.biases[x])
        return output

    def setMiniBatches(self, trainingData):
        random.shuffle(trainingData)
        self.miniBatches = [trainingData[k:k+self.batchSize] for k in range(0, len(trainingData), self.batchSize)]

    def evaluateCost(self, miniBatchOutputs, miniBatchExpectedOutputs):
        print("tu sie ma liczyć i returnowac koszt")

    def updateNetwork(self, cost):
        print("Tu trzeba updatować wagi i progi")

    def processMiniBatch(self, miniBatch):
        miniBatchOutputs = []
        miniBatchExpectedOutputs = []
        #data is each input/expected output pair from miniBatch
        for data in miniBatch:
            miniBatchOutputs += self.feedForward(data[0])
            miniBatchExpectedOutputs += data[1]
        self.updateNetwork(self.evaluateCost(miniBatchOutputs,miniBatchExpectedOutputs))
        
    def teachNetwork(self):
        for epoch in self.numOfEpochs:
            self.setMiniBatches(self.trainigData)
            for miniBatch in self.miniBatches:
                self.processMiniBatch(miniBatch)