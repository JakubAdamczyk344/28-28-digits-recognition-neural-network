#Neural network have changeable number of hidden layers and number of neurons in each hidden layer
# These vars are provided as a vector, each value stores number of neurons in each layer
#There will be 784 inputs (one for each pixel) and 10 outputs (one for each digit)
#Activation fucntion will be sigmoid
#Weight and biases will be tuned using backpropagation algorithm
#We can change size of batches and number of epochs

import numpy as np

class NeuralNetwork:

    numOfInputs = 2
    numOfOutputs = 2
    inputs = [[2],[1]]

    def __init__(self, hiddenLayers, numOfExamplesInBatch, numOfEpochs):

        #self.inputs = ["tu trzeba zaciągać dane, znajdź przykład w necie"]

        #Initializing biases and weights for each neuron in hidden layers
        #self.biases = [np.random.rand(x, 1) for x in hiddenLayers] + [np.random.rand(self.numOfOutputs,1)]
        self.biases = [np.around(np.random.rand(x, 1),1) for x in hiddenLayers] + [np.around(np.random.rand(self.numOfOutputs,1),1)]
        print("biases:")
        print(self.biases)
        #self.weights = [np.random.rand(hiddenLayers[0],self.numOfInputs)] + [np.random.rand(x,y) for x,y in zip(hiddenLayers[1:],hiddenLayers[:-1])] + [np.random.rand(self.numOfOutputs,hiddenLayers[-1])]
        self.weights = [np.around(np.random.rand(hiddenLayers[0],self.numOfInputs),1)] + [np.around(np.random.rand(x,y),1) for x,y in zip(hiddenLayers[1:],hiddenLayers[:-1])] + [np.around(np.random.rand(self.numOfOutputs,hiddenLayers[-1]),1)]
        print("weights:")
        print(self.weights)

    def sigmoid(self, x):
        return 1.0/(1.0 + np.exp(-x))

    def feedForward(self):
        output = self.inputs
        print(output)
        for x in range(len(self.weights)):
            output = np.dot(self.weights[x], output) + self.biases[x]
            print(output)
        print("output:")
        print(output)
            

Network = NeuralNetwork([2,2], 10, 100)
Network.feedForward()