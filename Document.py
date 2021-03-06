import os
import itertools
import math

os.system('clear')


class Network:
    deltas = {}
    layerOutputs = {}

    def __init__(self, layers, weights, biases, learningRate=0.1, toleranceValue=0.01):
        """
        :param layers: The layers of the neural network
        :param weights: The weights of the paths between nodes in adjacent layers
        :param biases: The biases to be applied into each node
        :param learningRate: The rate at which the neural network adjusts its weight
        :param toleranceValue: The maximum error allowed before the network is accepted as correct
        """
        self.layers = layers
        self.weights = weights
        self.learningRate = learningRate
        self.toleranceValue = toleranceValue
        self.biases = biases

    def learnOn(self, data):
        """
        :param data: The data on which the network should learn ([input1, input2, ..., inputn], [output1, output2, ..., inputn])
        """

        rounds = 1 # A count of the number of epochs made
        while (True):
            sumOfSquaredErrors = 0 # The sum of squared errors

            for (inputVector, targetValues) in ann_data:
                # seed input layers with input vector
                # The outputs of the first layer are equal to the input layer
                layerOutputs = dict(zip(layers[0], inputVector)) 
                # The inputs of the first layer are equal to the input layer
                layerInputs = dict(zip(layers[0], inputVector))

                for inputLayers, outputLayers in zip(layers, layers[1:]):
                    for outputLayer in outputLayers:
                        # The inputs of the current layer are the outputs 
                        # of the next layer
                        inputs = [layerOutputs[inputNode] for inputNode in inputLayers]

                        # Create an array of weights from the input layer 
                        # to the output layers
                        weightArray = [weights[input + outputLayer] for input in inputLayers]

                        # Apply a bias if one is supplied for the else do not provide one
                        bias = 0 if not biases.__contains__(
                            outputLayer) else biases[outputLayer]

                        # The total net input
                        sumOutput = self.rounded(
                            sum([weight * x for weight, x in zip(weightArray, inputs)]) + bias * 1
                        )
                        layerInputs[outputLayer] = sumOutput

                        # Apply the sigmoid function
                        roundedActivatedValue = self.rounded(self.activate(sumOutput))

                        # The output of the node is the activated value
                        layerOutputs[outputLayer] = roundedActivatedValue

                        inputs.clear()
                        weightArray.clear()

                # Find the outputs from the output layer
                neuralNetworkOutputs = [layerOutputs[key] for key in layers[-1]]

                # The squared error is equal to the distance between the expected
                # outputs and the system outputs
                squaredError = self.rounded((
                    sum(
                        (target - system) ** 2 for target, system in zip(targetValues, neuralNetworkOutputs)
                    )
                ))
                sumOfSquaredErrors += squaredError

                # The delta of the output nodes is equals to the error * nodeInput
                deltas = {node: (target - layerOutputs[node]) * self.deactivate(
                    layerOutputs[node]) for target, node in zip(targetValues, layers[-1])}

                # Select the second last layer
                currentLayer = len(layers) - 2
                while (currentLayer >= 0):

                    # For each node in the current layer and each node in the subsequent layer
                    for (currentNode, nextNode) in itertools.product(layers[currentLayer], layers[currentLayer + 1]):
                        # the path name
                        # eg a + c = ac meaning the weight of the connection between a and c
                        path = currentNode + nextNode 

                        # d* = d * w * deactivated value
                        deltas[currentNode] = deltas[nextNode] * \
                                              weights[path] * self.deactivate(layerOutputs[currentNode])

                        # W1 = W0 + 𝚫W                                              
                        weights[path] += self.learningRate * deltas[nextNode] * layerOutputs[currentNode]


                    currentLayer -= 1

            # find the mean squared error
            self.currentErrorRate = sumOfSquaredErrors / len(ann_data)

            # Only stop if the current mean squared error is less 
            # than the tolerance value
            if self.currentErrorRate < self.toleranceValue:
                print("rounds ", rounds)
                break
            
            # Print out the error at the current round
            print(rounds, " - ", self.currentErrorRate)
            rounds += 1

    def calculateSum(self, inputVector, weights):
        """
        Find ∑
        :param inputVector: The inputs into the node
        :param weights: The weights to be applied on each input
        :return:
        """
        return sum(w * x for (w, x) in zip(weights, inputVector))

    def activate(self, value):
        """
        Apply ɸ - in our case sigmoid
        :param value: The value to be put into the sigmoid function
        :return:
        """
        return 1 / (1 + math.exp(-(value)))

    def deactivate(self, value):
        """
        Remove ɸ - in our case sigmoid
        :param value: The value to be deactivated from the sigmoid activation
        :return: The deactivated value
        """
        return (1 - value) * value

    def calculateError(self, y, targetValue):
        """
        Find ε
        :param y: The system output
        :param targetValue: The target value of the network
        :return:
        """
        return (y - targetValue) ** 2

    def rounded(self, value) -> float:
        """
        :param value: The value to be rounded to 5 decimal points
        :return: The rounded value
        """
        return round(value, 5)


# The test data
ann_data = [
    ([0.4, -0.7], [0.1]),
    ([0.3, -0.5], [0.05]),
    ([0.6, 0.1], [0.3]),
    ([0.2, 0.4], [0.25]),
    ([0.1, -0.2], [0.12]),
]

# Define the weights and layers according
# to the network diagram

# Each weight is a mapping for path to value
# For example,
# xy: 0.2 means that the path 
# from node x to y has a weight of 0.2
weights = {
    "ac": 0.1,
    "ad": 0.4,
    "bc": -0.2,
    "bd": 0.2,
    "ce": 0.2,
    "de": -0.5
}

# Each layer is represented as an array of nodes
# Here, a & b are in the input layer
# c & d are in the hidden layer
# and e is in the output layer
layers = [
    ["a", "b"],
    ["c", "d"],
    ["e"]
]
# node -> input bias value
biases = {
    "c": 1,
    "d": 1,
    "e": 1,
}
bpNetwork = Network(layers, weights, biases)
bpNetwork.learnOn(ann_data)

print()
print()
print("Final Error Rate", bpNetwork.currentErrorRate)
print()
print("Final Weights\n", bpNetwork.weights)
# Back Propagation
# By Nduati Kuria
# P15/44056/2017
