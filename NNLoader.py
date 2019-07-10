from abc import ABC, abstractmethod


class NNLoader(ABC):

    def __init__(self, file):
        pass

    @abstractmethod
    def getNumLayers(self):
        pass

    @abstractmethod
    def getNumInputs(self):
        pass

    @abstractmethod
    def getNumOutputs(self):
        pass

    @abstractmethod
    def getActivationFunction(self, layer):
        pass

    @abstractmethod
    def getNumNeurons(self, layer):
        pass

    # weights to layer before, bias at last row of matrix
    @abstractmethod
    def getWeights(self, layer):
        pass
