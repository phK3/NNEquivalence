from nn_loader import NNLoader

from json import loads
import h5py
import numpy as np


class KerasLoader(NNLoader):

    class Layer:

        def __init__(self, layer_type, name, units, activation, inputs, weights):
            self.layer_type = layer_type
            self.name = name
            self.units = units
            self.activation = activation
            self.inputs = inputs
            self.weights = weights

    def __init__(self):
        self.filename = None
        self.f = None
        self.layers = []


    def load(self, filename):
        self.f = h5py.File(filename, 'r')

        model_config = loads(self.f.attrs['model_config'].decode('utf-8'))

        def create_layer(layer_dict):
            layer_type = layer_dict['class_name']
            layer_config = layer_dict['config']
            name = layer_config['name']
            units = layer_config['units']
            activation = layer_config['activation']
            inputs = None
            weights = None

            if 'batch_input_shape' in layer_config:
                inputs = layer_config['batch_input_shape']

            return self.Layer(layer_type, name, units, activation, inputs, weights)

        for layer in model_config['config']['layers']:
            # input layer is handled in first real layer via batch_input_shape
            if not layer['class_name'] == 'InputLayer':
                self.layers.append(create_layer(layer))


        model_weights_dict = self.f['model_weights']
        for layer in self.layers:
            layer_weights_dict = model_weights_dict[layer.name]
            weight_names = layer_weights_dict.attrs['weight_names']
            # [()] converts the dataset to a numpy array
            # (as would have done ds.value, but this is deprecated
            w = layer_weights_dict[weight_names[0]][()]
            b = layer_weights_dict[weight_names[1]][()]
            layer.weights = np.vstack((w, b))

    def getNumLayers(self):
        return len(self.layers)

    def getNumInputs(self):
        first_hidden = self.layers[0]
        inputs_with_bias, _ = first_hidden.weights.shape
        return inputs_with_bias - 1

    def getNumOutputs(self):
        last_hidden = self.layers[-1]
        _, outputs = last_hidden.weights.shape
        return outputs

    def getActivationFunction(self, layer):
        return self.layers[layer].activation

    def getNumNeurons(self, layer):
        return self.layers[layer].units

    # weights to layer before, bias at last row of matrix
    def getWeights(self, layer):
        return self.layers[layer].weights

    def getHiddenLayers(self):
        layers = []
        for i in range(0, self.getNumLayers()):
            activation = self.getActivationFunction(i)
            numNeurons = self.getNumNeurons(i)
            weights = self.getWeights(i)
            layers.append((activation, numNeurons, weights))

        return layers

    def get_overview(self):
        print('Inputs: ' + str(self.getNumInputs()))

        for i, (activation, numNeurons, _ )in enumerate(self.getHiddenLayers()):
            print('Layer_{idx}: {neurons} -- {function}'.format(idx=i+1, neurons=numNeurons, function=activation))
