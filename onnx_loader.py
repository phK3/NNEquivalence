import numpy as np
import onnx
from onnx import numpy_helper

from nn_loader import NNLoader


class OnnxLoader(NNLoader):

    def __init__(self):
        super().__init__()
        self.layers = []
        self.filename = None

    def load(self, filename):
        self.filename = filename
        model = onnx.load(self.filename)
        onnx.checker.check_model(model)

        input_map = {i.name: i for i in model.graph.input}
        init_map = {i.name: i for i in model.graph.initializer}

        weight = None
        bias = None
        for node in model.graph.node:
            op = node.op_type

            if op in ['Add', 'Sub']:
                init = init_map[node.input[1]]
                bias = numpy_helper.to_array(init)
                if op == 'Sub':
                    bias = -bias
            elif op == 'Flatten':
                if len(self.layers) > 0:
                    raise ValueError('Flatten Layer is not yet implemented!')
                else:
                    # we are now at the input layer
                    s = model.graph.input[0].type.tensor_type.shape
                    input_shape = tuple(d.dim_value for d in s.dim)

                    cnt_gt_one = 0
                    for i in input_shape:
                        if i > 1:
                            cnt_gt_one += 1

                    if cnt_gt_one > 1:
                        raise ValueError('More than one dimension in the input shape is > 1, cannot safely ignore flatten layer!')
                    else:
                        print('[Parsing] Safely ignoring flattening layer.')
                        bias = bias.ravel()
                        print('[Parsing] Reshaping bias')
            elif op == 'MatMul':
                if node.input[1] in init_map:
                    init = init_map[node.input[1]]
                else:
                    init = init_map[node.input[0]]
                weight = numpy_helper.to_array(init)
            elif op == 'Relu':
                if weight.shape[1]==bias.shape[0]:
                    weights = np.vstack((weight, bias))
                else:
                    weights = np.vstack((weight.T, bias))
                numNeurons = len(bias)
                print(numNeurons)
                self.layers.append(('relu', numNeurons, weights))
            else:
                raise ValueError('Operation {} is not supported!'.format(op))

        if weight is not None and bias is not None:
            if weight.shape[1]==bias.shape[0]:
                weights = np.vstack((weight, bias))
            else:
                weights = np.vstack((weight.T, bias))
            numNeurons = len(bias)
            self.layers.append(('linear', numNeurons, weights))

    def print_layers(self):
        for i, l in enumerate(self.layers):
            activation, numNeurons, weights = l
            print('{}: {}, numNeurons={}'.format(i, activation, numNeurons))

    def getHiddenLayers(self):
        # return copy of self.layers
        return self.layers[:]

    def getNumLayers(self):
        return len(self.layers)

    def getNumInputs(self):
        first_hidden = self.layers[0]
        _, _, weights = first_hidden
        inputs_with_bias, _ = weights.shape
        return inputs_with_bias - 1

    def getNumOutputs(self):
        last_hidden = self.layers[-1]
        _, _, weights = last_hidden
        _, outputs = weights.shape
        return outputs

    def getActivationFunction(self, layer):
        activation, _, _ = self.layers[layer]
        return activation

    def getNumNeurons(self, layer):
        _, numNeurons, _ = self.layers[layer]
        return numNeurons

    def getWeights(self, layer):
        _, _, weights = self.layers[layer]
        return weights