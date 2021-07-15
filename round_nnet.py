
import numpy as np
from NNet import NNet


def round_nnet_16bit(nnet: NNet, use_batch=False):
    weights = nnet.weights
    biases = nnet.biases
    new_weights = []
    new_biases = []
    for k, weightMatrix in enumerate(weights):
        new_weights.append(np.float16(weightMatrix))
        new_biases.append(np.float16(biases[k]))
    nnet.weights = new_weights
    nnet.biases = new_biases
    return nnet
