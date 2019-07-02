# NNEquivalence
Tool to encode NNs as MILP and check them for equivalence

To encode equivalence of two NNs import the module, create an instance of NNEncoder and call the encodeEquivalence method.

~~~~
from NNEncoding import *
enc = NNEncoder(None)
inputs = [1,2]
weights = [[1,4],[2,5],[3,6]]
layers = [('relu', 2, weights)]
weights2 = [[1,5],[2,5],[3,6]]
layers2 = [('relu', 2, weights2)]
print(enc.encodeEquivalence(layers, layers2, inputs, inputs, False))
~~~~

The networks above only have one layer. 
Layers is a list of layers, each of which is a 3-tuple of activation function, number of it's neurons and a weights matrix.
encodeEquivalence(...) takes the layers of the first NN, the layers of the second NN, the lower and upper bounds on the inputs
and a boolean flag, that represents, wether oneHot-encoding is used.

Encoding just an NN without the equivalence is similar:

~~~~
from NNEncoding import *
enc = NNEncoder(None)
inputs = [1,2]
weights = [[1,4],[2,5],[3,6]]
layers = [('relu', 2, weights)]
print(enc.encodeNN(layers, inputs, inputs, True))
~~~~

Here inputs where used as both lower and upper bound on the inputs.
