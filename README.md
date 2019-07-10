# NNEquivalence
Tool to encode NNs as MILP and check them for equivalence

To encode equivalence of two NNs import the module, create an instance of NNEncoder and call the encodeEquivalence method.

~~~~
from nn_encoding import *
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
from nn_encoding import *
enc = NNEncoder(None)
inputs = [1,2]
weights = [[1,4],[2,5],[3,6]]
layers = [('relu', 2, weights)]
print(enc.encodeNN(layers, inputs, inputs, True))
~~~~

Here inputs where used as both lower and upper bound on the inputs.

## Loading a saved NN

Load the linear cancer classifier.
~~~~
from keras_loader import KerasLoader
kl = KerasLoader()
kl.load('ExampleNNs/cancer_lin.h5')
kl.getNumInputs()
~~~~
Then encode the classifier using the NNEncoder. The example input should evaluate to 79.95576.
*Right now the encoding doesn't work, as scientific notation (eg. 1.76e-05 can't be handled by Z3). If this error is manually removed, online Z3 timeouts after ca. 1min*

~~~~
from nn_encoding import NNEncoder
enc = NNEncoder(None)
inputs = [-0.20175604,0.3290786,-0.13086754,-0.27145506,1.02919769,
          0.86411836,0.73363898,0.85669688,1.12032775,1.5535848,
          -0.04197566,-0.51588206,0.13154087,-0.13875636,-0.55953973,
          -0.13797354,0.0980708,0.28751196,-0.42446141,0.11305149,
          0.03150414,0.67628886,0.18528621,-0.06280808,1.10353068,0.87444267,
          1.2190909,1.3893291,1.08203284,1.54029664]
inputs[0] + inputs[2]
-0.33262358000000003
layers = []
for i in range(0, kl.getNumLayers()):
    activation = kl.getActivationFunction(i)
    numNeurons = kl.getNumNeurons(i)
    weights = kl.getWeights(i)
    layers.append((activation, numNeurons, weights))
    
s = enc.encodeNN(layers, inputs, inputs, False)
print(s)
~~~~
