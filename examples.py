
from expression_encoding import encodeNN, encode_maxpool_layer, encode_inputs, \
    pretty_print, interval_arithmetic, encode_linear_layer, encode_relu_layer
from keras_loader import KerasLoader



def encodeExampleFixed():
    input_lower_bounds = [0 ,1, 2]
    input_upper_bounds = [1, 2, 3]
    weights = [[1, 5], [2, 6], [3, 7], [4, 8]]
    layers = [('relu', 2, weights)]

    vars, constraints = encodeNN(layers, input_lower_bounds, input_upper_bounds, '')

    return vars, constraints


def encodeMaxpoolExample():
    invars = encode_inputs([0,1,2], [1,2,3])
    outs, deltas, ineqs = encode_maxpool_layer(invars, 1, '')

    vars = [invars, deltas, outs]
    constraints = [ineqs]

    pretty_print(vars, constraints)

    print('### now with interval arithmetic ###')
    interval_arithmetic(constraints)
    pretty_print(vars, constraints)


def encodeExample():
    invars = encode_inputs([0,1,2], [1,2,3])

    weights = [[1,5],[2,6],[3,7],[4,8]]
    linvars, eqs = encode_linear_layer(invars, weights, 2, 1, '')

    for eq in eqs:
        eq.tighten_interval()

    reluouts, reludeltas, ineqs = encode_relu_layer(linvars, 2, '')

    for relu in ineqs:
        relu.tighten_interval()


    print('### invars ###')
    for i in invars:
        print(str(i) + ': [' + str(i.getLo()) + ', ' + str(i.getHi()) + ']')

    print('### linears ###')
    for var, eq in zip(linvars, eqs):
        print(eq)
        print(str(var) + ': [' + str(var.getLo()) + ', ' + str(var.getHi()) + ']')

    print('### relus ###')
    for ineq in ineqs:
        print(ineq)
    for out, delta in zip(reluouts, reludeltas):
        print(str(out) + ': [' + str(out.getLo()) + ', ' + str(out.getHi()) + ']')
        print(str(delta) + ': [' + str(delta.getLo()) + ', ' + str(delta.getHi()) + ']')

def exampleEncodeSimpleCancer():
    # encode simple cancer classifier
    # result for given input should be 19.67078
    kl = KerasLoader()
    kl.load('ExampleNNs/cancer_simple_lin.h5')

    inputs = [8, 10, 10, 8, 6, 9, 3, 10, 10]
    layers = kl.getHiddenLayers()

    vars, constraints = encodeNN(layers, inputs, inputs, '')

    interval_arithmetic(constraints)

    pretty_print(vars, constraints)