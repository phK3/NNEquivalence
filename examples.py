
from expression_encoding import encodeNN, encode_maxpool_layer, encode_inputs, \
    pretty_print, interval_arithmetic, encode_linear_layer, encode_relu_layer, encode_from_file
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


def exampleEncodeCancer():
    # not finished yet
    input_los = [-2.019404, -2.272988, -1.977589, -1.426379, -3.176344, -1.664312, -1.125696, -1.262871, -2.738225,
                 -1.865718e+00, -1.024522, -1.569514, -1.016081, -0.6933525, -1.862462, -1.304206, -1.012913, -1.977069,
                 -1.544220, -1.080050, -1.704360, -2.218398, -1.673608, -1.188201, -2.711807, -1.468356, -1.341360,
                 -1.754014, -2.128278, -1.598903]
    input_his = [3.963628, 3.528104, 3.980919, 5.163006, 3.503046, 4.125777, 4.366097, 3.955644, 4.496561, 5.105021,
                 8.697088, 6.788612, 9.410281, 10.52718, 5.747718, 6.308377, 11.73186, 6.984494, 4.999672, 10.02360,
                 4.049783, 3.938555, 4.261315, 5.758096, 3.988374, 5.270909, 4.936910, 2.695096, 5.934052, 6.968987e+00]
    input_benign_one_hi = input_his
    input_benign_one_hi[3] = 0.945520
    vars, constraints = encode_from_file('ExampleNNs/cancer_lin.h5', input_los, input_benign_one_hi)
    pretty_print(vars, constraints)