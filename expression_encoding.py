
from expression import Variable, Linear, Relu, Max, Multiplication, Constant, Sum


def flatten(list):
    return [x for sublist in list for x in sublist]


def makeLeq(lhs, rhs):
    return '(assert (<= ' + lhs + ' ' + rhs + '))'


def makeGeq(lhs, rhs):
    # maybe switch to other representation later
    return makeLeq(rhs, lhs)


def makeEq(lhs, rhs):
    return '(assert (= ' + lhs + ' ' + rhs + '))'


def makeLt(lhs, rhs):
    return '(assert (< ' + lhs + ' ' + rhs + '))'


def makeGt(lhs, rhs):
    return makeLt(rhs, lhs)


def encode_inputs(lower_bounds, upper_bounds, netPrefix=''):
    vars = []
    for i, (l, h) in enumerate(zip(lower_bounds, upper_bounds)):
        input_var = Variable(0, i, netPrefix, 'i')
        input_var.setLo(l)
        input_var.setHi(h)
        vars.append(input_var)

    return vars


def encode_linear_layer(prev_neurons, weights, numNeurons, layerIndex, netPrefix):
    vars = []
    equations = []
    prev_num = len(prev_neurons)
    for i in range(0, numNeurons):
        var = Variable(layerIndex, i, netPrefix, 'x')
        vars.append(var)
        terms = [Multiplication(Constant(weights[row][i], netPrefix, layerIndex, row), prev_neurons[row]) for row in range(0, prev_num)]
        terms.append(Constant(weights[-1][i], netPrefix, layerIndex, prev_num))
        equations.append(Linear(Sum(terms), var))

    return vars, equations


def encode_relu_layer(prev_neurons, layerIndex, netPrefix):
    deltas = []
    outs = []
    ineqs = []
    for i, neuron in enumerate(prev_neurons):
        output = Variable(layerIndex, i, netPrefix, 'o')
        delta = Variable(layerIndex, i, netPrefix, 'd', 'Int')
        outs.append(output)
        deltas.append(delta)
        ineqs.append(Relu(neuron, output, delta))

    return outs, deltas, ineqs


def encode_maxpool_layer(prev_neurons, layerIndex, netPrefix):
    deltas = []
    outs = []
    ineqs = []

    current_neurons = prev_neurons
    depth = 0
    while len(current_neurons) >= 2:
        current_depth_outs = []
        for i in range(0, len(current_neurons), 2):
            if i + 1 >= len(current_neurons):
                out = current_neurons[i]
                current_depth_outs.append(out)
                # don't append to outs as already has constraint
            else:
                out = Variable(layerIndex, 0, netPrefix, 'o_' + str(depth))
                delta = Variable(layerIndex, 0, netPrefix, out.name + '_d', 'Int')
                ineq = Max(current_neurons[i], current_neurons[i + 1], out, delta)
                ineqs.append(ineq)
                deltas.append(delta)
                current_depth_outs.append(out)
                outs.append(out)

        current_neurons = current_depth_outs
        depth += 1

    return outs, deltas, ineqs


def encodeNN(layers, input_lower_bounds, input_upper_bounds, net_prefix):
    vars = []
    constraints = []

    invars = encode_inputs(input_lower_bounds, input_upper_bounds)
    vars.append(invars)

    for i, (activation, num_neurons, weights) in enumerate(layers):
        linvars, eqs = encode_linear_layer(invars, weights, num_neurons, i, net_prefix)
        vars.append(linvars)
        constraints.append(eqs)

        if activation == 'relu':
            reluouts, reludeltas, reluineqs = encode_relu_layer(linvars, num_neurons, net_prefix)

            vars.append(reluouts)
            vars.append(reludeltas)
            constraints.append(reluineqs)

            invars = reluouts
        elif activation == 'linear':
            invars = linvars

    return vars, constraints


def interval_arithmetic(constraints):
    for c in flatten(constraints):
        c.tighten_interval()


def pretty_print(vars, constraints):
    print('### Vars ###')
    for var in flatten(vars):
        print(str(var) + ': [' + str(var.getLo()) + ', ' + str(var.getHi()) + ']')

    print('### Constraints ###')
    for c in flatten(constraints):
        print(c)


def print_to_smtlib(vars, constraints):
    preamble = '(set-option :produce-models true)\n(set-logic AUFLIRA)'
    suffix = '(check-sat)\n(get-model)'
    decls = '; ### Variable declarations ###'
    bounds = '; ### Variable bounds ###'

    for var in flatten(vars):
        decls += '\n' + var.get_smtlib_decl()
        bound = var.get_smtlib_bounds()
        if not bound == '':
            bounds += '\n' + var.get_smtlib_bounds()

    consts = '; ### Constraints ###'

    for c in flatten(constraints):
        consts += '\n' + c.to_smtlib()

    return preamble + '\n' + decls + '\n' + bounds + '\n' + consts + '\n' + suffix
