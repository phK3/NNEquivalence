
from expression import Variable, Linear, Relu, Max, Multiplication, Constant, Sum, Neg, One_hot, Greater_Zero, Geq
from keras_loader import KerasLoader


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
    # last variable in outs is output of maxpool
    deltas = []
    outs = []
    ineqs = []

    if len(prev_neurons) == 1:
        # will create duplicate bounds for input_var, but needed,
        # s.t. other encodings can access output of this layer
        # through the outs list.
        return prev_neurons, deltas, ineqs

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


def encode_one_hot(prev_neurons, layerIndex, netPrefix):
    max_outs, deltas, ineqs = encode_maxpool_layer(prev_neurons, layerIndex, netPrefix)
    max_out = max_outs[-1]

    outs = []
    diffs = []
    eqs = []
    one_hot_constraints = []

    for i, x in enumerate(prev_neurons):
        output = Variable(layerIndex, i, netPrefix, 'o', 'Int')
        diff = Variable(layerIndex, i, netPrefix, 'x')
        outs.append(output)
        diffs.append(diff)

        eqs.append(Linear(Sum([x, Neg(max_out)]), diff))
        one_hot_constraints.append(One_hot(diff, output))

    constraints = ineqs + eqs + one_hot_constraints
    return outs, (deltas + diffs + max_outs), constraints


def encode_layers(input_vars, layers, net_prefix):

    def hasLinear(activation):
        if activation == 'one_hot':
            return False
        elif activation == 'relu':
            return True
        elif activation == 'linear':
            return True


    vars = []
    constraints = []

    invars = input_vars
    # output vars always appended last!
    for i, (activation, num_neurons, weights) in enumerate(layers):
        if hasLinear(activation):
            linvars, eqs = encode_linear_layer(invars, weights, num_neurons, i, net_prefix)
            vars.append(linvars)
            constraints.append(eqs)

            if activation == 'relu':
                reluouts, reludeltas, reluineqs = encode_relu_layer(linvars, num_neurons, net_prefix)

                vars.append(reludeltas)
                vars.append(reluouts)
                constraints.append(reluineqs)

                invars = reluouts
            elif activation == 'linear':
                invars = linvars
        else:
            # just use weights = None for one_hot layer
            if activation == 'one_hot':
                oh_outs, oh_vars, oh_constraints = encode_one_hot(invars, i, net_prefix)
                vars.append(oh_vars)
                vars.append(oh_outs)
                constraints.append(oh_constraints)

                invars = oh_outs

    return vars, constraints


def encodeNN(layers, input_lower_bounds, input_upper_bounds, net_prefix):
    invars = encode_inputs(input_lower_bounds, input_upper_bounds)

    layer_vars, layer_constrains = encode_layers(invars, layers, net_prefix)

    return [invars] + layer_vars, layer_constrains


def encode_equivalence_layer(outs1, outs2, mode='normal'):
    deltas = []
    diffs = []
    constraints = []

    for i, (out1, out2) in enumerate(zip(outs1, outs2)):
        delta_gt = Variable(0, i, 'E', 'dg', 'Int')
        delta_lt = Variable(0, i, 'E', 'dl', 'Int')
        diff = Variable(0, i, 'E', 'x')

        deltas.append(delta_gt)
        deltas.append(delta_lt)
        diffs.append(diff)

        constraints.append(Linear(Sum([out1, Neg(out2)]), diff))
        constraints.append(Greater_Zero(diff, delta_gt))
        constraints.append(Greater_Zero(Neg(diff), delta_lt))

    constraints.append(Geq(Sum(deltas), Constant(1, 'E', 1, 0)))

    return deltas, diffs, constraints


def encode_equivalence(layers1, layers2, input_lower_bounds, input_upper_bounds, with_one_hot = False):
    if with_one_hot:
        _, num_outs1, _ = layers1[-1]
        _, num_outs2, _ = layers2[-1]

        if not num_outs1 == num_outs2:
            raise ValueError("both NNs must have the same number of outputs")

        oh_layer = ('one_hot', num_outs1, None)
        layers1.append(oh_layer)
        layers2.append(oh_layer)

    invars = encode_inputs(input_lower_bounds, input_upper_bounds)
    net1_vars, net1_constraints = encode_layers(invars, layers1, 'A')
    net2_vars, net2_constraints = encode_layers(invars, layers2, 'B')

    eq_deltas, eq_diffs, eq_constraints = encode_equivalence_layer(net1_vars[-1], net2_vars[-1])

    vars = [invars] + net1_vars + net2_vars + [eq_diffs] + [eq_deltas]
    constraints = net1_constraints + net2_constraints + [eq_constraints]

    return vars, constraints


def encode_from_file(path, input_lower_bounds, input_upper_bounds):
    kl = KerasLoader()
    kl.load(path)

    layers = kl.getHiddenLayers()

    return encodeNN(layers, input_lower_bounds, input_upper_bounds, '')


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
