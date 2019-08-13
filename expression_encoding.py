
from expression import Variable, Linear, Relu, Max, Multiplication, Constant, Sum, Neg, One_hot, Greater_Zero, \
    Geq, BinMult, Gt_Int
from keras_loader import KerasLoader
import gurobipy as grb
import datetime

# set this flag to only print bounds for deltas and inputs
# (for smtlib format)
hide_non_deltas = True


def flatten(collection):
    for x in collection:
        if isinstance(x, list):
            for y in flatten(x):
                yield y
        else:
            yield x


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


def encode_ranking_layer(prev_neurons, layerIndex, netPrefix):
    permute_vars = []
    res_vars = []
    outs = []

    order_constrs = []
    lin_constrs = []
    permute_constrs = []

    for i in range(len(prev_neurons)):
        output = Variable(layerIndex, i, netPrefix, 'o')
        outs.append(output)

        res_vars_i = []
        permute_vars_i = []
        for j, neuron in enumerate(prev_neurons):
            y = Variable(j, i, netPrefix, 'y')
            # !!! careful, because NN rows and columns in index are swapped
            # p_ij in matrix, but p_j_i in printed output
            # but for calculation permute matrix is stored as array of rows (as in math)
            pij = Variable(j, i, netPrefix, 'pi', type='Int')
            res_vars_i.append(y)
            permute_vars_i.append(pij)

            # TODO: check indexes in BinMult for printing
            lin_constrs.append(BinMult(pij, neuron, y))

        permute_constrs.append(Linear(Sum(res_vars_i), output))

        res_vars.append(res_vars_i)
        permute_vars.append(permute_vars_i)

    # o_i >= o_i+1
    for o, o_next in zip(outs, outs[1:]):
        order_constrs.append(Geq(o, o_next))

    # doubly stochastic
    one = Constant(1, netPrefix, layerIndex, 0)
    for i in range(len(prev_neurons)):
        # row stochastic
        permute_constrs.append(Linear(Sum(permute_vars[i]), one))

    for j in range(len(prev_neurons)):
        # column stochastic
        permute_constrs.append(Linear(Sum([p[j] for p in permute_vars]), one))

    # lin_constrs before permute_constrs, s.t. interval arithmetic can tighten intervals
    # as we have no dependency graph, order of constraints is important
    constraints = lin_constrs + permute_constrs + order_constrs
    return permute_vars, (res_vars + outs), constraints


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
                reluouts, reludeltas, reluineqs = encode_relu_layer(linvars, i, net_prefix)

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

            if activation == 'ranking':
                rank_perms, rank_vars, rank_constraints = encode_ranking_layer(invars, i, net_prefix)
                vars.append(rank_vars)
                # rank_perms is permutation matrix !!!
                vars.append(rank_perms)
                constraints.append(rank_constraints)

                invars = rank_perms

    return vars, constraints


def encodeNN(layers, input_lower_bounds, input_upper_bounds, net_prefix, mode='normal'):
    if mode == 'one_hot':
        _, num_outs, _ = layers[-1]

        oh_layer = ('one_hot', num_outs, None)
        layers.append(oh_layer)
    elif mode == 'ranking':
        _, num_outs, _ = layers[-1]
        ranking_layer = ('ranking', num_outs, None)
        layers.append(ranking_layer)

    invars = encode_inputs(input_lower_bounds, input_upper_bounds)

    layer_vars, layer_constrains = encode_layers(invars, layers, net_prefix)

    return [invars] + layer_vars, layer_constrains


def encode_equivalence_layer(outs1, outs2, mode='diff_zero'):
    deltas = []
    diffs = []
    constraints = []

    if mode == 'diff_zero':
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
    elif mode == 'diff_one_hot':
        # requires that outs_i are the pi_1_js in of the respective permutation matrices
        # or input to this layer are one-hot vectors
        terms = []
        x = 1
        for i, (out1, out2) in enumerate(zip(outs1, outs2)):
            const = Constant(x, 'E', 0, i)
            terms.append(Multiplication(const, out1))
            terms.append(Neg(Multiplication(const, out2)))
            x *= 2

        sumvar = Variable(1, 0, 'E', 's', 'Int')
        constraints.append(Linear(Sum(terms), sumvar))

        delta_gt = Variable(0, 0, 'E', 'dg', 'Int')
        delta_lt = Variable(0, 0, 'E', 'dl', 'Int')
        zero = Constant(0, delta_lt.net, 0, 0)

        constraints.append(Gt_Int(sumvar, zero, delta_gt))
        constraints.append(Gt_Int(zero, sumvar, delta_lt))
        constraints.append(Geq(Sum([delta_lt, delta_gt]), Constant(1, 'E', 1, 0)))

        deltas.append(delta_gt)
        deltas.append(delta_lt)

        diffs.append(sumvar)


    return deltas, diffs, constraints


def encode_equivalence(layers1, layers2, input_lower_bounds, input_upper_bounds, compared='outputs',
                       comparator='diff_zero'):
    '''
    :param layers1: first neural network as a list of layers of form (activation, num_neurons, weights)
    :param layers2: second neural network as a list of layers of form (activation, num_neurons, weights)
    :param input_lower_bounds: list of lower bounds for the input values
    :param input_upper_bounds: list of upper bounds for the input values
    :param compared: keyword for which element of the NNs should be compared.
        outputs - compares the outputs of NN1 and NN2 directly,
        one_hot - compares one-hot vectors of NN1 and NN2 generated from their output,
        ranking - compares ranking vectors of NN1 and NN2 generated from their output,
        ranking_one_hot - compares one-hot vectors of NN1 and NN2 generated from a permutation matrix
    :param comparator: keyword for how the selected elements should be compared.
        diff_zero    - elements should be equal
        diff_one_hot - one-hot vectors should be equal (only works for one-hot encoding)
        ranking      - ???
    :return: encoding of the equivalence of NN1 and NN2 as a set of variables and
        mixed integer linear programming constraints
    '''

    if compared == 'one_hot':
        _, num_outs1, _ = layers1[-1]
        _, num_outs2, _ = layers2[-1]

        if not num_outs1 == num_outs2:
            raise ValueError("both NNs must have the same number of outputs")

        oh_layer = ('one_hot', num_outs1, None)
        layers1.append(oh_layer)
        layers2.append(oh_layer)
    elif compared in {'ranking', 'ranking_one_hot'}:
        _, num_outs1, _ = layers1[-1]
        _, num_outs2, _ = layers2[-1]

        if not num_outs1 == num_outs2:
            raise ValueError("both NNs must have the same number of outputs")

        # not sure what to specify as num_neurons (num of sorted outs or num of p_ij in permutation matrix?)
        ranking_layer = ('ranking', num_outs1, None)
        layers1.append(ranking_layer)
        layers2.append(ranking_layer)

    invars = encode_inputs(input_lower_bounds, input_upper_bounds)
    net1_vars, net1_constraints = encode_layers(invars, layers1, 'A')
    net2_vars, net2_constraints = encode_layers(invars, layers2, 'B')

    if compared in {'outputs', 'one_hot'}:
        outs1 = net1_vars[-1]
        outs2 = net2_vars[-1]
    elif compared == 'ranking_one_hot':
        matrix1 = net1_vars[-1]
        matrix2 = net2_vars[-1]
        outs1 = matrix1[0]
        outs2 = matrix2[0]
    else:
        # default case
        outs1 = net1_vars[-1]
        outs2 = net2_vars[-1]

    eq_deltas, eq_diffs, eq_constraints = encode_equivalence_layer(outs1, outs2, comparator)

    vars = [invars] + net1_vars + net2_vars + [eq_diffs] + [eq_deltas]
    constraints = net1_constraints + net2_constraints + [eq_constraints]

    return vars, constraints


def encode_from_file(path, input_lower_bounds, input_upper_bounds, mode='normal'):
    kl = KerasLoader()
    kl.load(path)

    layers = kl.getHiddenLayers()

    return encodeNN(layers, input_lower_bounds, input_upper_bounds, '', mode)


def encode_equivalence_from_file(path1, path2, input_lower_bounds, input_upper_bounds, mode='normal'):
    kl1 = KerasLoader()
    kl1.load(path1)
    layers1 = kl1.getHiddenLayers()

    kl2 = KerasLoader()
    kl2.load(path2)
    layers2 = kl2.getHiddenLayers()

    return encode_equivalence(layers1, layers2, input_lower_bounds, input_upper_bounds, mode)




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

    def is_input_or_delta(var_name):
        # distinguish deltas, inputs and other intermediate vars
        # relies on convention, that only deltas contain d
        # and only inputs contain i
        return 'd' in var_name or 'i' in var_name

    for var in flatten(vars):
        decls += '\n' + var.get_smtlib_decl()
        bound = var.get_smtlib_bounds()
        if not bound == '':
            if hide_non_deltas:
                # TODO: find better way to exclude non-delta and input bounds
                # independent of string representation
                if is_input_or_delta(var.to_smtlib()):
                    bounds += '\n' + var.get_smtlib_bounds()
            else:
                bounds += '\n' + var.get_smtlib_bounds()

    consts = '; ### Constraints ###'

    for c in flatten(constraints):
        consts += '\n' + c.to_smtlib()

    return preamble + '\n' + decls + '\n' + bounds + '\n' + consts + '\n' + suffix


def create_gurobi_model(vars, constraints, name='NN_model'):
    if name == 'NN_model':
        date = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
        name += '_' + date

    model = grb.Model(name)

    for var in flatten(vars):
        var.register_to_gurobi(model)

    model.update()

    # model.setObjective(0, grb.GRB.MAXIMIZE)

    for c in flatten(constraints):
        c.to_gurobi(model)

    model.update()

    return model

