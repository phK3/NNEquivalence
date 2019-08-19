
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


def encode_binmult_matrix(prev_neurons, layerIndex, netPrefix, matrix, outs):
    res_vars = []

    lin_constrs = []
    permute_constrs = []

    for i in range(len(prev_neurons)):
        res_vars_i = []
        for j, neuron in enumerate(prev_neurons):
            y = Variable(j, i, netPrefix, 'y')
            res_vars_i.append(y)

            # TODO: check indexes in BinMult for printing
            lin_constrs.append(BinMult(matrix[i][j], neuron, y))

        permute_constrs.append(Linear(Sum(res_vars_i), outs[i]))

        res_vars.append(res_vars_i)

    # lin_constrs before permute_constrs, s.t. interval arithmetic can tighten intervals
    # as we have no dependency graph, order of constraints is important
    return res_vars, (lin_constrs + permute_constrs)


def encode_ranking_layer(prev_neurons, layerIndex, netPrefix):
    order_constrs = []

    n = len(prev_neurons)
    outs = [Variable(layerIndex, i, netPrefix, 'o') for i in range(n)]
    # !!! careful, because NN rows and columns in index are swapped
    # p_ij in matrix, but p_j_i in printed output
    # but for calculation permute matrix is stored as array of rows (as in math)
    permute_matrix = [[Variable(j, i, netPrefix, 'pi', type='Int') for j in range(n)] for i in range(n)]

    # perm_matrix * prev_neurons = outs
    res_vars, permute_constrs = encode_binmult_matrix(prev_neurons, layerIndex, netPrefix, permute_matrix, outs)

    # o_i >= o_i+1
    for o, o_next in zip(outs, outs[1:]):
        order_constrs.append(Geq(o, o_next))

    # doubly stochastic
    one = Constant(1, netPrefix, layerIndex, 0)
    for i in range(len(prev_neurons)):
        # row stochastic
        permute_constrs.append(Linear(Sum(permute_matrix[i]), one))

    for j in range(len(prev_neurons)):
        # column stochastic
        permute_constrs.append(Linear(Sum([p[j] for p in permute_matrix]), one))

    constraints = permute_constrs + order_constrs
    return permute_matrix, (res_vars + outs), constraints


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

    def one_hot_comparison(oh1, oh2, net, layer, row, desired='different'):
        '''
        Compares two one-hot vectors and returns constraints that can only be satisfied,
        if the vectors are equal/different
        :param oh1: one-hot vector
        :param oh2: one-hot vector
        :param net: netPrefix
        :param layer: layer of the net, in which this operation takes place
        :param row: row of the net, in which this operation takes place
        :param desired: keyword
            different - the constraints can only be satisfied, if the vectors are different
            equal - the constraints can only be satisfied, if the vectors are equal
        :return: a tuple of (deltas, diffs, constraints) where constraints are as described above and deltas, diffs
            are variables used in these constraints
        '''
        # requires that oh_i are one-hot vectors
        oh_deltas = []
        oh_diffs = []
        oh_constraints = []

        desired_result = 1
        if desired == 'different':
            desired_result = 1
        elif desired == 'equal':
            desired_result = 0

        terms = []
        x = 1
        for i, (oh1, oh2) in enumerate(zip(oh1, oh2)):
            constant = Constant(x, net, layer, row)
            terms.append(Multiplication(constant, oh1))
            terms.append(Neg(Multiplication(constant, oh2)))
            x *= 2

        sumvar = Variable(layer, row, net, 's', 'Int')
        oh_constraints.append(Linear(Sum(terms), sumvar))

        delta_gt = Variable(layer, row, net, 'dg', 'Int')
        delta_lt = Variable(layer, row, net, 'dl', 'Int')
        zero = Constant(0, net, layer, row)

        oh_constraints.append(Gt_Int(sumvar, zero, delta_gt))
        oh_constraints.append(Gt_Int(zero, sumvar, delta_lt))
        oh_constraints.append(Geq(Sum([delta_lt, delta_gt]), Constant(desired_result, net, layer, row)))

        oh_deltas.append(delta_gt)
        oh_deltas.append(delta_lt)

        oh_diffs.append(sumvar)

        return oh_deltas, oh_diffs, oh_constraints

    def number_comparison(n1, n2, net, layer, row, epsilon=0):
        '''
        Compares two arbitrary numbers and returns constraints, s.t. one of the deltas is equal to 1, if the numbers
        are not equal
        :param n1: number
        :param n2: number
        :param net: netPrefix
        :param layer: layer of the net, in which this operation takes place
        :param row: row of the net, in which this operation takes place
        :return: a tuple of (deltas, diffs, constraints) where constraints are as described above and deltas, diffs
            are variables used in these constraints
        '''
        v_deltas = []
        v_diffs = []
        v_constraints = []

        delta_gt = Variable(layer, row, net, 'dg', 'Int')
        delta_lt = Variable(layer, row, net, 'dl', 'Int')

        if epsilon > 0:
            eps = Constant(epsilon, net, layer + 1, row)
            diff_minus_eps = Variable(layer, row, net, 'x_m')
            diff_plus_eps = Variable(layer, row, net, 'x_p')

            v_constraints.append(Linear(Sum([n2, Neg(n1), Neg(eps)]), diff_minus_eps))
            v_constraints.append(Linear(Sum([n2, Neg(n1), eps]), diff_plus_eps))

            v_constraints.append(Greater_Zero(diff_minus_eps, delta_gt))
            v_constraints.append(Greater_Zero(Neg(diff_plus_eps), delta_lt))

            v_diffs.append(diff_minus_eps)
            v_diffs.append(diff_plus_eps)
        else:
            diff = Variable(layer, row, net, 'x')

            v_constraints.append(Linear(Sum([n1, Neg(n2)]), diff))
            v_constraints.append(Greater_Zero(diff, delta_gt))
            v_constraints.append(Greater_Zero(Neg(diff), delta_lt))

            v_diffs.append(diff)

        v_deltas.append(delta_gt)
        v_deltas.append(delta_lt)


        #v_constraints.append(Geq(Sum(v_deltas), Constant(desired_result, net, layer + 1, row)))

        return v_deltas, v_diffs, v_constraints


    deltas = []
    diffs = []
    constraints = []

    if mode == 'diff_zero' or mode.startswith('epsilon_'):
        eps = 0
        if mode.startswith('epsilon_'):
            eps = float(mode.split('_')[-1])

        for i, (out1, out2) in enumerate(zip(outs1, outs2)):
            n_deltas, n_diffs, n_constraints = number_comparison(out1, out2, 'E', 0, i, epsilon=eps)

            deltas += n_deltas
            diffs += n_diffs
            constraints += n_constraints

        constraints.append(Geq(Sum(deltas), Constant(1, 'E', 1, 0)))
    elif mode == 'diff_one_hot':
        # requires that outs_i are the pi_1_js in of the respective permutation matrices
        # or input to this layer are one-hot vectors

        deltas, diffs, constraints = one_hot_comparison(outs1, outs2, 'E', 0, 0, desired='different')
    elif mode.startswith('ranking_top_'):
        # assumes outs1 = one-hot vector with maximum output of NN1
        # outs2 = (one-hot biggest, one-hot 2nd biggest, ...) of NN2

        k = int(mode.split('_')[-1])

        for i in range(k):
            k_deltas, k_diffs, k_constraints = one_hot_comparison(outs1, outs2[i], 'E', 0, i, desired='different')
            deltas += k_deltas
            diffs += k_diffs
            constraints += k_constraints
    else:
        raise ValueError('There is no \'' + mode + '\' keyword for parameter mode')

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
        ranking_top_k - checks, whether greatest output of NN1 is within top k outputs of NN2 (k is a natural number)
        ranking - (not supported yet) compares ranking vectors of NN1 and NN2 generated from their output,
        ranking_one_hot - compares one-hot vectors of NN1 and NN2 generated from a permutation matrix
    :param comparator: keyword for how the selected elements should be compared.
        diff_zero    - elements should be equal
        epsilon_e   - elements of output vector of NN2 should not differ by more than epsilon from the
                        respective output element of NN1. Epsilon is equal to e (any positive number entered)
        diff_one_hot - one-hot vectors should be equal (only works for one-hot encoding)
        ranking_top_k - one-hot vector of NN1 should be within top k ranked outputs of NN2
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
    elif compared in {'ranking', 'ranking_one_hot'} or compared.startswith('ranking_top_'):
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

    # should never be used
    outs1 = net1_vars[-1]
    outs2 = net2_vars[-1]
    if compared in {'outputs', 'one_hot'}:
        outs1 = net1_vars[-1]
        outs2 = net2_vars[-1]
    elif compared == 'ranking_one_hot':
        matrix1 = net1_vars[-1]
        matrix2 = net2_vars[-1]
        outs1 = matrix1[0]
        outs2 = matrix2[0]
    elif compared.startswith('ranking_top_'):
        k = int(compared.split('_')[-1])

        matrix1 = net1_vars[-1]
        matrix2 = net2_vars[-1]
        outs1 = matrix1[0]
        outs2 = matrix2[0:k]
    else:
        # default case
        raise ValueError('There is no ' + compared + ' keyword for param compared!!!')

    eq_deltas, eq_diffs, eq_constraints = encode_equivalence_layer(outs1, outs2, comparator)

    vars = [invars] + net1_vars + net2_vars + [eq_diffs] + [eq_deltas]
    constraints = net1_constraints + net2_constraints + [eq_constraints]

    return vars, constraints


def encode_from_file(path, input_lower_bounds, input_upper_bounds, mode='normal'):
    kl = KerasLoader()
    kl.load(path)

    layers = kl.getHiddenLayers()

    return encodeNN(layers, input_lower_bounds, input_upper_bounds, '', mode)


def encode_equivalence_from_file(path1, path2, input_lower_bounds, input_upper_bounds, compared='outputs',
                       comparator='diff_zero'):
    kl1 = KerasLoader()
    kl1.load(path1)
    layers1 = kl1.getHiddenLayers()

    kl2 = KerasLoader()
    kl2.load(path2)
    layers2 = kl2.getHiddenLayers()

    return encode_equivalence(layers1, layers2, input_lower_bounds, input_upper_bounds, compared, comparator)




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

