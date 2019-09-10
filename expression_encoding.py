
from expression import Variable, Linear, Relu, Max, Multiplication, Constant, Sum, Neg, One_hot, Greater_Zero, \
    Geq, BinMult, Gt_Int, Impl, IndicatorToggle
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

    for i in range(len(outs)):
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


def encode_partial_layer(top_k, prev_neurons, layerIndex, netPrefix):
    order_constrs = []

    n = len(prev_neurons)
    outs = [Variable(layerIndex, i, netPrefix, 'o') for i in range(top_k)]
    # !!! careful, because NN rows and columns in index are swapped
    # p_ij in matrix, but p_j_i in printed output
    # but for calculation permute matrix is stored as array of rows (as in math)
    partial_matrix = [[Variable(j, i, netPrefix, 'pi', type='Int') for j in range(n)] for i in range(top_k)]

    # perm_matrix * prev_neurons = outs
    res_vars, permute_constrs = encode_binmult_matrix(prev_neurons, layerIndex, netPrefix, partial_matrix, outs)

    # almost doubly stochastic
    one = Constant(1, netPrefix, layerIndex, 0)
    for i in range(top_k):
        # row stochastic
        permute_constrs.append(Linear(Sum(partial_matrix[i]), one))

    set_vars = []
    for j in range(len(prev_neurons)):
        # almost column stochastic (<= 1)
        s = Variable(layerIndex, j, netPrefix, 'set', type='Int')
        set_vars.append(s)
        permute_constrs.append(Linear(Sum([p[j] for p in partial_matrix]), s))
        permute_constrs.append(Geq(one, s))

    # o_i >= o_i+1 (for top_k)
    for o, o_next in zip(outs, outs[1:]):
        order_constrs.append(Geq(o, o_next))

    # x_i <= o_k-1 for all i, that are not inside top_k
    for i, s in enumerate(set_vars):
        order_constrs.append(Impl(s, 0, prev_neurons[i], outs[-1]))

    constraints = permute_constrs + order_constrs
    return [partial_matrix, set_vars], (res_vars + outs), constraints


def encode_sort_one_hot_layer(prev_neurons, layerIndex, netPrefix, mode):
    n = len(prev_neurons)
    one_hot_vec = [Variable(layerIndex, i, netPrefix, 'pi', type='Int') for i in range(n)]

    top = Variable(layerIndex, 0, netPrefix, 'top')
    # one_hot_vec and top need to be enclosed in [], so that indexing in binmult_matrix works
    res_vars, mat_constrs = encode_binmult_matrix(prev_neurons, 0, netPrefix, [one_hot_vec], [top])

    oh_constraint = Linear(Sum(one_hot_vec), Constant(1, netPrefix, layerIndex, 0))
    order_constrs = [Geq(top, neuron) for neuron in prev_neurons]

    outs = None
    vars = None
    if mode == 'vector':
        outs = one_hot_vec
        vars = res_vars + [top]
    elif mode == 'out':
        outs = [top]
        vars = res_vars + one_hot_vec
    else:
        raise ValueError('Unknown mode for encoding of sort_one_hot layer: {name}'.format(name=mode))

    return outs, vars, [oh_constraint] + mat_constrs + order_constrs


def hasLinear(activation):
    if activation == 'one_hot':
        return False
    elif activation == 'relu':
        return True
    elif activation == 'linear':
        return True


def encode_layers(input_vars, layers, net_prefix):
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

            if activation.startswith('partial_'):
                top_k = int(activation.split('_')[-1])
                vectors, rank_vars, rank_constraints = encode_partial_layer(top_k, invars, i, net_prefix)
                vars.append(rank_vars)
                # vectors is [partial_matrix (k rows, invers cols), set-vector (1 <-> s_i not amongst top k)]
                vars.append(vectors)
                constraints.append(rank_constraints)

                invars = vectors

            if activation.startswith('sort_one_hot_'):
                #modes 'vector' and 'out' are allowed and define what is returned as out
                mode = activation.split('_')[-1]
                oh_outs, oh_vars, oh_constraints = encode_sort_one_hot_layer(invars, i, net_prefix, mode)
                vars.append(oh_vars)
                vars.append(oh_outs)
                constraints.append(oh_constraints)

                invars = oh_outs

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
    elif mode.startswith('partial_'):
        _, num_outs, _ = layers[-1]
        partial_layer = (mode, num_outs, None)
        layers.append(partial_layer)
    elif mode.startswith('sort_one_hot_'):
        _, num_outs, _ = layers[-1]
        oh_layer = (mode, num_outs, None)
        layers.append(oh_layer)
    else:
        raise ValueError('Invalid mode for NN encoding: {name}'.format(name=mode))

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
        delta_lt = Variable(layer + 1, row, net, 'dl', 'Int')

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
    elif mode == 'optimize_diff':
        for i, (out1, out2) in enumerate(zip(outs1, outs2)):
            diff_i = Variable(0, i, 'E', 'diff')
            constraints.append(Linear(Sum([out1, Neg(out2)]), diff_i))

            diffs.append(diff_i)
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
    elif mode.startswith('one_ranking_top_'):
        # assumes outs1 = permutation matrix of NN1
        # outs2 = outputs of NN1

        k = int(mode.split('_')[-1])

        matrix = outs1
        ordered2 = [Variable(0, i, 'E', 'o') for i in range(len(outs2))]

        res_vars, mat_constrs = encode_binmult_matrix(outs2, 0, 'E', matrix, ordered2)

        order_constrs = []
        deltas = []
        for i in range(k, len(outs2)):
            delta_i = Variable(0, i, 'E', 'd', type='Int')
            deltas.append(delta_i)
            # o_1 < o_i <--> d = 1
            # 0 < o_i - o_1 <--> d = 1
            order_constrs.append(Greater_Zero(Sum([ordered2[i], Neg(ordered2[0])]), delta_i))

        order_constrs.append(Geq(Sum(deltas), Constant(1, 'E', 0, 0)))

        constraints = mat_constrs + order_constrs
        diffs = res_vars + ordered2
    elif mode.startswith('optimize_ranking_top_'):
        k = int(mode.split('_')[-1])

        matrix = outs1
        ordered2 = [Variable(0, i, 'E', 'o') for i in range(len(outs2))]

        res_vars, mat_constrs = encode_binmult_matrix(outs2, 0, 'E', matrix, ordered2)

        order_constrs = []
        diffs = []
        for i in range(k, len(outs2)):
            diff_i = Variable(0, i, 'E', 'diff')
            diffs.append(diff_i)
            order_constrs.append(Linear(Sum([ordered2[i], Neg(ordered2[0])]), diff_i))

        constraints = mat_constrs + order_constrs
        deltas = res_vars + ordered2
    elif mode.startswith('partial_top_'):
        # assumes outs1 = [partial matrix, set-var] of NN1
        # assumes outs2 = outputs of NN2
        partial_matrix = outs1[0]
        one_hot_vec = partial_matrix[0]
        set_var = outs1[1]

        top = Variable(0, 0, 'E', 'top')
        # one_hot_vec and top need to be enclosed in [], so that indexing in binmult_matrix works
        res_vars, mat_constrs = encode_binmult_matrix(outs2, 0, 'E', [one_hot_vec], [top])

        order_constrs = []
        for i in range(len(outs2)):
            order_constrs.append(Impl(set_var[i], 0, Sum([outs2[i], Neg(top)]), Constant(0, 'E', 0, 0)))

        constraints = mat_constrs + order_constrs
        deltas = res_vars
        diffs = [top]
    elif mode.startswith('optimize_partial_top_'):
        # assumes outs1 = [partial matrix, set-var] of NN1
        # assumes outs2 = outputs of NN2
        partial_matrix = outs1[0]
        one_hot_vec = partial_matrix[0]
        set_var = outs1[1]

        top = Variable(0, 0, 'E', 'top')
        # one_hot_vec and top need to be enclosed in [], so that indexing in binmult_matrix works
        res_vars, mat_constrs = encode_binmult_matrix(outs2, 0, 'E', [one_hot_vec], [top])

        order_constrs = []
        diffs = [Variable(0, i, 'E', 'diff') for i in range(len(outs2))]
        order_constrs.append(IndicatorToggle(set_var, 0, [Sum([outs2[i], Neg(top)]) for i in range(len(outs2))], diffs))

        max_diff_vec = [Variable(1, i, 'E', 'pi', 'Int') for i in range(len(diffs))]
        max_diff = Variable(1, 0, 'E', 'max_diff')
        res_vars2, mat_constrs2 = encode_binmult_matrix(diffs, 1, 'Emax', [max_diff_vec], [max_diff])
        for diff in diffs:
            order_constrs.append(Geq(max_diff, diff))

        diffs.append(max_diff)

        constraints = mat_constrs + order_constrs + mat_constrs2
        deltas = res_vars + [top] + max_diff_vec + res_vars2

    elif mode.startswith('one_hot_partial_top_'):
        k = int(mode.split('_')[-1])
        # assumes outs1 = one hot vector of NN1
        # assumes outs2 = output of NN2
        one_hot_vec = outs1

        top = Variable(0, 0, 'E', 'top')
        # one_hot_vec and top need to be enclosed in [], so that indexing in binmult_matrix works
        res_vars, mat_constrs = encode_binmult_matrix(outs2, 0, 'E', [one_hot_vec], [top])

        partial_matrix, partial_vars, partial_constrs = encode_partial_layer(k, outs2, 1, 'E')

        diff = Variable(0, k, 'E', 'diff')
        diff_constr = Linear(Sum([partial_vars[-1], Neg(top)]), diff)

        deltas = [top] + res_vars + partial_matrix + partial_vars
        diffs = [diff]
        constraints = mat_constrs + partial_constrs + [diff_constr]
    elif mode == 'one_hot_diff':
        # assumes outs1 = one hot vector of NN1
        # assumes outs2 = output of NN2
        one_hot_vec = outs1
        top = Variable(0, 0, 'E', 'top')
        # one_hot_vec and top need to be enclosed in [], so that indexing in binmult_matrix works
        res_vars, mat_constrs = encode_binmult_matrix(outs2, 0, 'E', [one_hot_vec], [top])

        diffs = [Variable(0, i, 'E', 'diff') for i in range(len(outs2))]
        diff_constrs = [Linear(Sum([out, Neg(top)]), diff) for out, diff in zip(outs2, diffs)]

        deltas = [top] + res_vars
        constraints = mat_constrs + diff_constrs
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
        one_ranking_top_k - calculates one permutation matrix on outputs of NN1 and checks, for sortedness between
                            top k outputs of NN2 and the rest of the outputs
        optimize_ranking_top_k - calculates one permutation matrix on outputs of NN1 and checks for sortedness between
                            top k outputs of NN2 and rest of outputs by computing the difference between o_1' and the non
                            o_k+1'... outputs, needs manual optimization function
        [not implemented yet] partial_top_k - calculates only necessary part of permutation matrix (only k rows) on NN1
                            and checks for sortedness between top k outputs of NN2 and rest of the outputs
        optimize_partial_top_k - calculates only necessary part of permutation matrix (only k rows) on NN1 and checks for
                            sortedness between top k outputs of NN2 and rest of the outputs, needs manual optimization
                            function
    :param comparator: keyword for how the selected elements should be compared.
        diff_zero    - elements should be equal
        epsilon_e   - elements of output vector of NN2 should not differ by more than epsilon from the
                        respective output element of NN1. Epsilon is equal to e (any positive number entered)
        diff_one_hot - one-hot vectors should be equal (only works for one-hot encoding)
        ranking_top_k - one-hot vector of NN1 should be within top k ranked outputs of NN2
        ranking      - ???
        one_ranking_top_k - calculates one permutation matrix on outputs of NN1 and checks, for sortedness between
                            top k outputs of NN2 and the rest of the outputs
        optimize_ranking_top_k - calculates one permutation matrix on outputs of NN1 and checks for sortedness between
                            top k outputs of NN2 and rest of outputs by computing the difference between o_1' and the non
                            o_k+1'... outputs, needs manual optimization function
        optimize_partial_top_k - calculates only necessary part of permutation matrix (only k rows) on NN1 and checks for
                            sortedness between top k outputs of NN2 and rest of the outputs, needs manual optimization
                            function
        optimize_diff - calculates difference for each element of output of NN1 and NN2, needs manual optimization
                        function
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
    elif compared.startswith('one_ranking_top_') or compared.startswith('optimize_ranking_top_'):
        _, num_outs1, _ = layers1[-1]
        _, num_outs2, _ = layers2[-1]

        if not num_outs1 == num_outs2:
            raise ValueError("both NNs must have the same number of outputs")

        # not sure what to specify as num_neurons (num of sorted outs or num of p_ij in permutation matrix?)
        ranking_layer = ('ranking', num_outs1, None)
        layers1.append(ranking_layer)
    elif compared.startswith('partial_top_') or compared.startswith('optimize_partial_top_'):
        _, num_outs1, _ = layers1[-1]
        _, num_outs2, _ = layers2[-1]

        if not num_outs1 == num_outs2:
            raise ValueError("both NNs must have the same number of outputs")

        k = int(compared.split('_')[-1])
        # not sure what to specify as num_neurons (num of sorted outs or num of p_ij in permutation matrix?)
        partial_layer = ('partial_{topk}'.format(topk=k), num_outs1, None)
        layers1.append(partial_layer)
    elif compared.startswith('one_hot_partial_top_') or compared == 'one_hot_diff':
        _, num_outs1, _ = layers1[-1]
        _, num_outs2, _ = layers2[-1]

        if not num_outs1 == num_outs2:
            raise ValueError("both NNs must have the same number of outputs")

        one_hot_layer = ('sort_one_hot_vector', num_outs1, None)
        layers1.append(one_hot_layer)
    else:
        raise ValueError('Invalid parameter \'compared\' for encode_equivalence: {name}'.format(name=compared))


    invars = encode_inputs(input_lower_bounds, input_upper_bounds)
    net1_vars, net1_constraints = encode_layers(invars, layers1, 'A')
    net2_vars, net2_constraints = encode_layers(invars, layers2, 'B')

    # should never be used
    outs1 = net1_vars[-1]
    outs2 = net2_vars[-1]
    if compared in {'outputs', 'one_hot', 'one_hot_diff'} or compared.startswith('one_hot_partial_top_'):
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
    elif compared.startswith('one_ranking_top_') or compared.startswith('optimize_ranking_top_') \
            or compared.startswith('optimize_partial_top_') or compared.startswith('partial_top_'):
        k = int(compared.split('_')[-1])

        matrix1 = net1_vars[-1]

        outs1 = matrix1
        outs2 = net2_vars[-1]
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


def encode_optimize_equivalence(path1, path2, input_lower_bounds, input_upper_bounds, target_output, interval=True,
                                compared='optimize_ranking_top_3', comparator='optimize_ranking_top_3'):
    vars, constraints = encode_equivalence_from_file(path1, path2, input_lower_bounds, input_upper_bounds, compared,
                                                 comparator)

    if interval:
        interval_arithmetic(constraints)

    # assumes unique var E_diff_0_i for differences
    model = create_gurobi_model(vars, constraints)
    diff = model.getVarByName('E_diff_0_{index}'.format(index=target_output))
    model.setObjective(diff, grb.GRB.MAXIMIZE)

    return model, vars, constraints
