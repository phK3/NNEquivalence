
from abc import ABC, abstractmethod
from expression import Expression, Variable, Linear, Sum, Neg
from keras_loader import KerasLoader
from expression_encoding import encode_equivalence, interval_arithmetic, hasLinear, encode_linear_layer, \
    encode_relu_layer, encode_one_hot, encode_ranking_layer, encode_equivalence_layer, create_gurobi_model, pretty_print, \
    encode_partial_layer, encode_sort_one_hot_layer
import gurobipy as grb


class Layer(ABC):

    def __init__(self, activation, num_neurons, invars, intervars, outvars, constraints):
        self.activation = activation
        self.num_neurons = num_neurons
        self.invars = invars
        self.outvars = outvars
        self.intervars = intervars
        self.constraints = constraints

    def get_invars(self):
        return self.invars

    def get_invar(self, row):
        return self.invars[row]

    def get_intervars(self):
        return self.intervars

    def get_outvars(self):
        return self.outvars

    def get_outvar(self, row):
        return self.outvars[row]

    def get_all_vars(self):
        return self.intervars + self.outvars

    def get_constraints(self):
        return self.constraints

    @abstractmethod
    def get_optimization_vars(self):
        pass

    @abstractmethod
    def get_optimization_constraints(self):
        pass


class DefaultLayer(Layer):

    def __init__(self, activation, num_neurons, invars, intervars, outvars, constraints):
        super(DefaultLayer, self).__init__(activation, num_neurons, invars, intervars, outvars, constraints)

    def get_optimization_vars(self):
        return self.outvars

    def get_optimization_constraints(self):
        return self.constraints


class InputLayer(Layer):

    def __init__(self, num_neurons, invars):
        super(InputLayer, self).__init__('input', num_neurons, invars, [], invars, [])
        self.invars = invars

    def get_optimization_vars(self):
        return []

    def get_optimization_constraints(self):
        return []

    def get_all_vars(self):
        return self.invars


class ReLULayer(Layer):

    def __init__(self, lin_layer, num_neurons, invars, intervars, outvars, constraints):
        super(ReLULayer, self).__init__('relu', num_neurons, invars, intervars, outvars, constraints)
        self.lin_layer = lin_layer
        self.intervars = self.lin_layer.intervars + self.lin_layer.outvars + intervars

    def get_constraints(self):
        return self.lin_layer.get_constraints() + self.constraints

    def get_optimization_vars(self):
        return self.lin_layer.get_optimization_vars()

    def get_optimization_constraints(self):
        return self.lin_layer.get_optimization_constraints()


class Encoder:

    def __init__(self):
        self.a_layers = []
        self.b_layers = []
        self.input_layer = None
        self.equivalence_layer = None

        self.opt_timeout = 20

    def set_opt_timeout(self, new_val):
        self.opt_timeout = new_val

    def encode_inputs(self, lower_bounds, upper_bounds, netPrefix=''):
        vars = []
        for i, (l, h) in enumerate(zip(lower_bounds, upper_bounds)):
            input_var = Variable(0, i, netPrefix, 'i')
            input_var.setLo(l)
            input_var.setHi(h)
            vars.append(input_var)

        num_neurons = len(lower_bounds)
        return InputLayer(num_neurons, vars)

    def encode_layers(self, input_vars, layers, net_prefix, output_mode=('matrix', -1)):
        # outputs modes to specify what is output of ranking layer:
        # ('matrix', -1) whole matrix is output
        # ('matrix', k) matrix[0:k] is output
        # ('out', -1) whole sorted vector is output

        vars = []
        constraints = []
        net_layers = []

        invars = input_vars
        # output vars always appended last!
        for i, (activation, num_neurons, weights) in enumerate(layers):
            if hasLinear(activation):
                linvars, eqs = encode_linear_layer(invars, weights, num_neurons, i, net_prefix)
                vars.append(linvars)
                constraints.append(eqs)

                lin_layer = DefaultLayer('linear', num_neurons, invars, [], linvars, eqs)

                if activation == 'relu':
                    reluouts, reludeltas, reluineqs = encode_relu_layer(linvars, i, net_prefix)

                    vars.append(reludeltas)
                    vars.append(reluouts)
                    constraints.append(reluineqs)

                    net_layers.append(ReLULayer(lin_layer, num_neurons, invars, reludeltas, reluouts, reluineqs))

                    invars = reluouts
                elif activation == 'linear':
                    net_layers.append(lin_layer)

                    invars = linvars
            else:
                # just use weights = None for one_hot layer
                if activation == 'one_hot':
                    oh_outs, oh_vars, oh_constraints = encode_one_hot(invars, i, net_prefix)
                    vars.append(oh_vars)
                    vars.append(oh_outs)
                    constraints.append(oh_constraints)

                    net_layers.append(DefaultLayer('one_hot', num_neurons, invars, oh_vars, oh_outs, oh_constraints))

                    invars = oh_outs

                if activation == 'ranking':
                    rank_perms, rank_vars, rank_constraints = encode_ranking_layer(invars, i, net_prefix)
                    vars.append(rank_vars)
                    # rank_perms is permutation matrix !!!
                    vars.append(rank_perms)
                    constraints.append(rank_constraints)

                    # !!! not sure, what to take for output of ranking layer !!!
                    descriptor, k = output_mode

                    l = None
                    if descriptor == 'matrix' and k == -1:
                        l = DefaultLayer('ranking', num_neurons, invars, rank_vars, rank_perms, rank_constraints)
                    elif descriptor == 'matrix' and k >= 0:
                        l = DefaultLayer('ranking', num_neurons, invars, rank_vars + rank_perms[k:],
                                         rank_perms[:k], rank_constraints)
                    elif descriptor == 'out':
                        l = DefaultLayer('ranking', num_neurons, invars, rank_vars[:-num_neurons] + rank_perms,
                                         rank_vars[-num_neurons:], rank_constraints)
                    else:
                        raise ValueError('output mode ({desc}, {num}) does not exist'.format(desc=descriptor, num=k))

                    net_layers.append(l)
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
                    # modes 'vector' and 'out' are allowed and define what is returned as out
                    mode = activation.split('_')[-1]
                    oh_outs, oh_vars, oh_constraints = encode_sort_one_hot_layer(invars, i, net_prefix, mode)
                    vars.append(oh_vars)
                    vars.append(oh_outs)
                    constraints.append(oh_constraints)

                    net_layers.append(DefaultLayer('one_hot', num_neurons, invars, oh_vars, oh_outs, oh_constraints))
                    invars = oh_outs

        return net_layers

    def append_compare_layer(self, layers1, layers2, compared):
        _, num_outs1, _ = layers1[-1]
        _, num_outs2, _ = layers2[-1]

        if not num_outs1 == num_outs2:
            raise ValueError("both NNs must have the same number of outputs")

        if compared == 'one_hot':
            oh_layer = ('one_hot', num_outs1, None)
            layers1.append(oh_layer)
            layers2.append(oh_layer)
        elif compared in {'ranking', 'ranking_one_hot'} or compared.startswith('ranking_top_'):
            # not sure what to specify as num_neurons (num of sorted outs or num of p_ij in permutation matrix?)
            ranking_layer = ('ranking', num_outs1, None)
            layers1.append(ranking_layer)
            layers2.append(ranking_layer)
        elif compared.startswith('one_ranking_top_') or compared.startswith('optimize_ranking_top_'):
            # not sure what to specify as num_neurons (num of sorted outs or num of p_ij in permutation matrix?)
            ranking_layer = ('ranking', num_outs1, None)
            layers1.append(ranking_layer)
        elif compared.startswith('partial_top_') or compared.startswith('optimize_partial_top_'):
            k = int(compared.split('_')[-1])
            # not sure what to specify as num_neurons (num of sorted outs or num of p_ij in permutation matrix?)
            partial_layer = ('partial_{topk}'.format(topk=k), num_outs1, None)
            layers1.append(partial_layer)
        elif compared.startswith('one_hot_partial_top_') or compared == 'one_hot_diff':
            one_hot_layer = ('sort_one_hot_vector', num_outs1, None)
            layers1.append(one_hot_layer)
        else:
            raise ValueError('Invalid parameter \'compared\' for encode_equivalence: {name}'.format(name=compared))

        return layers1, layers2

    def determine_output_modes(self, compared):
        # should never be used
        mode1 = ('', -1)
        mode2 = ('', -1)
        if compared in {'outputs', 'one_hot', 'one_hot_diff'} or compared.startswith('one_hot_partial_top_'):
            mode1 = ('', -1)
            mode2 = ('', -1)
        elif compared == 'ranking_one_hot':
            # matrix[0:1] == matrix[0]
            mode1 = ('matrix', 1)
            mode2 = ('matrix', 1)
        elif compared.startswith('ranking_top_'):
            k = int(compared.split('_')[-1])
            mode1 = ('matrix', 1)
            mode2 = ('matrix', k)
        elif compared.startswith('one_ranking_top_') or compared.startswith('optimize_ranking_top_') \
                or compared.startswith('optimize_partial_top_') or compared.startswith('partial_top_'):
            mode1 = ('matrix', -1)
            mode2 = ('', -1)
        else:
            # default case
            raise ValueError('There is no ' + compared + ' keyword for param compared!!!')

        return mode1, mode2

    def encode_equivalence(self, layers1, layers2, input_lower_bounds, input_upper_bounds, compared='outputs',
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
            optimize_diff - calculates difference for each element of output of NN1 and NN2, needs manual optimization
                        function
        :return: encoding of the equivalence of NN1 and NN2 as a set of variables and
            mixed integer linear programming constraints
        '''

        layers1, layers2 = self.append_compare_layer(layers1, layers2, compared)
        mode1, mode2 = self.determine_output_modes(compared)

        self.input_layer = self.encode_inputs(input_lower_bounds, input_upper_bounds)
        self.a_layers = self.encode_layers(self.input_layer.get_outvars(), layers1, 'A', mode1)
        self.b_layers = self.encode_layers(self.input_layer.get_outvars(), layers2, 'B', mode2)

        # only need output for optimize_ranking_top_k
        outs1 = self.a_layers[-1].get_outvars()
        outs2 = self.b_layers[-1].get_outvars()
        eq_deltas, eq_diffs, eq_constraints = encode_equivalence_layer(outs1, outs2, comparator)

        self.equivalence_layer = DefaultLayer('equiv', -1, outs1 + outs2, eq_deltas, eq_diffs, eq_constraints)

    def get_vars(self):
        input_vars = self.input_layer.get_all_vars()
        net1_vars = [layer.get_all_vars() for layer in self.a_layers]
        net2_vars = [layer.get_all_vars() for layer in self.b_layers]
        equiv_vars = self.equivalence_layer.get_all_vars()

        return input_vars + net1_vars + net2_vars + equiv_vars

    def get_constraints(self):
        # right now input layer has no constraints
        input_constraints = self.input_layer.get_constraints()
        net1_constraints = [layer.get_constraints() for layer in self.a_layers]
        net2_constraints = [layer.get_constraints() for layer in self.b_layers]
        equiv_constraints = self.equivalence_layer.get_constraints()

        return input_constraints + net1_constraints + net2_constraints + equiv_constraints

    def encode_equivalence_from_file(self, path1, path2, input_lower_bounds, input_upper_bounds, compared='outputs',
                                     comparator='diff_zero'):
        kl1 = KerasLoader()
        kl1.load(path1)
        layers1 = kl1.getHiddenLayers()

        kl2 = KerasLoader()
        kl2.load(path2)
        layers2 = kl2.getHiddenLayers()

        self.encode_equivalence(layers1, layers2, input_lower_bounds, input_upper_bounds, compared, comparator)

    def optimize_variable(self, var, opt_vars, opt_constraints):
        model_ub = create_gurobi_model(opt_vars, opt_constraints,
                                       name='{vname} upper bound optimization'.format(vname=str(var)))
        model_ub.setObjective(var.to_gurobi(model_ub), grb.GRB.MAXIMIZE)

        model_lb = create_gurobi_model(opt_vars, opt_constraints,
                                       name='{vname} lower bound optimization'.format(vname=str(var)))
        model_lb.setObjective(var.to_gurobi(model_lb), grb.GRB.MINIMIZE)

        model_ub.setParam('TimeLimit', self.opt_timeout)
        model_lb.setParam('TimeLimit', self.opt_timeout)
        model_ub.optimize()
        model_lb.optimize()

        ub = model_ub.ObjBound
        lb = model_lb.ObjBound

        return lb, ub

    def optimize_layer(self, net, layer_idx):
        if layer_idx < 1:
            # for first layer we can't get better than interval arithmetic
            return

        if layer_idx == 1:
            opt_layers = []
            opt_layers.append(self.input_layer)
            opt_layers.append(net[layer_idx - 1])
        else:
            opt_layers = net[layer_idx - 2:layer_idx]

        opt_vars = opt_layers[0].get_outvars()[:]
        opt_vars += opt_layers[1].get_all_vars()[:]

        opt_constraints = opt_layers[1].get_constraints()

        for i, (var, constr) in enumerate(
                zip(net[layer_idx].get_optimization_vars(), net[layer_idx].get_optimization_constraints())):
            lb, ub = self.optimize_variable(var, opt_vars + [var], opt_constraints + [constr])
            var.update_bounds(lb, ub)


    def check_equivalence_layer(self, layer_idx):
        opt_vars = []
        opt_constrs = []
        if layer_idx == 0:
            opt_vars += self.input_layer.get_outvars()[:]
        else:
            a_outs = self.a_layers[layer_idx - 1].get_outvars()[:]
            b_outs = self.b_layers[layer_idx - 1].get_outvars()[:]
            opt_vars += a_outs + b_outs

            # at this stage we assume the previous layers to be equivalent
            for avar, bvar in zip(a_outs, b_outs):
                opt_constrs += [Linear(avar, bvar)]

        bounds = []

        for i, (a_var, a_constr, b_var, b_constr) in enumerate(
                zip(self.a_layers[layer_idx].get_optimization_vars(), self.a_layers[layer_idx].get_optimization_constraints(),
                    self.b_layers[layer_idx].get_optimization_vars(), self.b_layers[layer_idx].get_optimization_constraints())):
            diff = Variable(layer_idx, i, 'E', 'diff')
            diff_constr = Linear(Sum([a_var, Neg(b_var)]), diff)

            if i == 1:
                pretty_print(opt_vars + [a_var, b_var, diff], opt_constrs + [a_constr, b_constr, diff_constr])

            lb, ub = self.optimize_variable(diff, opt_vars + [a_var, b_var, diff], opt_constrs + [a_constr, b_constr, diff_constr])
            diff.update_bounds(lb, ub)

            bounds.append((lb, ub))

        return bounds