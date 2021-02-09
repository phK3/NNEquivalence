
from abc import ABC, abstractmethod
from expression import Expression, Variable, Linear, Sum, Neg, Constant, Geq, Abs, Multiplication
from keras_loader import KerasLoader
import flags_constants as fc
import numpy as np
from expression_encoding import encode_equivalence, interval_arithmetic, hasLinear, encode_linear_layer, \
    encode_relu_layer, encode_one_hot, encode_ranking_layer, encode_equivalence_layer, create_gurobi_model, pretty_print, \
    encode_partial_layer, encode_sort_one_hot_layer, flatten
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
        self.constraints = []

    def add_input_constraints(self, in_constraints, additional_vars):
        self.constraints += in_constraints
        self.intervars += additional_vars

    def get_optimization_vars(self):
        return []

    def get_optimization_constraints(self):
        return self.constraints

    def get_all_vars(self):
        return self.invars + self.intervars


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

        self.equiv_mode = None
        self.radius_mode = None

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

    def pretty_print(self):
        pretty_print(self.get_vars(), self.get_constraints())

    def create_gurobi_model(self):
        """
        Creates gurobi model as specified by constraints added through the methods encode_equivalence or
        encode_equivalence_from_file and add_input_radius.

        :return: A gurobi model of the equivalence property encoded
        """
        if not self.equiv_mode:
            raise ValueError('No equivalence mode specified!')

        r_str = self.radius_mode
        if not self.radius_mode:
            r_str = ''

        model = create_gurobi_model(self.get_vars(), self.get_constraints())

        if r_str == 'variable':
            r = model.getVarByName('r_0_0')
            model.setObjective(r, grb.GRB.MINIMIZE)
        elif self.equiv_mode.startswith('one_hot_partial_top_'):
            k = self.equiv_mode.split('_')[-1]
            diff = model.getVarByName('E_diff_0_' + k)
            model.setObjective(diff, grb.GRB.MAXIMIZE)
        elif self.equiv_mode.startswith('optimize_diff_'):
            diff = model.getVarByName('E_norm_1_0')
            model.setObjective(diff, grb.GRB.MAXIMIZE)

        return model

    def encode_equiv(self, reference_nn, test_nn, input_lower_bounds, input_upper_bounds, mode):
        if not mode.startswith(('optimize_diff_', 'one_hot_partial_top_')):
            raise ValueError('Mode {} is not supported!\nSupported modes are: \n\toptimize_diff_[manhattan | chebyshev]'
                             '\n\tone_hot_partial_top_[k]')

        if mode.startswith('optimize_diff_'):
            self.encode_equivalence_from_file(reference_nn, test_nn, input_lower_bounds, input_upper_bounds, 'outputs', mode)
        elif mode.startswith('one_hot_partial_top_'):
            self.encode_equivalence_from_file(reference_nn, test_nn, input_lower_bounds, input_upper_bounds, mode, mode)




    def add_input_radius(self, center, radius, metric='manhattan', radius_mode='constant', radius_lo=0):
        '''
        Constrains input values, s.t. they have to be within a circle around a specified center
         of specified radius according to a specified metric.

         If radius_mode = 'variable' is chosen, the radius, for which the difference of the top-values is
         positive (i.e. the NNs are not equivalent) is minimized.
         -> For the found solutions of the radius there are counterexamples to equivalence
         -> For the calculated bounds, the NNs are equivalent (at least within bound - eps they should be)

        :param center: The center of the circle
        :param radius: The radius of the circle if radius_mode = 'constant' or the upper bound on the radius variable,
            if radius_mode = 'variable'
        :param metric: either chebyshev OR manhattan is supported
        :param radius_mode: 'constant' - the radius is a constant and the difference between two nn2
            around the center with this radius can be optimized.
            'variable' - the radius is a variable and the radius, for which two nns are equivalent can be optimized
        :param radius_lo: lower bound of the radius, if radius_mode = 'variable' is selected
        :return:
        '''

        self.radius_mode = radius_mode

        def add_absolute_value_constraints(radius, dimension, netPrefix, centered_inputs):
            ineqs = []
            additional_vars = []

            deltas = [Variable(0, i, netPrefix, 'd', 'Int') for i in range(dimension)]
            abs_outs = [Variable(0, i, netPrefix, 'abs') for i in range(dimension)]
            ineqs.append([Abs(ci, aout, d) for ci, aout, d in zip(centered_inputs, abs_outs, deltas)])
            ineqs.append(Geq(radius, Sum(abs_outs)))

            additional_vars.append(deltas)
            additional_vars.append(abs_outs)

            return ineqs, additional_vars

        def add_lp_constraints(radius, dimension, netPrefix, centered_inputs):
            ineqs = []
            additional_vars = []

            abs_outs = [Variable(0, i, netPrefix, 'abs') for i in range(dimension)]
            ineqs.append([Geq(aout, ci) for aout, ci in zip(abs_outs, centered_inputs)])
            ineqs.append([Geq(aout, Neg(ci)) for aout, ci in zip(abs_outs, centered_inputs)])
            ineqs.append(Geq(radius, Sum(abs_outs)))

            # absolute values must be greater equal 0
            # interval arithmetic is not sufficient to compute these bounds, as it is only enforced by combination of
            # the two inequalities
            # for v in abs_outs:
            #    v.setLo(0)

            additional_vars.append(abs_outs)

            return ineqs, additional_vars

        def add_direct_constraints(radius, dimension, centered_inputs):
            ineqs = []
            for i in range(2 ** dimension):
                terms = []
                for j in range(dimension):
                    neg = (i // 2 ** j) % 2
                    if neg > 0:
                        terms.append(Neg(centered_inputs[j]))
                    else:
                        terms.append(centered_inputs[j])

                ineqs.append(Geq(radius, Sum(terms)))

            return ineqs, []

        if not metric in ['manhattan', 'chebyshev']:
            raise ValueError('Metric {m} is not supported!'.format(m=metric))

        invars = self.input_layer.get_outvars()
        dim = len(invars)

        if not len(center) == dim:
            raise ValueError('Center has dimension {cdim}, but input has dimension {idim}'.format(cdim=len(center),
                                                                                                idim=dim))

        for i, invar in enumerate(invars):
            invar.update_bounds(center[i] - radius, center[i] + radius)

        netPrefix, _, _ = invars[0].getIndex()
        additional_vars = []
        additional_ineqs = []
        r = None

        if radius_mode == 'constant':
            # need float as somehow gurobi can't handle float64 as type
            r = Constant(float(radius), netPrefix, 0, 0)
        elif radius_mode == 'variable':
            r = Variable(0, 0, netPrefix, 'r')
            r.update_bounds(float(radius_lo), float(radius))
            additional_vars.append(r)

            diff = self.equivalence_layer.get_outvars()[-1]
            # can't append E_diff_0_1 >= eps in input layer, otherwise individual bounds optimization has no definition
            # for E_diff_0_1, which is defined in equivalence layer
            self.equivalence_layer.constraints.append(Geq(diff, Constant(fc.not_equiv_tolerance, netPrefix, 0, 0)))
            #additional_ineqs.append(Geq(diff, Constant(fc.not_equiv_tolerance, netPrefix, 0, 0)))
            #additional_ineqs.append(Geq(diff, Constant(0, netPrefix, 0, 0)))
        else:
            raise ValueError('radius_mode: {} is not supported!'.format(radius_mode))

        if metric == 'chebyshev' and radius_mode == 'variable':
            for i, invar in enumerate(invars):
                center_i = Constant(float(center[i]), netPrefix, 0, 0)
                additional_ineqs.append(Geq(invar, Sum([center_i, Neg(r)])))
                additional_ineqs.append(Geq(Sum([center_i, r]), invar))

        if metric == 'manhattan':
            centered_inputs = []

            for i in range(dim):
                centered_inputs.append(Sum([invars[i], Neg(Constant(center[i], netPrefix, 0, i))]))

            if fc.manhattan_use_absolute_value:
                ineqs, constraint_vars = add_absolute_value_constraints(r, dim, netPrefix, centered_inputs)
            elif fc.manhattan_use_lp_constraints:
                ineqs, constraint_vars = add_lp_constraints(r, dim, netPrefix, centered_inputs)
            else:
                ineqs, constraint_vars = add_direct_constraints(r, dim, centered_inputs)

            additional_vars += constraint_vars
            additional_ineqs += ineqs

        self.input_layer.add_input_constraints(additional_ineqs, additional_vars)

    def calc_cluster_boundary(self, c1, c2, epsilon):
        c1 = np.array(c1)
        c2 = np.array(c2)

        factors = c2 - c1
        constant = (epsilon / 2) * np.linalg.norm(c2 - c1)**2
        constant += (np.linalg.norm(c1)**2 - np.linalg.norm(c2)**2) / 2

        invars = self.input_layer.get_outvars()
        netPrefix, _, _ = invars[0].getIndex()
        zero = Constant(0, netPrefix, 0, 0)

        terms = [Multiplication(Constant(factor, netPrefix, 0, 0), i) for factor, i in zip(factors, invars)]
        terms.append(Constant(constant, netPrefix, 0, 0))

        bound = [Geq(zero, Sum(terms))]

        return bound

    def add_convex_hull_restriction(self, cluster_trees, center, epsilon=0.5, bounds=None):
        """
        Adds convex hull constraints to the input of the NNs.

        Inputs need to be (hierarchically) clustered. The bounds are then calculated as the
        boundaries of the voronoi region of the current cluster-center
        :param cluster_trees: List of cluster-trees
        :param center: The cluster-center that corresponds to the input region that about to be inpspected
        :param epsilon: Ratio of how close to the convex hull the boundaries should be.
            points up until epsilon / 2 of the distance to the next cluster centers are feasible
            for epsilon = 1, points on the boundary of the convex hull are feasible,
            for epsilon = 0, only the center is feasible
        :param bounds: existing bounds on the input
        :return: epsilon-bounds of the voronoi region around the cluster-center
        """

        if bounds is None:
            bounds = []

        center = np.array(center)

        dists = [np.linalg.norm(center - c.center) for c in cluster_trees]
        min_i = np.argmin(dists)
        cluster = cluster_trees[min_i]

        bounds += [self.calc_cluster_boundary(cluster.center, c2.center, epsilon)
                   for c2 in cluster_trees if not np.array_equal(c2.center, cluster.center)]

        if not np.array_equal(cluster.center, center):
            bounds += self.add_convex_hull_restriction(cluster.get_children(), center, epsilon, bounds)
            return bounds
        else:
            self.input_layer.add_input_constraints(bounds, [])

            in_layer_vars = self.input_layer.get_all_vars()
            for v in self.input_layer.get_outvars():
                lb, ub = self.optimize_variable(v, in_layer_vars, bounds)
                v.update_bounds(lb, ub)

            return bounds

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
        elif compared == 'outputs':
            # don't have to append any layer
            pass
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
        self.equiv_mode = comparator

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

        if not fc.bounds_gurobi_print_to_console:
            model_ub.setParam('LogToConsole', 0)
            model_lb.setParam('LogToConsole', 0)

        model_ub.setParam('TimeLimit', self.opt_timeout)
        model_lb.setParam('TimeLimit', self.opt_timeout)
        model_ub.optimize()
        model_lb.optimize()

        ub = model_ub.ObjBound
        lb = model_lb.ObjBound

        return lb, ub

    def optimize_layer(self, net, layer_idx, metric='chebyshev'):
        if metric =='chebyshev' and layer_idx < 1:
            # for first layer we can't get better than interval arithmetic
            return

        if layer_idx == 0:
            # for manhattan we need the input layer constraints for tightening
            opt_layers = []
            opt_layers.append(self.input_layer)

            opt_vars = opt_layers[0].get_all_vars()[:]
            opt_constraints = opt_layers[0].get_constraints()[:]
        elif layer_idx == 1:
            opt_layers = []
            opt_layers.append(self.input_layer)
            opt_layers.append(net[layer_idx - 1])

            if metric == 'manhattan':
                opt_vars = opt_layers[0].get_all_vars()[:]
                opt_vars += opt_layers[1].get_all_vars()[:]

                opt_constraints = opt_layers[0].get_constraints()[:]
                opt_constraints += opt_layers[1].get_constraints()[:]
        else:
            opt_layers = net[layer_idx - 2:layer_idx]

        if metric == 'chebyshev' or layer_idx > 1:
            opt_vars = opt_layers[0].get_outvars()[:]
            opt_vars += opt_layers[1].get_all_vars()[:]

            opt_constraints = opt_layers[1].get_constraints()

        if fc.bounds_create_individual_models:
            for i, (var, constr) in enumerate(
                    zip(net[layer_idx].get_optimization_vars(), net[layer_idx].get_optimization_constraints())):
                lb, ub = self.optimize_variable(var, opt_vars + [var], opt_constraints + [constr])
                var.update_bounds(lb, ub)
        else:
            bounds_variables = net[layer_idx].get_optimization_vars()[:]
            bounds_constraints = net[layer_idx].get_optimization_constraints()[:]
            self.optimize_variables(bounds_variables, opt_vars, opt_constraints + bounds_constraints)

    def optimize_variables(self, opt_vars, vars, constraints):
        model_vars = opt_vars + vars
        m = create_gurobi_model(model_vars, constraints, name='bounds optimization model')

        if not fc.bounds_gurobi_print_to_console:
            m.setParam('LogToConsole', 0)

        for v in opt_vars:
            m.reset()
            m.setObjective(v.to_gurobi(m), grb.GRB.MAXIMIZE)
            m.setParam('TimeLimit', self.opt_timeout)
            m.optimize()
            ub = m.ObjBound

            m.reset()
            m.setObjective(v.to_gurobi(m), grb.GRB.MINIMIZE)
            m.setParam('TimeLimit', self.opt_timeout)
            m.optimize()
            lb = m.ObjBound

            v.update_bounds(lb, ub)



    def optimize_net(self, net, metric='chebyshev'):
        for i in range(len(net)):
            if not net[i].activation == 'one_hot':
                self.optimize_layer(net, i, metric=metric)
                interval_arithmetic(self.get_constraints())


    def optimize_constraints(self, metric='chebyshev'):
        interval_arithmetic(self.get_constraints())
        self.optimize_net(self.a_layers, metric=metric)
        self.optimize_net(self.b_layers, metric=metric)


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