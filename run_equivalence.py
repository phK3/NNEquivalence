
from performance import Encoder
import expression
from expression_encoding import pretty_print, interval_arithmetic, create_gurobi_model
import gurobipy as grb
import sys
import flags_constants as fc
import numpy as np
import pickle
from math import factorial
from timeit import default_timer as timer
import pandas as pd

examples = 'ExampleNNs/'

def balance_scale_eqiv_top_2():
    path = examples + 'balance_scale_lin.h5'
    inl = [1,1,1,1]
    inh = [5,5,5,5]

    enc = Encoder()
    enc.encode_equivalence_from_file(path, path, inl, inh, 'optimize_ranking_top_2', 'optimize_ranking_top_2')

    interval_arithmetic(enc.get_constraints())
    for i in range(1, 3):
        enc.optimize_layer(enc.a_layers, i)
        enc.optimize_layer(enc.b_layers, i)
        interval_arithmetic(enc.get_constraints())

    model = create_gurobi_model(enc.get_vars(), enc.get_constraints(), 'balance_scale_opt_ranking_top_2')
    diff = model.getVarByName('E_diff_0_2')
    model.setObjective(diff, grb.GRB.MAXIMIZE)
    model.setParam('TimeLimit', 10*60)

    # maximum for diff should be below 0
    return model

def balance_scale_eqiv_top_1():
    path = examples + 'balance_scale_lin.h5'
    inl = [1, 1, 1, 1]
    inh = [5, 5, 5, 5]

    enc = Encoder()
    enc.encode_equivalence_from_file(path, path, inl, inh, 'optimize_ranking_top_1', 'optimize_ranking_top_1')

    interval_arithmetic(enc.get_constraints())
    for i in range(1, 3):
        enc.optimize_layer(enc.a_layers, i)
        enc.optimize_layer(enc.b_layers, i)
        interval_arithmetic(enc.get_constraints())

    model1 = create_gurobi_model(enc.get_vars(), enc.get_constraints(), 'balance_scale_opt_ranking_top_1_diff1')
    diff = model1.getVarByName('E_diff_0_1')
    model1.setObjective(diff, grb.GRB.MAXIMIZE)
    model1.setParam('TimeLimit', 10 * 60)

    model2 = create_gurobi_model(enc.get_vars(), enc.get_constraints(), 'balance_scale_opt_ranking_top_1_diff2')
    diff = model2.getVarByName('E_diff_0_2')
    model2.setObjective(diff, grb.GRB.MAXIMIZE)
    model2.setParam('TimeLimit', 10 * 60)

    # maximum for both diffs should be below 0
    return model1, model2

def balance_scale_not_eqiv_top_2():
    path1 = examples + 'balance_scale_lin.h5'
    path2 = examples + 'balance_scale_lin2.h5'
    inl = [1, 1, 1, 1]
    inh = [5, 5, 5, 5]

    enc = Encoder()
    enc.encode_equivalence_from_file(path1, path2, inl, inh, 'optimize_ranking_top_2', 'optimize_ranking_top_2')

    interval_arithmetic(enc.get_constraints())

    model = create_gurobi_model(enc.get_vars(), enc.get_constraints(), 'balance_scale_opt_different_top_2')
    diff = model.getVarByName('E_diff_0_2')
    model.setObjective(diff, grb.GRB.MAXIMIZE)
    model.setParam('TimeLimit', 10*60)

    # maximal diff should be greater 0
    return model

def cancer_eqiv():
    path = examples + 'cancer_lin.h5'
    inl = [-3.18 for i in range(30)]
    inh = [11.74 for i in range(30)]

    enc = Encoder()
    enc.encode_equivalence_from_file(path, path, inl, inh, 'outputs', 'optimize_diff')

    interval_arithmetic(enc.get_constraints())
    for i in range(1, 4):
        enc.optimize_layer(enc.a_layers, i)
        enc.optimize_layer(enc.b_layers, i)
        interval_arithmetic(enc.get_constraints())

    model = create_gurobi_model(enc.get_vars(), enc.get_constraints(), 'cancer_lin_opt_diff')
    diff = model.getVarByName('E_diff_0_0')
    model.setObjective(diff, grb.GRB.MAXIMIZE)
    model.setParam('TimeLimit', 10 * 60)

    # maximum for diff should be exactly 0
    return model

def cancer_not_eqiv():
    path1 = examples + 'cancer_lin.h5'
    path2 = examples + 'cancer_lin2.h5'
    inl = [-3.18 for i in range(30)]
    inh = [11.74 for i in range(30)]

    enc = Encoder()
    enc.encode_equivalence_from_file(path1, path2, inl, inh, 'outputs', 'optimize_diff')

    interval_arithmetic(enc.get_constraints())
    for i in range(1, 4):
        enc.optimize_layer(enc.a_layers, i)
        enc.optimize_layer(enc.b_layers, i)
        interval_arithmetic(enc.get_constraints())

    model = create_gurobi_model(enc.get_vars(), enc.get_constraints(), 'cancer_lin_opt_diff')
    diff = model.getVarByName('E_diff_0_0')
    model.setObjective(diff, grb.GRB.MAXIMIZE)
    model.setParam('TimeLimit', 10 * 60)

    # maximum for diff should be greater 0
    return model

def mnist_not_eqiv(mode):
    # optimize_ranking_top_k allowed as mode for k = 1..9
    path1 = examples + 'mnist8x8_lin.h5'
    path2 = examples + 'mnist8x8_lin2.h5'
    inl = [0 for i in range(64)]
    inh = [16 for i in range(64)]

    enc = Encoder()
    enc.encode_equivalence_from_file(path1, path2, inl, inh, mode, mode)

    interval_arithmetic(enc.get_constraints())
    for i in range(1, 3):
        enc.optimize_layer(enc.a_layers, i)
        enc.optimize_layer(enc.b_layers, i)
        interval_arithmetic(enc.get_constraints())

    k = int(mode.split('_')[-1])
    model = create_gurobi_model(enc.get_vars(), enc.get_constraints(), 'mnist_lin_' + mode)
    diff = model.getVarByName('E_diff_0_{num}'.format(num=k))
    model.setObjective(diff, grb.GRB.MAXIMIZE)
    model.setParam('TimeLimit', 30 * 60)

    # maximum for diff should be greater 0
    return model

def mnist_eqiv(mode):
    # optimize_ranking_top_k allowed as mode for k = 1..9
    path = examples + 'mnist8x8_lin.h5'
    inl = [0 for i in range(64)]
    inh = [16 for i in range(64)]

    enc = Encoder()
    enc.encode_equivalence_from_file(path, path, inl, inh, mode, mode)

    interval_arithmetic(enc.get_constraints())
    for i in range(1, 3):
        enc.optimize_layer(enc.a_layers, i)
        enc.optimize_layer(enc.b_layers, i)
        interval_arithmetic(enc.get_constraints())

    if mode == 'one_hot_diff':
        k = 0
    else:
        k = int(mode.split('_')[-1])

    model = create_gurobi_model(enc.get_vars(), enc.get_constraints(), 'mnist_lin_' + mode)
    diff = model.getVarByName('E_diff_0_{num}'.format(num=k))
    model.setObjective(diff, grb.GRB.MAXIMIZE)
    model.setParam('TimeLimit', 30 * 60)

    # maximum for diff should be greater 0
    return model


def encode_equiv(path1, path2, input_los, input_his, mode, name):
    # accepts one_hot_partial_top_k, one_hot_diff as mode
    enc = Encoder()
    enc.encode_equivalence_from_file(path1, path2, input_los, input_his, mode, mode)

    interval_arithmetic(enc.get_constraints())
    for i in range(1, 3):
        enc.optimize_layer(enc.a_layers, i)
        enc.optimize_layer(enc.b_layers, i)
        interval_arithmetic(enc.get_constraints())

    if mode == 'one_hot_diff':
        k = 0
    else:
        k = int(mode.split('_')[-1])

    model = create_gurobi_model(enc.get_vars(), enc.get_constraints(), name)
    diff = model.getVarByName('E_diff_0_{num}'.format(num=k))
    model.setObjective(diff, grb.GRB.MAXIMIZE)
    model.setParam('TimeLimit', 30 * 60)

    # maximum for diff should be greater 0
    return model


def all_combinations(base, digits, c_num):
    if c_num >= base ** digits:
        raise ValueError('Insufficient number of combinations')

    array = []
    for i in range(digits):
        mod = c_num % base
        array.insert(0, mod)
        c_num = (c_num - mod) // base

    return array


def set_branch_priorities(model, s, delta, pi):
    for v in model.getVars():
        if 's' in v.varName:
            v.setAttr('BranchPriority', s)
        elif 'pi' in v.varName:
            v.setAttr('BranchPriority', pi)
        elif 'd' in v.varName and 'diff' not in v.varName:
            v.setAttr('BranchPriority', delta)
        else:
            v.setAttr('BranchPriority', 0)

    model.update()


def evaluate_branching(limit_minutes):
    fc.use_grb_native = False

    stdout = sys.stdout
    models = []
    teststart = timer()

    '''
    # test MIPFocus = 3
    sys.stdout = open('Evaluation/mnist_eqiv_branch_MIPFocus3.txt', 'w')
    start = timer()
    model = mnist_eqiv('one_hot_partial_top_3')

    model.setParam('MIPFocus', 3)
    model.update()

    model.setParam('TimeLimit', limit_minutes * 60)

    model.optimize()
    end = timer()
    models.append(model)
    print('### Total Time elapsed: {t}'.format(t=end - start))

    sys.stdout = stdout
    now = timer()
    print('mnist_eqiv_branch_MIPFocus3 evaluated, time={t}'.format(t=now - teststart))
    '''

    for combination in range(0, 27):
        priorities = all_combinations(3, 3, combination)
        s = priorities[0]
        delta = priorities[1]
        pi = priorities[2]

        sys.stdout = open('Evaluation/mnist_eqiv_branch_s={set}_delta={d}_pi={p}.txt'.format(set=s, d=delta, p=pi), 'w')
        start = timer()
        model = mnist_eqiv('one_hot_partial_top_3')

        set_branch_priorities(model, s, delta, pi)
        model.setParam('TimeLimit', limit_minutes * 60)

        model.optimize()
        end = timer()
        models.append(model)
        print('### Total Time elapsed: {t}'.format(t=end - start))

        sys.stdout = stdout
        now = timer()
        print('mnist_eqiv_branch_s={set}_delta={d}_pi={p} evaluated, time={t}'.format(set=s, d=delta, p=pi,t=now - teststart))

    return models


def encode_equiv_radius(path1, path2, input_los, input_his, equiv_mode, center, radius, metric, name):
    # accepts one_hot_partial_top_k, one_hot_diff as mode

    fc.use_asymmetric_bounds = True
    fc.use_context_groups = True
    fc.use_grb_native = False
    fc.use_eps_maximum = True
    fc.manhattan_use_absolute_value = True
    fc.epsilon = 1e-4

    enc = Encoder()
    enc.encode_equivalence_from_file(path1, path2, input_los, input_his, equiv_mode, equiv_mode)
    enc.add_input_radius(center, radius, metric)

    interval_arithmetic(enc.get_constraints())
    for i in range(1, 3):
        if '30_10' in path1:
            if i < 2:
                enc.optimize_layer(enc.a_layers, i)
        else:
            enc.optimize_layer(enc.a_layers, i)

        if '30_10' in path2:
            if i < 2:
                enc.optimize_layer(enc.b_layers, i)
        else:
            enc.optimize_layer(enc.b_layers, i)
        interval_arithmetic(enc.get_constraints())

    if equiv_mode == 'one_hot_diff':
        k = 0
    else:
        k = int(equiv_mode.split('_')[-1])

    model = create_gurobi_model(enc.get_vars(), enc.get_constraints(), name)
    diff = model.getVarByName('E_diff_0_{num}'.format(num=k))
    model.setObjective(diff, grb.GRB.MAXIMIZE)
    model.setParam('TimeLimit', 30 * 60)

    # maximum for diff should be greater 0
    return model

def encode_optimize_radius(path1, path2, input_los, input_his, equiv_mode, center, radius_lo, radius_hi, metric, name):
    # accepts one_hot_partial_top_k as mode
    # also one_hot_diff ???

    fc.use_asymmetric_bounds = True
    fc.use_context_groups = True
    fc.use_grb_native = False
    fc.use_eps_maximum = True
    fc.manhattan_use_absolute_value = True
    fc.epsilon = 1e-4

    enc = Encoder()
    enc.encode_equivalence_from_file(path1, path2, input_los, input_his, equiv_mode, equiv_mode)
    enc.add_input_radius(center, radius_hi, metric, radius_mode='variable', radius_lo=radius_lo)

    interval_arithmetic(enc.get_constraints())
    for i in range(1, 3):
        enc.optimize_layer(enc.a_layers, i)
        enc.optimize_layer(enc.b_layers, i)
        interval_arithmetic(enc.get_constraints())

    model = create_gurobi_model(enc.get_vars(), enc.get_constraints(), name)
    r = model.getVarByName('r_0_0')
    model.setObjective(r, grb.GRB.MINIMIZE)
    model.setParam('TimeLimit', 30 * 60)

    # for obj val of r -> NNs are different
    # for (bound val - eps) of r -> NNs are equivalent
    return model


def run_radius_optimization(testname, path1='mnist8x8_70p_retrain.h5', path2='mnist8x8_80p_retrain.h5', cluster_idx=0,
                            radius_lo=6.8, radius_hi=27.2):
    path1 = examples + path1
    path2 = examples + path2
    mode = 'one_hot_partial_top_3'
    inl = [0 for i in range(64)]
    inh = [16 for i in range(64)]

    # 10 most dense clusters in hierarchical manhattan clustering of mnist8x8 training data
    clusters_to_verify = pickle.load(open("to_verify.pickle", "rb"))

    clno = cluster_idx
    cluster = clusters_to_verify[clno]

    models = []
    ins = []

    stdout = sys.stdout
    teststart = timer()

    # manhattan distance
    metric = 'manhattan'

    name = testname + '_manhattan_cluster_{cl}_radius_opt'.format(cl=clno)

    sys.stdout = open('Evaluation/' + name + '.txt', 'w')

    model = encode_optimize_radius(path1, path2, inl, inh, mode, cluster.center, radius_lo, radius_hi, metric, name)
    models.append(model)
    model.optimize()

    sys.stdout = stdout
    inputs = [model.getVarByName('i_0_{idx}'.format(idx=j)).X for j in range(64)]
    ins.append(inputs)

    fname = name + '.pickle'
    with open(fname, 'wb') as fp:
        pickle.dump(inputs, fp)

    now = timer()
    print('### {name} finished. Total time elapsed: {t}'.format(name=name, t=now - teststart))
    print('    radius = (val, bound) = ({v}, {bd})'.format(v=model.ObjVal, bd=model.ObjBound))
    print('    ins = {i}'.format(i=str(inputs)))

    teststart = timer()

    radius = model.ObjBound - 0.1
    name = testname + '_manhattan_cluster_{cl}_radius_test'.format(cl=clno)

    sys.stdout = open('Evaluation/' + name + '.txt', 'w')
    model = encode_equiv_radius(path1, path2, inl, inh, mode, cluster.center, radius, metric, name)
    models.append(model)
    model.optimize()

    sys.stdout = stdout
    inputs = [model.getVarByName('i_0_{idx}'.format(idx=j)).X for j in range(64)]
    ins.append(inputs)

    fname = name + '.pickle'
    with open(fname, 'wb') as fp:
        pickle.dump(inputs, fp)

    now = timer()
    print('### {name} finished. Total time elapsed: {t}'.format(name=name, t=now - teststart))
    print('    radius = {r}'.format(r=radius))
    print('    (val, bound) = ({v}, {bd})'.format(v=model.ObjVal, bd=model.ObjBound))
    print('    ins = {i}'.format(i=str(inputs)))

    return models, ins

def run_radius_evaluation():
    path70 = examples + 'mnist8x8_70p_retrain.h5'
    path80 = examples + 'mnist8x8_80p_retrain.h5'
    mode = 'one_hot_partial_top_3'
    inl = [0 for i in range(64)]
    inh = [16 for i in range(64)]

    # centroid of 0-cluster, when clustering training data using manhattan metric
    center = np.array([[0, 0, 4, 13, 11, 2, 0, 0],
                       [0, 1, 12, 13, 12, 10, 0, 0],
                       [0, 3, 14, 4, 3, 12, 3, 0],
                       [0, 5, 12, 1, 0, 9, 6, 0],
                       [0, 5, 11, 0, 0, 8, 7, 0],
                       [0, 3, 13, 1, 1, 11, 6, 0],
                       [0, 0, 13, 10, 10, 13, 2, 0],
                       [0, 0, 4, 13, 13, 5, 0, 0]])

    center = center.reshape(-1)

    # average manhattan distance to centroid of all training images that were assigned to the 0 cluster
    avg_dist = 98.90813648293964

    dims = 64
    steps = [1/40, 1/20, 1/10, 1/5, 1/3, 1/2]

    models = []
    ins = []

    stdout = sys.stdout
    teststart = timer()
    for s in steps:
        for i in range(2):
            if i % 2 == 0:
                # manhattan distance
                r = s * avg_dist
                metric = 'manhattan'

            else:
                # chebyshev distance
                # choose radius, s.t. manhattan ball and chebyshev ball have same volume
                r = ((s * avg_dist)**dims / factorial(dims))**(1/dims)
                metric = 'chebyshev'

            name = 'mnist_70_vs_80_{metric}_0_step_{s}'.format(metric=metric, s=s)

            sys.stdout = open('Evaluation/' + name + '.txt', 'w')

            model = encode_equiv_radius(path70, path80, inl, inh, mode, center, r, metric, name)
            models.append(model)
            model.optimize()

            sys.stdout = stdout
            inputs = [model.getVarByName('i_0_{idx}'.format(idx=j)).X for j in range(64)]
            ins.append(inputs)

            fname = name + '.pickle'
            with open(fname, 'wb') as fp:
                pickle.dump(inputs, fp)

            now = timer()
            print('### {name} finished. Total time elapsed: {t}'.format(name=name, t=now - teststart))
            print('    radius = {r}'.format(r=r))
            print('    (val, bound) = ({v}, {bd})'.format(v=model.ObjVal, bd=model.ObjBound))
            print('    ins = {i}'.format(i=str(inputs)))

    return models, ins


def run_hierarchical_cluster_evaluation(testname, path1='mnist8x8_70p_retrain.h5', path2='mnist8x8_80p_retrain.h5',
                                        no_clusters=10, no_steps=3):
    path1 = examples + path1
    path2 = examples + path2
    mode = 'one_hot_partial_top_3'
    inl = [0 for i in range(64)]
    inh = [16 for i in range(64)]

    # 10 most dense clusters in hierarchical manhattan clustering of mnist8x8 training data
    clusters_to_verify = pickle.load(open("to_verify.pickle", "rb"))

    dims = 64
    steps = [1/20, 1/10, 1/5]

    models = []
    ins = []

    dict_list = []

    stdout = sys.stdout
    for s in steps[:no_steps]:
        for clno, cluster in enumerate(clusters_to_verify[:no_clusters]):
            teststart = timer()

            # manhattan distance
            r = s * cluster.distance
            metric = 'manhattan'

            name = testname + '_manhattan_cluster_{cl}_step_{s}'.format(cl=clno, s=s)
            #name = 'mnist_70_vs_80_manhattan_cluster_{cl}_step_{s}'.format(cl=clno, s=s)

            logfile = 'Evaluation/' + name + '.txt'
            sys.stdout = open(logfile, 'w')

            model = encode_equiv_radius(path1, path2, inl, inh, mode, cluster.center, r, metric, name)
            # stop optimization, if counterexample with at least 10 difference is found
            model.setParam('BestObjStop', 10.0)
            models.append(model)
            model.optimize()

            sys.stdout = stdout
            inputs = [model.getVarByName('i_0_{idx}'.format(idx=j)).X for j in range(64)]
            ins.append(inputs)

            fname = name + '.pickle'
            with open(fname, 'wb') as fp:
                pickle.dump(inputs, fp)

            now = timer()
            print('### {name} finished. Total time elapsed: {t}'.format(name=name, t=now - teststart))
            print('    radius = {r}'.format(r=r))
            print('    (val, bound) = ({v}, {bd})'.format(v=model.ObjVal, bd=model.ObjBound))
            print('    ins = {i}'.format(i=str(inputs)))

            eval_dict = {'testname': testname, 'cluster': clno, 'step': s, 'radius': r, 'obj': model.ObjVal,
                         'bound': model.ObjBound, 'time': now - teststart, 'logfile': logfile, 'inputfile': fname}

            dict_list.append(eval_dict)

    df = pd.DataFrame(dict_list)
    df.to_pickle('df_' + testname + '.pickle')

    return models, ins, dict_list


def run_student_evaluation():
    fc.use_asymmetric_bounds = True
    fc.use_context_groups = True
    fc.use_grb_native = False
    fc.use_eps_maximum = True
    fc.manhattan_use_absolute_value = True
    fc.epsilon = 1e-4

    path = examples + 'mnist8x8_50p_student.h5'
    mode = 'one_hot_partial_top_3'
    inl = [0 for i in range(64)]
    inh = [16 for i in range(64)]

    stdout = sys.stdout
    teststart = timer()

    name = 'mnist8x8_50p_student_equiv'
    sys.stdout = open('Evaluation/' + name + '.txt', 'w')

    model = encode_equiv(path, path, inl, inh, mode, name)
    model.setParam('TimeLimit', 60*60)
    model.setParam('MIPFocus', 3)
    model.update()
    model.optimize()

    sys.stdout = stdout
    now = timer()
    print('### {name} finished. Total time elapsed: {t}'.format(name=name, t=now - teststart))

    return model

def run_evaluation():
    fc.use_grb_native = False

    stdout = sys.stdout
    models = []
    teststart = timer()

    '''
    sys.stdout = open('Evaluation/balance_scale_eqiv_top1.txt', 'w')
    start = timer()
    model1, model2 = balance_scale_eqiv_top_1()
    model1.optimize()
    model2.optimize()
    end = timer()
    models.append(model1)
    models.append(model2)
    print('### Total Time elapsed: {t}'.format(t=end - start))

    sys.stdout = stdout
    now = timer()
    print('balance_scale_eqiv_top1 evaluated, time={t}'.format(t=now - teststart))

    sys.stdout = open('Evaluation/balance_scale_eqiv_top2.txt', 'w')
    start = timer()
    model = balance_scale_eqiv_top_2()
    model.optimize()
    end = timer()
    models.append(model)
    print('### Total Time elapsed: {t}'.format(t=end - start))

    sys.stdout = stdout
    now = timer()
    print('balance_scale_eqiv_top2 evaluated, time={t}'.format(t=now - teststart))

    sys.stdout = open('Evaluation/balance_scale_not_eqiv_top2.txt', 'w')
    start = timer()
    model = balance_scale_not_eqiv_top_2()
    model.optimize()
    end = timer()
    models.append(model)
    print('### Total Time elapsed: {t}'.format(t=end - start))

    sys.stdout = stdout
    now = timer()
    print('balance_scale_not_eqiv_top2 evaluated, time={t}'.format(t=now - teststart))

    sys.stdout = open('Evaluation/cancer_eqiv.txt', 'w')
    start = timer()
    model = cancer_eqiv()
    model.optimize()
    end = timer()
    models.append(model)
    print('### Total Time elapsed: {t}'.format(t=end - start))

    sys.stdout = stdout
    now = timer()
    print('cancer_eqiv evaluated, time={t}'.format(t=now - teststart))

    sys.stdout = open('Evaluation/cancer_not_eqiv.txt', 'w')
    start = timer()
    model = cancer_not_eqiv()
    model.optimize()
    end = timer()
    models.append(model)
    print('### Total Time elapsed: {t}'.format(t=end - start))

    sys.stdout = stdout
    now = timer()
    print('cancer_not_eqiv evaluated, time={t}'.format(t=now - teststart))

    for k in range(1,4):
        sys.stdout = open('Evaluation/mnist_not_eqiv_ranking_top_{num}.txt'.format(num=k), 'w')
        start = timer()
        model = mnist_not_eqiv('optimize_ranking_top_{num}'.format(num=k))
        model.optimize()
        end = timer()
        models.append(model)
        print('### Total Time elapsed: {t}'.format(t=end - start))

        sys.stdout = stdout
        now = timer()
        print('mnist_not_eqiv_ranking_top_{num} evaluated, time={t}'.format(t=now - teststart, num=k))

    for k in range(1, 4):
        sys.stdout = open('Evaluation/mnist_eqiv_ranking_top_{num}.txt'.format(num=k), 'w')
        start = timer()
        model = mnist_eqiv('optimize_ranking_top_{num}'.format(num=k))
        model.optimize()
        end = timer()
        models.append(model)
        print('### Total Time elapsed: {t}'.format(t=end - start))

        sys.stdout = stdout
        now = timer()
        print('mnist_not_eqiv_ranking_top_{num} evaluated, time={t}'.format(t=now - teststart, num=k))
    

    sys.stdout = open('Evaluation/mnist_eqiv_one_hot_diff.txt', 'w')
    start = timer()
    model = mnist_eqiv('one_hot_diff')
    model.optimize()
    end = timer()
    models.append(model)
    print('### Total Time elapsed: {t}'.format(t=end - start))

    sys.stdout = stdout
    now = timer()
    print('mnist_eqiv_one_hot_diff evaluated, time={t}'.format(t=now - teststart))
    

    for k in range(1, 4):
        sys.stdout = open('Evaluation/mnist_not_eqiv_one_hot_partial_top_{num}.txt'.format(num=k), 'w')
        start = timer()
        model = mnist_not_eqiv('one_hot_partial_top_{num}'.format(num=k))
        model.optimize()
        end = timer()
        models.append(model)
        print('### Total Time elapsed: {t}'.format(t=end - start))

        sys.stdout = stdout
        now = timer()
        print('mnist_not_eqiv_one_hot_partial_top_{num} evaluated, time={t}'.format(t=now - teststart, num=k))

    for k in range(1, 4):
        sys.stdout = open('Evaluation/mnist_eqiv_one_hot_partial_top_{num}.txt'.format(num=k), 'w')
        start = timer()
        model = mnist_eqiv('one_hot_partial_top_{num}'.format(num=k))
        model.optimize()
        end = timer()
        models.append(model)
        print('### Total Time elapsed: {t}'.format(t=end - start))

        sys.stdout = stdout
        now = timer()
        print('mnist_eqiv_one_hot_partial_top_{num} evaluated, time={t}'.format(t=now - teststart, num=k))
    
    branching_models = evaluate_branching(10)
    models.append(branching_models)

    
    expression.use_asymmetric_bounds = True
    for k in range(1, 5):
        sys.stdout = open('Evaluation/mnist_vs_p{per}_one_hot_partial_top_3.txt'.format(per=10*k), 'w')
        start = timer()
        path1 = examples + 'mnist8x8_lin.h5'
        path2 = examples + 'mnist8x8_{per}p.h5'.format(per=10*k)
        inl = [0 for i in range(64)]
        inh = [16 for i in range(64)]
        mode = 'one_hot_partial_top_3'
        model = encode_equiv(path1, path2, inl, inh, mode, 'mnist_vs_p{per}_one_hot_partial_top_3'.format(per=10*k))
        model.optimize()
        end = timer()
        models.append(model)
        print('### Total Time elapsed: {t}'.format(t=end - start))

        sys.stdout = stdout
        now = timer()
        print('mnist_vs_p_{per}_one_hot_partial_top_3 evaluated, time={t}'.format(t=now - teststart, per=10*k))


    for k in range(3, 5):
        sys.stdout = open('Evaluation/mnist_p{per}_eqiv_one_hot_partial_top_3.txt'.format(per=10*k), 'w')
        start = timer()
        path = examples + 'mnist8x8_{per}p.h5'.format(per=10*k)
        inl = [0 for i in range(64)]
        inh = [16 for i in range(64)]
        mode = 'one_hot_partial_top_3'
        model = encode_equiv(path, path, inl, inh, mode, 'mnist_p{per}_eqiv_one_hot_partial_top_3'.format(per=10*k))
        model.optimize()
        end = timer()
        models.append(model)
        print('### Total Time elapsed: {t}'.format(t=end - start))

        sys.stdout = stdout
        now = timer()
        print('mnist_p{per}_eqiv_one_hot_partial_top_3 evaluated, time={t}'.format(t=now - teststart, per=10*k))


    '''
    fc.use_asymmetric_bounds = True
    for k in range(5, 10):
        name = 'mnist_vs_p{per}_retrain_one_hot_partial_top_3.txt'.format(per=10 * k)
        sys.stdout = open('Evaluation/' + name, 'w')
        start = timer()
        path1 = examples + 'mnist8x8_lin.h5'
        path2 = examples + 'mnist8x8_{per}p_retrain.h5'.format(per=10 * k)
        inl = [0 for i in range(64)]
        inh = [16 for i in range(64)]
        mode = 'one_hot_partial_top_3'
        model = encode_equiv(path1, path2, inl, inh, mode, name)
        model.optimize()
        end = timer()
        models.append(model)
        print('### Total Time elapsed: {t}'.format(t=end - start))

        sys.stdout = stdout
        now = timer()
        print(name + ', time={t}'.format(t=now - teststart))

    for k in range(5, 10):
        name = 'mnist_p{per}_retrain_eqiv_one_hot_partial_top_3.txt'.format(per=10 * k)
        sys.stdout = open('Evaluation/' + name, 'w')
        start = timer()
        path = examples + 'mnist8x8_{per}p_retrain.h5'.format(per=10 * k)
        inl = [0 for i in range(64)]
        inh = [16 for i in range(64)]
        mode = 'one_hot_partial_top_3'
        model = encode_equiv(path, path, inl, inh, mode, name)
        model.optimize()
        end = timer()
        models.append(model)
        print('### Total Time elapsed: {t}'.format(t=end - start))

        sys.stdout = stdout
        now = timer()
        print(name + ', time={t}'.format(t=now - teststart))


    '''
    models[-2].setParam('TimeLimit', 300 * 60)
    models[-2].optimize()
    '''

    return models