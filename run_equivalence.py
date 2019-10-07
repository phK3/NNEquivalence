
from performance import Encoder
import expression
from expression_encoding import pretty_print, interval_arithmetic, create_gurobi_model
import gurobipy as grb
import sys
import flags_constants as fc
from timeit import default_timer as timer

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
    expression.use_asymmetric_bounds = True
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