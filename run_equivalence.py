
from performance import Encoder
import expression
from expression_encoding import pretty_print, interval_arithmetic, create_gurobi_model
import gurobipy as grb
import sys
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

    k = int(mode.split('_')[-1])
    model = create_gurobi_model(enc.get_vars(), enc.get_constraints(), 'mnist_lin_' + mode)
    diff = model.getVarByName('E_diff_0_{num}'.format(num=k))
    model.setObjective(diff, grb.GRB.MAXIMIZE)
    model.setParam('TimeLimit', 30 * 60)

    # maximum for diff should be greater 0
    return model

def run_evaluation(models):
    expression.use_grb_native = False

    stdout = sys.stdout
    models = []
    teststart = timer()

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

    models[-1].setParam('TimeLimit', 300 * 60)
    models[-1].optimize()

    return models