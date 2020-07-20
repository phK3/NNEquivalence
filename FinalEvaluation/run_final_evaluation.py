
import flags_constants as fc
from performance import Encoder
from expression_encoding import create_gurobi_model
import sys
from timeit import default_timer as timer
import pickle
import pandas as pd
import gurobipy as grb
import numpy as np
import os.path

examples = 'ExampleNNs/'


def encode_equiv_radius(path1, path2, input_los, input_his, equiv_mode, center, radius, metric, name):
    # accepts one_hot_partial_top_k as mode

    fc.use_asymmetric_bounds = True
    fc.use_context_groups = True
    fc.use_grb_native = False
    fc.use_eps_maximum = True
    fc.manhattan_use_absolute_value = True
    fc.epsilon = 1e-4

    enc = Encoder()
    enc.encode_equivalence_from_file(path1, path2, input_los, input_his, equiv_mode, equiv_mode)
    enc.add_input_radius(center, radius, metric)

    enc.optimize_constraints()

    k = int(equiv_mode.split('_')[-1])

    model = create_gurobi_model(enc.get_vars(), enc.get_constraints(), name)
    diff = model.getVarByName('E_diff_0_{num}'.format(num=k))
    model.setObjective(diff, grb.GRB.MAXIMIZE)
    model.setParam('TimeLimit', 30 * 60)

    # maximum for diff should be greater 0
    return model

def encode_equiv(path1, path2, input_los, input_his, mode, name):
    # accepts one_hot_partial_top_k

    fc.use_asymmetric_bounds = True
    fc.use_context_groups = True
    fc.use_grb_native = False
    fc.use_eps_maximum = True
    fc.manhattan_use_absolute_value = True
    fc.epsilon = 1e-4

    enc = Encoder()
    enc.encode_equivalence_from_file(path1, path2, input_los, input_his, mode, mode)

    enc.optimize_constraints()

    k = int(mode.split('_')[-1])

    model = create_gurobi_model(enc.get_vars(), enc.get_constraints(), name)
    diff = model.getVarByName('E_diff_0_{num}'.format(num=k))
    model.setObjective(diff, grb.GRB.MAXIMIZE)
    model.setParam('TimeLimit', 30 * 60)

    # maximum for diff should be greater 0
    return model


def run_hierarchical_cluster_evaluation(testname, path1='mnist8x8_70p_retrain.h5', path2='mnist8x8_80p_retrain.h5',
                                        no_clusters=10, no_steps=3, metric='manhattan', logdir='FinalEvaluation',
                                        obj_stop=20, timer_stop=1800, mode='one_hot_partial_top_3'):
    path1 = examples + path1
    path2 = examples + path2
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
            metric = metric

            name = testname + '_' + metric + '_cluster_{cl}_step_{s}'.format(cl=clno, s=s)

            logfile = logdir + '/' + name + '.txt'
            sys.stdout = open(logfile, 'w')

            model = encode_equiv_radius(path1, path2, inl, inh, mode, cluster.center, r, metric, name)
            # stop optimization, if counterexample with at least 10 difference is found
            model.setParam('BestObjStop', obj_stop)
            model.setParam('TimeLimit', timer_stop)
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


def run_radius_optimization(testname, path1='mnist8x8_70p_retrain.h5', path2='mnist8x8_80p_retrain.h5',
                            no_clusters=10, metric='manhattan', logdir='FinalEvaluation/VariableRadius', timer_stop=1800,
                            mode='one_hot_partial_top_3'):
    # returns empty lists, if no evaluation data was found
    path1 = examples + path1
    path2 = examples + path2
    inl = [0 for i in range(64)]
    inh = [16 for i in range(64)]

    # 10 most dense clusters in hierarchical manhattan clustering of mnist8x8 training data
    clusters_to_verify = pickle.load(open("to_verify.pickle", "rb"))
    # TODO: remove hack
    #df_fixed = pickle.load(open('FinalEvaluation/FixedRadius/dataframes/df_summary.pickle', 'rb'))
    df_fixed = pickle.load(open('FinalEvaluation/FixedRadius/dataframes/df_' + testname + '.pickle', 'rb'))

    models = []
    ins = []
    dict_list = []

    stdout = sys.stdout
    for clno, cluster in enumerate(clusters_to_verify[:no_clusters]):

        radius_lo, radius_hi = find_radius(testname, clno, df_fixed)

        if radius_lo < 0:
            # no evaluation data found
            break

        teststart = timer()

        name = testname + '_' + metric + '_cluster_{cl}'.format(cl=clno)

        logfile = logdir + '/' + name + '.txt'
        sys.stdout = open(logfile, 'w')

        enc, model = encode_r_opt(path1, path2, inl, inh, cluster.center, radius_lo,
                                  radius_hi, mode, time_limit=timer_stop)
        models.append(model)
        model.optimize()

        sys.stdout = stdout

        if model.SolCount > 0:
            inputs = [model.getVarByName('i_0_{idx}'.format(idx=j)).X for j in range(64)]
        else:
            inputs = 'No Solution found for {}'.format(name)

        ins.append(inputs)

        fname = logdir + '/' + name + '.pickle'
        with open(fname, 'wb') as fp:
            pickle.dump(inputs, fp)

        now = timer()
        print('### {name} finished. Total time elapsed: {t}'.format(name=name, t=now - teststart))
        print('    (val, bound) = ({v}, {bd})'.format(v=model.ObjVal, bd=model.ObjBound))
        print('    ins = {i}'.format(i=str(inputs)))

        if model.SolCount > 0:
            eval_dict = {'testname': testname, 'cluster': clno, 'obj': model.ObjVal,
                         'bound': model.ObjBound, 'time': now - teststart, 'logfile': logfile, 'inputfile': fname,
                         'model_name': model.getAttr('ModelName'),
                         'BoundVio': model.getAttr('BoundVio'),
                         'BoundVioIndex': model.getAttr('BoundVioIndex'),
                         'ConstrVio': model.getAttr('ConstrVio'),
                         'ConstrVioIndex': model.getAttr('ConstrVioIndex'),
                         'ConstrVioSum': model.getAttr('ConstrVioSum'),
                         'IntVio': model.getAttr('IntVio'),
                         'IntVioIndex': model.getAttr('IntVioIndex'),
                         'IntVioSum': model.getAttr('IntVioSum'),
                         'MaxBound': model.getAttr('MaxBound'),
                         'MaxCoeff': model.getAttr('MaxCoeff'),
                         'MaxRHS': model.getAttr('MaxRHS'),
                         'MinBound': model.getAttr('MinBound'),
                         'MinCoeff': model.getAttr('MinCoeff')}
        else:
            eval_dict = {'testname': testname, 'cluster': clno}

        dict_list.append(eval_dict)

    df = pd.DataFrame(dict_list)
    df.to_pickle(logdir + '/' + 'df_' + testname + '.pickle')

    return models, ins, dict_list


def run_no_cluster_evaluation(testname, path1='mnist8x8_70p_retrain.h5', path2='mnist8x8_80p_retrain.h5',
                              logdir='FinalEvaluation', obj_stop=20, timer_stop=1800,
                              mode='one_hot_partial_top_3'):
    path1 = examples + path1
    path2 = examples + path2
    inl = [0 for i in range(64)]
    inh = [16 for i in range(64)]

    stdout = sys.stdout
    teststart = timer()

    logfile = logdir + '/' + testname + '.txt'
    sys.stdout = open(logfile, 'w')

    model = encode_equiv(path1, path2, inl, inh, mode, testname)
    # stop optimization, if counterexample with at least obj_stop difference is found
    model.setParam('BestObjStop', obj_stop)
    model.setParam('TimeLimit', timer_stop)
    model.optimize()

    sys.stdout = stdout
    inputs = [model.getVarByName('i_0_{idx}'.format(idx=j)).X for j in range(64)]

    fname = logdir + '/' + testname + '.pickle'
    with open(fname, 'wb') as fp:
        pickle.dump(inputs, fp)

    now = timer()
    print('### {name} finished. Total time elapsed: {t}'.format(name=testname, t=now - teststart))
    print('    (val, bound) = ({v}, {bd})'.format(v=model.ObjVal, bd=model.ObjBound))
    print('    ins = {i}'.format(i=str(inputs)))

    eval_dict = {'testname': testname, 'obj': model.ObjVal, 'bound': model.ObjBound,
                 'time': now - teststart, 'logfile': logfile, 'inputfile': fname,
                 'model_name': model.getAttr('ModelName'),
                 'BoundVio': model.getAttr('BoundVio'),
                 'BoundVioIndex': model.getAttr('BoundVioIndex'),
                 'ConstrVio': model.getAttr('ConstrVio'),
                 'ConstrVioIndex': model.getAttr('ConstrVioIndex'),
                 'ConstrVioSum': model.getAttr('ConstrVioSum'),
                 'IntVio': model.getAttr('IntVio'),
                 'IntVioIndex': model.getAttr('IntVioIndex'),
                 'IntVioSum': model.getAttr('IntVioSum'),
                 'MaxBound': model.getAttr('MaxBound'),
                 'MaxCoeff': model.getAttr('MaxCoeff'),
                 'MaxRHS': model.getAttr('MaxRHS'),
                 'MinBound': model.getAttr('MinBound'),
                 'MinCoeff': model.getAttr('MinCoeff')}

    with open(logdir + '/dict_' + testname + '.pickle', 'wb') as fp:
        pickle.dump(eval_dict, fp)

    return model, inputs, eval_dict


def run_final_evaluation_clusters(time_limit=60*60*5, testrun=False, k_start=1, order='normal',
                                  nn1start=0, nn2start=0):
    if order == 'normal':
        nns = ['mnist8x8_lin.h5', 'mnist8x8_student_18_18_10.h5', 'mnist8x8_student_30_10.h5',
               'mnist8x8_70p_retrain.h5', 'mnist8x8_50p_retrain.h5', 'mnist8x8_20p_retrain.h5']
    elif order == 'other':
        # other order of networks
        nns = ['mnist8x8_20p_retrain.h5', 'mnist8x8_50p_retrain.h5', 'mnist8x8_70p_retrain.h5']

    model_list = []
    ins_list = []
    dicts_list = []

    t_start = timer()
    t_end = t_start + time_limit

    k = k_start

    while timer() < t_end and k <= 3:
        mode = 'one_hot_partial_top_{}'.format(k)

        for i in range(nn1start, len(nns)):
            # for now exclude comparison of same nns
            # TODO: remove hack
            rstart = i
            rstop = i+1
            if order == 'other':
                rstart = i + 1
                rstop = len(nns)

            for j in range(max(rstart, nn2start), rstop):
            #for j in range(i, i+1): # hack to include self equiv
            #for j in range(i + 1, len(nns)):
                if testrun:
                    timer_stop = 20
                    no_clusters = 1
                else:
                    # 30mins time limit for each optimization
                    timer_stop = 60*30
                    no_clusters = 5

                # [:-3] to exclude .h5 from name
                testname = '{}_vs_{}_{}'.format(nns[i][:-3], nns[j][:-3], mode)
                models, ins, dict_lists = run_hierarchical_cluster_evaluation(testname=testname, path1=nns[i],
                                                                              path2=nns[j], no_clusters=no_clusters,
                                                                              mode=mode, timer_stop=timer_stop)
                model_list.append(models)
                ins_list.append(ins)
                dicts_list.append(dict_lists)
        k += 1
        nn1start = 0
        nn2start = 0

    return model_list, ins_list, dicts_list


def run_final_evaluation_no_clusters(time_limit=60*60*5, testrun=False, k_start=1):
    nns = ['mnist8x8_lin.h5', 'mnist8x8_student_18_18_10.h5', 'mnist8x8_student_30_10.h5',
           'mnist8x8_70p_retrain.h5', 'mnist8x8_50p_retrain.h5', 'mnist8x8_20p_retrain.h5']

    model_list = []
    ins_list = []
    dicts_list = []

    t_start = timer()
    t_end = t_start + time_limit

    k = k_start

    while timer() < t_end and k <= 3:
        mode = 'one_hot_partial_top_{}'.format(k)

        for i in range(len(nns)):
            for j in range(i + 1, len(nns)):
                if testrun:
                    timer_stop = 20
                else:
                    # 30mins time limit for each optimization
                    timer_stop = 60*30

                # [:-3] to exclude .h5 from name
                testname = '{}_vs_{}_{}'.format(nns[i][:-3], nns[j][:-3], mode)
                model, ins, eval_dict = run_no_cluster_evaluation(testname=testname, path1=nns[i],
                                                                  path2=nns[j], mode=mode,
                                                                  timer_stop=timer_stop)
                model_list.append(model)
                ins_list.append(ins)
                dicts_list.append(eval_dict)
        k += 1

    df = pd.DataFrame(dicts_list)
    df.to_pickle('df_no_clusters.pickle')

    return model_list, ins_list, dicts_list


def encode_r_opt(path1, path2, inl, inh, center, radius_lo, radius_hi, mode, time_limit=30*60):
    fc.use_asymmetric_bounds = True
    fc.use_context_groups = True
    fc.use_grb_native = False
    fc.use_eps_maximum = True
    fc.manhattan_use_absolute_value = True
    fc.epsilon = 1e-4
    fc.not_equiv_tolerance = 1e-2

    enc = Encoder()
    enc.encode_equivalence_from_file(path1, path2, inl, inh, mode, mode)
    enc.add_input_radius(center, radius_hi, radius_mode='variable', radius_lo=radius_lo)

    enc.optimize_constraints()

    model = create_gurobi_model(enc.get_vars(), enc.get_constraints())
    r = model.getVarByName('r_0_0')
    model.setObjective(r, grb.GRB.MINIMIZE)
    model.setParam('TimeLimit', time_limit)

    return enc, model


def find_radius(tname, clno, df):
    dfr = df[(df['testname'] == tname) & (df['cluster'] == clno)]

    # no entry for combination of testname and cluster
    if dfr.empty:
        return -1, -1

    radius_lo = 0
    radius_hi = 16 * 64
    for s in [1 / 20, 1 / 10, 1 / 5]:
        dfr_step = dfr[dfr['step'] == s]
        # pandas returns dataframe with 1 row and 1 col -> need iloc to get float
        diff = dfr_step['obj'].iloc[0]
        r = dfr_step['radius'].iloc[0]
        if diff <= 0:
            radius_lo = r
        elif diff > fc.not_equiv_tolerance:
            radius_hi = r

    return radius_lo, radius_hi

def run_final_evaluation_radius_opt(time_limit=60*60*5, testrun=False, k_start=1, nn1_start=0, nn2_start=0):
    #nns = ['mnist8x8_lin.h5', 'mnist8x8_student_18_18_10.h5', 'mnist8x8_student_30_10.h5',
    #      'mnist8x8_70p_retrain.h5', 'mnist8x8_50p_retrain.h5', 'mnist8x8_20p_retrain.h5']

    # other order of networks
    #nns = ['mnist8x8_20p_retrain.h5', 'mnist8x8_50p_retrain.h5', 'mnist8x8_70p_retrain.h5']

    # retry student-30-10 vs 50p
    nns = ['mnist8x8_student_30_10.h5', 'mnist8x8_50p_retrain.h5']

    model_list = []
    ins_list = []
    dicts_list = []

    empty_tests = []

    t_start = timer()
    t_end = t_start + time_limit

    k = k_start

    while timer() < t_end and k <= 3:
        mode = 'one_hot_partial_top_{}'.format(k)

        for i in range(nn1_start, len(nns)):
            for j in range(max(i + 1, nn2_start), len(nns)):
                if testrun:
                    timer_stop = 20
                    no_clusters = 5
                else:
                    # 30mins time limit for each optimization
                    timer_stop = 60*30
                    no_clusters = 5

                # [:-3] to exclude .h5 from name
                testname = '{}_vs_{}_{}'.format(nns[i][:-3], nns[j][:-3], mode)
                model, ins, eval_dict = run_radius_optimization(testname=testname, path1=nns[i], path2=nns[j],
                                                                no_clusters=no_clusters, mode=mode, timer_stop=timer_stop)
                # no entry found for these nns
                if not model:
                    empty_tests.append((nns[i], nns[j], mode))
                else:
                    model_list.append(model)
                    ins_list.append(ins)
                    dicts_list.append(eval_dict)
        k += 1
        # started at (k, nn1s, nn2s) for larger ks want to check all nns again
        nn1_start = 0
        nn2_start = 0

    df = pd.DataFrame(dicts_list)
    df.to_pickle('df_variable_radius.pickle')

    return model_list, ins_list, dicts_list, empty_tests

def evaluate_hierarchical_cluster(testname, path1='mnist8x8_70p_retrain.h5', path2='mnist8x8_80p_retrain.h5',
                                clno=10, no_steps=3, metric='manhattan', logdir='FinalEvaluation',
                                obj_stop=20, bd_stop=-0.5, timer_stop=1800, mode='one_hot_partial_top_3'):
    path1 = examples + path1
    path2 = examples + path2
    inl = [0 for i in range(64)]
    inh = [16 for i in range(64)]

    # 10 most dense clusters in hierarchical manhattan clustering of mnist8x8 training data
    clusters_to_verify = pickle.load(open("to_verify.pickle", "rb"))

    dims = 64
    steps = [1/2, 1/4, 1/5, 1/10, 1/20]
    #steps = [1/2, 1/30, 1/40]

    models = []
    ins = []

    dict_list = []

    stdout = sys.stdout
    for s in steps[:no_steps]:
        cluster = clusters_to_verify[clno]
        teststart = timer()

        # manhattan distance
        r = s * cluster.distance

        name = testname + '_' + metric + '_cluster_{cl}_step_{s}'.format(cl=clno, s=s)

        logfile = logdir + '/' + name + '.txt'
        sys.stdout = open(logfile, 'w')

        model = encode_equiv_radius(path1, path2, inl, inh, mode, cluster.center, r, metric, name)

        model.setParam('BestObjStop', obj_stop)
        model.setParam('BestBdStop', bd_stop)
        model.setParam('TimeLimit', timer_stop)
        models.append(model)
        model.optimize()

        sys.stdout = stdout
        inputs = [model.getVarByName('i_0_{idx}'.format(idx=j)).X for j in range(64)]
        ins.append(inputs)

        fname = logdir + '/' + name + '.pickle'
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

        # only check radius from above, until nns no longer different
        if model.ObjVal <= 0:
            break

    df = pd.DataFrame(dict_list)
    df.to_pickle(logdir + '/df_' + testname + '.pickle')

    return models, ins, dict_list



def run_additional_evaluation_clusters(time_limit=60*60*5, testrun=False, k_start=1, k_end=1, clno=0,
                                       j_start=0, j_end=1000, logdir='FinalEvaluation/AdditionalEvaluation'):

    nns = ['mnist8x8_student_36_10.h5', 'mnist8x8_student_12_12_12_10.h5', 'mnist8x8_student_24_12_10.h5']
    #nns = ['mnist8x8_90p_retrain.h5', 'mnist8x8_80p_retrain.h5', 'mnist8x8_60p_retrain.h5',
    #       'mnist8x8_40p_retrain.h5', 'mnist8x8_30p_retrain.h5', 'mnist8x8_10p_retrain.h5']
    refnn = 'mnist8x8_lin.h5'

    model_list = []
    ins_list = []
    dicts_list = []

    t_start = timer()
    t_end = t_start + time_limit

    k = k_start

    while timer() < t_end and k <= k_end:
        mode = 'one_hot_partial_top_{}'.format(k)

        for j in range(j_start, min(j_end, len(nns))):
            if testrun:
                timer_stop = 20
            else:
                # 30mins time limit for each optimization
                timer_stop = 60*30

            # [:-3] to exclude .h5 from name
            testname = '{}_vs_{}_{}'.format(refnn[:-3], nns[j][:-3], mode)
            models, ins, dict_lists = evaluate_hierarchical_cluster(testname=testname, path1=refnn, path2=nns[j],clno=clno,
                                                                    logdir=logdir, obj_stop=1,
                                                                    timer_stop=timer_stop, mode=mode)

            model_list.append(models)
            ins_list.append(ins)
            dicts_list.append(dict_lists)
        k += 1

    return model_list, ins_list, dicts_list


def evaluate_radius_optimization(testname, radius_lo, radius_hi, path1='mnist8x8_70p_retrain.h5',
                                 path2='mnist8x8_80p_retrain.h5', clno=10, metric='manhattan',
                                 logdir='FinalEvaluation/VariableRadius', timer_stop=1800,
                                 mode='one_hot_partial_top_3', solution_file=None):
    # returns empty lists, if no evaluation data was found
    path1 = examples + path1
    path2 = examples + path2
    inl = [0 for i in range(64)]
    inh = [16 for i in range(64)]

    # 10 most dense clusters in hierarchical manhattan clustering of mnist8x8 training data
    clusters_to_verify = pickle.load(open("to_verify.pickle", "rb"))

    models = []
    ins = []
    dict_list = []

    cluster = clusters_to_verify[clno]

    stdout = sys.stdout
    teststart = timer()

    name = testname + '_' + metric + '_cluster_{cl}'.format(cl=clno)

    logfile = logdir + '/' + name + '.txt'
    sys.stdout = open(logfile, 'w')

    enc, model = encode_r_opt(path1, path2, inl, inh, cluster.center, radius_lo,
                              radius_hi, mode, time_limit=timer_stop)
    if solution_file:
        model.read(solution_file)
        model.update()

    models.append(model)
    model.optimize()

    sys.stdout = stdout

    if model.SolCount > 0:
        inputs = [model.getVarByName('i_0_{idx}'.format(idx=j)).X for j in range(64)]
    else:
        inputs = 'No Solution found for {}'.format(name)

    ins.append(inputs)

    fname = logdir + '/' + name + '.pickle'
    with open(fname, 'wb') as fp:
        pickle.dump(inputs, fp)

    now = timer()
    print('### {name} finished. Total time elapsed: {t}'.format(name=name, t=now - teststart))
    print('    (val, bound) = ({v}, {bd})'.format(v=model.ObjVal, bd=model.ObjBound))
    print('    ins = {i}'.format(i=str(inputs)))

    if model.SolCount > 0:
        eval_dict = {'testname': testname, 'cluster': clno, 'obj': model.ObjVal,
                     'bound': model.ObjBound, 'time': now - teststart, 'logfile': logfile, 'inputfile': fname,
                     'model_name': model.getAttr('ModelName'),
                     'BoundVio': model.getAttr('BoundVio'),
                     'BoundVioIndex': model.getAttr('BoundVioIndex'),
                     'ConstrVio': model.getAttr('ConstrVio'),
                     'ConstrVioIndex': model.getAttr('ConstrVioIndex'),
                     'ConstrVioSum': model.getAttr('ConstrVioSum'),
                     'IntVio': model.getAttr('IntVio'),
                     'IntVioIndex': model.getAttr('IntVioIndex'),
                     'IntVioSum': model.getAttr('IntVioSum'),
                     'MaxBound': model.getAttr('MaxBound'),
                     'MaxCoeff': model.getAttr('MaxCoeff'),
                     'MaxRHS': model.getAttr('MaxRHS'),
                     'MinBound': model.getAttr('MinBound'),
                     'MinCoeff': model.getAttr('MinCoeff')}
    else:
        eval_dict = {'testname': testname, 'cluster': clno}

    dict_list.append(eval_dict)

    df = pd.DataFrame(dict_list)
    df.to_pickle(logdir + '/' + 'df_' + testname + '.pickle')

    return models, ins, dict_list

def run_additional_evaluation_radius_opt(time_limit=60*60*5, testrun=False, k_start=1, k_end=1,
                                         nn_start=0, nn_end=100,
                                         directory='FinalEvaluation/AdditionalEvaluation/top2/',
                                         df_name='df_variable_radius', clusters=None):
    if not clusters:
        clusters = [0,1,2,3,4]

    #nns = ['mnist8x8_student_36_10.h5', 'mnist8x8_student_12_12_12_10.h5', 'mnist8x8_student_24_12_10.h5']
    #nns = ['mnist8x8_90p_retrain.h5', 'mnist8x8_80p_retrain.h5', 'mnist8x8_60p_retrain.h5',
    #       'mnist8x8_40p_retrain.h5', 'mnist8x8_30p_retrain.h5', 'mnist8x8_10p_retrain.h5']
    nns = ['mnist8x8_90p_retrain.h5', 'mnist8x8_80p_retrain.h5', 'mnist8x8_70p_retrain.h5',
           'mnist8x8_60p_retrain.h5', 'mnist8x8_50p_retrain.h5', 'mnist8x8_40p_retrain.h5',
           'mnist8x8_30p_retrain.h5', 'mnist8x8_20p_retrain.h5', 'mnist8x8_10p_retrain.h5']
    refnn = 'mnist8x8_lin.h5'

    # radius_h/l [cluster0, cluster1, ...] for each nn in nns
    # for k = 1
    """
    radius_los = [[6.8, 6.8, 6.8, 6.8, 13.6, 13.6],
                  [8.6, 8.6, 17.2, 17.2, 17.2, 17.2],
                  [2.56, 7.7, 19.25, 15.4, 19.25, 19.25],
                  [6, 14.5, 14.5, 14.5, 14.5, 14.5],
                  [5.7, 11.4, 11.4, 11.4, 11.4, 11.4]]
    radius_his = [[13.6, 13.6, 13.6, 13.6, 27.2, 27.2],
                  [17.2, 17.2, 43, 43, 43, 43],
                  [7.7, 15.4, 38.5, 38.5, 38.5, 38.5],
                  [14.5, 29, 29, 29, 29, 29],
                  [11.4, 22.8, 28.5, 22.8, 22.8, 22.8]]
    """
    # for k = 1 student nns
    """
    radius_los = [[10 for i in range(3)],
                  [12.5, 12.5, 18.9],
                  [25, 15, 20],
                  [15 for i in range(3)],
                  [20, 15, 15]]
    radius_his = [[17.5 for i in range(3)],
                  [25 for i in range(3)],
                  [32.5, 21, 26],
                  [25 for i in range(3)],
                  [32.5, 21, 32.5]]
    """
    # for k = 2

    radius_los = np.array([[ 8.49545739,  9.73472046, 12.06893292, 11.42882252, 14.50115273,
        12.34467454, 14.21174948, 11.53147137, 14.80919797],
       [ 9.14349471, 11.75877521, 18.5, 19.25842156, 22.97110569,
        20.55964865, 18.91349745, 22.72601289, 18.75246031],
       [ 11.5, 22, 27, 27.75, 25.19751363,
        27.5, 25.34286314, 23.35930689, 26.22844268],
       [ 5.93228736, 14.36809879, 13.59461373, 16.76782717, 19.02852781,
        19.02821702, 19.02892419, 18.28669633, 19.02892419],
       [ 5.93976088, 10.55476282, 22.79057401, 21.87319625, 19.10883133,
        20.9016096 , 19.47948759, 19.26479002, 20.00012839]])
    radius_his = [[20 for i in range(9)],
                  [32.5 for i in range(9)],
                  # previous radius_his[2]: [12, 22.5, 27.5, 28.75, 29, 28, 30.75, 30.75, 30.75],
                  [12, 22.5, 35, 35, 30, 30, 35, 35, 35],
                  [30, 30, 30, 30, 30, 30, 30, 24.5, 24.5],
                  [37.5 for i in range(9)]]


    # for students
    """
    radius_los = [[14, 12, 14],
                  [18, 14, 19],
                  [25, 17, 23],
                  [19, 18, 19],
                  [22, 14, 22]]
    radius_his = [[20 for i in range(3)],
                  [30 for i in range(3)],
                  [35 for i in range(3)],
                  [30 for i in range(3)],
                  [37.5 for i in range(3)]]
    """

    model_list = []
    ins_list = []
    dicts_list = []

    empty_tests = []

    t_start = timer()
    t_end = t_start + time_limit

    k = k_start
    while timer() < t_end and k <= k_end:
        mode = 'one_hot_partial_top_{}'.format(k)

        for j in range(nn_start, min(nn_end, len(nns))):
            if testrun:
                timer_stop = 20
            else:
                # 30mins time limit for each optimization
                #timer_stop = 60*30
                timer_stop = 3*60*60 # 3 hours

            for clno in clusters:
                # [:-3] to exclude .h5 from name
                testname = 'r_opt_{}_vs_{}_{}'.format(refnn[:-3], nns[j][:-3], mode)

                solfile = directory + 'mip_solutions/' + testname \
                          + '_cluster_' + str(clno) + '.sol'
                if not os.path.isfile(solfile):
                    solfile = None

                model, ins, eval_dict = evaluate_radius_optimization(testname=testname, radius_lo=radius_los[clno][j],
                                                                     radius_hi=radius_his[clno][j], path1=refnn,
                                                                     path2=nns[j], clno=clno,
                                                                     mode=mode, timer_stop=timer_stop,
                                                                     logdir=directory, solution_file=solfile)
                # no entry found for these nns
                if not model:
                    empty_tests.append((refnn, nns[j], mode))
                else:
                    model_list.append(model)
                    ins_list.append(ins)
                    dicts_list.append(eval_dict)
        k += 1

    df = pd.DataFrame(dicts_list)
    df.to_pickle(directory + df_name + '.pickle')

    return model_list, ins_list, dicts_list, empty_tests