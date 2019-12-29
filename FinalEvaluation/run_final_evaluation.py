
import flags_constants as fc
from performance import Encoder
from expression_encoding import create_gurobi_model
import sys
from timeit import default_timer as timer
import pickle
import pandas as pd
import gurobipy as grb

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


def run_final_evaluation_clusters(time_limit=60*60*5, testrun=False):
    nns = ['mnist8x8_lin.h5', 'mnist8x8_student_18_18_10.h5', 'mnist8x8_student_30_10.h5',
           'mnist8x8_70p_retrain.h5', 'mnist8x8_50p_retrain.h5', 'mnist8x8_20p_retrain.h5']

    model_list = []
    ins_list = []
    dicts_list = []

    t_start = timer()
    t_end = t_start + time_limit

    k = 1

    while timer() < t_end:
        mode = 'one_hot_partial_top_{}'.format(k)

        for i in range(len(nns)):
            # for now exclude comparison of same nns
            for j in range(i + 1, len(nns)):
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

    return model_list, ins_list, dicts_list