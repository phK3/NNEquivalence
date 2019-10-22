
from analysis import get_optimization_data_dict, separate_logs
import pandas as pd

def get_info_from_name(experiment_name):
    s = experiment_name.split('_')
    info_dict = {}

    if s[0] == 'mnist':
        info_dict['nn1'] = 'mnist8x8'
    elif s[0] == 'mnist50':
        info_dict['nn1'] = 'mnist8x8_50p_retrain'

    if s[2] == '70':
        info_dict['nn2'] = 'mnist8x8_70p_retrain'
    elif s[2] == '50manhattan':
        info_dict['nn2'] = 'mnist8x8_50p_retrain'

    info_dict['cluster'] = int(s[-3])
    info_dict['step'] = float(s[-1])

    return info_dict

def output_to_df(logfile):
    """
    Converts the console output of run_hierarchical_cluster_evaluation() to a dataframe.
    Right now this is only intended to be used on the evaluation of October 22 2019

    :param logfile: the console output of the evaluation run
    :return: a dataframe containing
    """
    dictlist = []
    with open(logfile) as f:
        for line in f:
            dict = {}
            if line.startswith('###'):
                data = line.split(' ')
                name = data[1]
                dict['time'] = float(data[6])

                dir_name = 'Evaluation/hierarchical_clustering_big_steps/'
                dict['logfile'] = dir_name + name + '.txt'
                dict['inputfile'] = dir_name + name + '.pickle'

                data = next(f).split(' ')

                dict['radius'] = float(data[-1])
                # skip next two lines as they don't contain new information
                next(f)
                next(f)

                log_dict = get_optimization_data_dict(separate_logs(dict['logfile'])[-1])
                dict.update(log_dict)

                name_dict = get_info_from_name(name)
                dict.update(name_dict)

                dictlist.append(dict)

    df = pd.DataFrame(dictlist)
    df = df[['nn1', 'nn2', 'cluster', 'step', 'radius', 'obj', 'bound', 'time', 'grbTime',
             'raw_continuous', 'raw_integer', 'raw_binary', 'continuous', 'integer', 'binary',
             'rows', 'columns', 'nonzeros', 'logfile', 'inputfile']]
    return df


