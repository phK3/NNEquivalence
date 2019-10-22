from expression import Expression, Variable, ffp
from expression_encoding import flatten, encode_NN_from_file, interval_arithmetic
import gurobipy as grb
import numpy as np
import matplotlib.pyplot as plt
import itertools as itt
import texttable as tt
import pandas as pd
import re


def print_table(vars, model):
    var_dict = {'': [], 'A': [], 'B': [], 'E': []}

    for v in flatten(vars):
        net, _, _ = v.getIndex()
        var_dict[net].append((v, model.getVarByName(str(v)).X))
        #var_dict[net].append('{v_name} = {value}'.format(v_name=str(v), value=model.getVarByName(str(v)).X))

    tab = tt.Texttable()
    headings = ['input', 'NN 1', 'NN 2', 'diff', 'Equivalence']
    tab.header(headings)

    def none_handler(x):
        if x == None:
            return ''
        else:
            return x

    for i, a, b, e in itt.zip_longest(var_dict[''], var_dict['A'], var_dict['B'], var_dict['E'], fillvalue=('', None)):
        istring = '{v_name} = {value}'.format(v_name=str(i[0]), value=none_handler(i[1]))
        astring = '{v_name} = {value}'.format(v_name=str(a[0]), value=none_handler(a[1]))
        bstring = '{v_name} = {value}'.format(v_name=str(b[0]), value=none_handler(b[1]))
        estring = '{v_name} = {value}'.format(v_name=str(e[0]), value=none_handler(e[1]))

        #if not a[1] == None and not b[1] == None:
        #    diff = a[1] - b[1]
        #else:
        #    diff = ''

        if none_handler(a[1]) == none_handler(b[1]):
            diff = ''
        else:
            diff = '!!'

        row = [istring, astring, bstring, diff, estring]

        tab.add_row(row)

    s = tab.draw()
    print(s)


def check_outputs(nn_file, ins, sort=True, printing=True):
    nn_vars, nn_constraints = encode_NN_from_file(nn_file, ins, ins, '')
    interval_arithmetic(nn_constraints)

    outs = nn_vars[-1]
    if sort:
        outs = sorted(outs, key=lambda x: x.lo)

    if printing:
        for v in outs:
            print(str(v) + ' : ' + str(v.lo))

    return outs


def compare_outputs(nn1, nn2, ins, sort=False):
    outs1 = check_outputs(nn1, ins, sort, printing=False)
    outs2 = check_outputs(nn2, ins, sort, printing=False)

    tab = tt.Texttable()
    if sort:
        headings = ['NN 1', 'NN 2']
    else:
        headings = ['Neuron', 'NN 1', 'NN 2']

    tab.header(headings)

    for i, (a, b) in enumerate(zip(outs1, outs2)):
        if sort:
            astring = str(a) + ' : ' + str(a.lo)
            bstring = str(b) + ' : ' + str(b.lo)
            tab.add_row([astring, bstring])
        else:
            tab.add_row([i, ffp(a.lo), ffp(b.lo)])

    s = tab.draw()
    print(s)


def plot_diffmap(in1, in2, vmin, vmax):
    diff = np.subtract(in1, in2)

    fig = plt.figure()
    s = fig.add_subplot(1, 1, 1, xlabel='x', ylabel='y')
    im = s.imshow(diff, interpolation='nearest', cmap='bwr', vmin=vmin, vmax=vmax)
    fig.colorbar(im)
    return fig


def plot_grb_solution(model, xdim, ydim):
    solution = [model.getVarByName('i_0_{idx}'.format(idx=j)).X for j in range(xdim * ydim)]

    fig = plt.figure()
    s = fig.add_subplot(1, 1, 1, xlabel='x', ylabel='y')
    im = s.imshow(np.array(solution).reshape(xdim, ydim), interpolation='nearest', cmap=plt.cm.binary)
    return fig

def separate_logs(logfile):
    logs = []
    current_log = ''
    with open(logfile) as f:
        for line in f.readlines():
            current_log += line

            # new optimization
            if line.startswith('Optimize'):
                logs.append(current_log)
                current_log = line

    logs.append(current_log)
    return logs


def get_table(logstring, res_name=''):
    # 11 columns:
    # SolFound Expl Unexpl Obj Depth IntInf Incumbent BestBd Gap It/Node Time
    table = pd.DataFrame(columns=['SolFound', 'Expl', 'Unexpl', 'Obj', 'Depth', 'IntInf',
                                  'Incumbent', 'BestBd', 'Gap', 'It/Node', 'Time'])

    if not res_name == '':
        table.name = res_name

    def fill_row_dict(columns):
        int_idxs = [0,1,2,4,5,10]
        float_idxs = [3,6,7,8,9]

        columns[-1] = columns[-1][:-1]  # remove second's s

        if columns[0] == '':
            filled = [0]
        else:
            filled = [columns[0]]

        for i, e in enumerate(columns[1:]):
            real_idx = i+1
            if real_idx in int_idxs:
                if e == '':
                    filled.append(0)
                else:
                    filled.append(int(e))
            else:
                if e in ['-', '']:
                    filled.append(float('nan'))
                elif e.endswith('%'):
                    filled.append(float(e[:-1]))
                else:
                    filled.append(float(e))

        return filled

    num_start = False
    for line in logstring.splitlines():
        tabline = re.sub(r'\ (\ )*', ';', line)
        if tabline.startswith(';0'):
            num_start = True

        if tabline.startswith('H') and num_start:
            if not tabline[1:].startswith(';'):
                tabline = tabline[0] + ';' + tabline[1:]

            # should be 8 columns (Obj, InfInf, Depth missing)
            columns = tabline.split(';')
            columns[0] = '1'
            columns.insert(3, '')  # empty Obj column
            columns.insert(4, '')  # empty Depth column
            columns.insert(5, '')  # empty InfInf column
            tabline = ';'.join(columns)
        elif tabline.startswith('*') and num_start:
            if not tabline[1:].startswith(';'):
                tabline = tabline[0] + ';' + tabline[1:]

            # should be 9 columns (Obj, IntInf missing)
            columns = tabline.split(';')
            columns[0] = '2'
            columns.insert(3, '')  # empty Obj column
            columns.insert(5, '')  # empty InfInf column
            tabline = ';'.join(columns)
        elif ('cutoff' in tabline or 'infeasible' in tabline) and num_start:
            # for cutoff/infeas obj is cutoff/infeas instead of number
            columns = tabline.split(';')
            columns[3] = ''  # empty Obj column
            columns.insert(5, '')  # empty InfInf column
            tabline = ';'.join(columns)

        if len(tabline.split(';')) < 11 and num_start:
            return table

        if num_start:
            #print(tabline + ' ### start={s}'.format(s=num_start))
            columns = tabline.split(';')
            table = table.append(pd.Series(fill_row_dict(columns), index=table.columns), ignore_index=True)

    return table


def get_optimization_data_dict(logstring):
    data_dict = {}
    first_occurrence_var_types = True

    for line in logstring.splitlines():
        if line.startswith('Presolved:'):
            data = line.split(' ')
            data_dict['rows'] = int(data[1])
            data_dict['columns'] = int(data[3])
            data_dict['nonzeros'] = int(data[5])
        elif line.startswith('Variable types:'):
            data = line.split(' ')
            if first_occurrence_var_types:
                data_dict['raw_continuous'] = int(data[2])
                data_dict['raw_integer'] = int(data[4])
                # ignore leading '('
                data_dict['raw_binary'] = int(data[6][1:])
                first_occurrence_var_types = False
            else:
                data_dict['continuous'] = int(data[2])
                data_dict['integer'] = int(data[4])
                # ignore leading '('
                data_dict['binary'] = int(data[6][1:])
        elif line.startswith('Explored'):
            data = line.split(' ')
            data_dict['grbTime'] = float(data[7])
        elif line.startswith('Best objective'):
            data = line.split(' ')
            # ignore trailing commata
            data_dict['obj'] = float(data[2][:-1])
            data_dict['bound'] = float(data[5][:-1])

    return data_dict
