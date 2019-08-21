from expression import Expression, Variable
from expression_encoding import flatten
import gurobipy as grb
import itertools as itt
import texttable as tt


def print_table(vars, model):
    var_dict = {'': [], 'A': [], 'B': [], 'E': []}

    for v in flatten(vars):
        net, _, _ = v.getIndex()
        var_dict[net].append('{v_name} = {value}'.format(v_name=str(v), value=model.getVarByName(str(v)).X))

    tab = tt.Texttable()
    headings = ['input', 'NN 1', 'NN 2', 'Equivalence']
    tab.header(headings)

    for row in itt.zip_longest(var_dict[''], var_dict['A'], var_dict['B'], var_dict['E'], fillvalue=''):
        tab.add_row(row)

    s = tab.draw()
    print(s)

