from expression import Expression, Variable
from expression_encoding import flatten
import gurobipy as grb
import itertools as itt
import texttable as tt


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

