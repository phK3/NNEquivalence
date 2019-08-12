
import gurobipy as grb


def lp_sort(sort_vec, mode=''):
    n = len(sort_vec)

    model = grb.Model()

    # add vars for permutation matrix
    ps = []
    for i in range(0, n):
        p_row = []
        for j in range(0, n):
            if mode == 'sos':
                pij = model.addVar(vtype=grb.GRB.CONTINUOUS, ub=1, name='p_{row}_{col}'.format(row=i, col=j))
            else:
                pij = model.addVar(vtype=grb.GRB.BINARY, name='p_{row}_{col}'.format(row=i, col=j))

            p_row.append(pij)

        ps.append(p_row)

    # add vars to be sorted
    xs = []
    for i in range(0, n):
        x = model.addVar(lb=sort_vec[i], ub=sort_vec[i], vtype=grb.GRB.CONTINUOUS, name='x_{idx}'.format(idx=i))
        xs.append(x)

    # output vec
    os = []
    for i in range(0, n):
        o = model.addVar(vtype=grb.GRB.CONTINUOUS, name='o_{idx}'.format(idx=i))
        os.append(o)

    model.update()
    model.setObjective(0)

    # multiplication with permutation matrix
    for i in range(0, n):

        model.addConstr(grb.quicksum(ps[i][j] * xs[j] for j in range(0, n)) == os[i], name='MatMul_{idx}'.format(idx=i))

    # ensure output is sorted
    for i in range(0, n-1):
        model.addConstr(os[i] >= os[i + 1], name='sort_{idx}'.format(idx=i))

    # enforce that one pij per row and column is 1
    for i in range(0, n):
        model.addConstr(grb.quicksum(ps[i][j] for j in range(0, n)) == 1, name='One_row_{idx}'.format(idx=i))

    for j in range(0, n):
        model.addConstr(grb.quicksum(ps[i][j] for i in range(0, n)) == 1, name='One_col_{idx}'.format(idx=j))

    if mode == 'sos':
        # only one for each row can be 1
        for i in range(0, n):
            model.addSOS(grb.GRB.SOS_TYPE1, ps[i])

        # only one for each column can be 1
        for j in range(0, n):
            model.addSOS(grb.GRB.SOS_TYPE1, [row[j] for row in ps])

    model.update()

    return model


def ranking_encoding(sort_vec):
    n = len(sort_vec)
    m = min(sort_vec)
    M = max(sort_vec)

    model = grb.Model()

    ins = []
    os = []
    for i in range(0, n):
        ini = model.addVar(lb=sort_vec[i], ub=sort_vec[i], vtype=grb.GRB.CONTINUOUS, name='i_{idx}'.format(idx=i))
        ins.append(ini)

        oi = model.addVar(lb=0, ub=n, vtype=grb.GRB.CONTINUOUS, name='o_{idx}'.format(idx=i))
        os.append(oi)

    deltas = []
    for i in range(0, n):
        d_rows = []
        for j in range(0, n):
            if i == j:
                d = None
            else:
                d = model.addVar(vtype=grb.GRB.BINARY,obj=1, name='d_{great}_{less}'.format(great=i, less=j))
            d_rows.append(d)

        deltas.append(d_rows)

    model.update()

    for i in range(0, n):
        for j in range(0, n):
            if not i == j:
                # i_i > i_j -> d_i_j = 1
                model.addConstr(ins[i] - ins[j] <= M * deltas[i][j], name='imp_{idxi}_gt_{idxj}'.format(idxi=i, idxj=j))

    for i in range(0, n):
        model.addConstr(grb.quicksum(deltas[i][j] for j in range(0, n) if not j == i) == os[i])

    model.setAttr('ModelSense', grb.GRB.MINIMIZE)
    model.update()

    return model