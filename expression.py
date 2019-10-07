from abc import ABC, abstractmethod
from numpy import format_float_positional
import numbers
import flags_constants as fc
import gurobipy as grb

def ffp(x):
    if x < 0:
        s = format_float_positional(-x, trim='-')
        return '(- ' + s + ')'
    else:
        return format_float_positional(x, trim='-')


def flatten(list):
    return [x for sublist in list for x in sublist]


def makeLeq(lhs, rhs):
    return '(assert (<= ' + lhs + ' ' + rhs + '))'


def makeGeq(lhs, rhs):
    # maybe switch to other representation later
    return makeLeq(rhs, lhs)


def makeEq(lhs, rhs):
    return '(assert (= ' + lhs + ' ' + rhs + '))'


def makeLt(lhs, rhs):
    return '(assert (< ' + lhs + ' ' + rhs + '))'


def makeGt(lhs, rhs):
    return makeLt(rhs, lhs)


class Expression(ABC):

    def __init__(self, net, layer, row):
        self.lo = -fc.default_bound
        self.hi = fc.default_bound
        self.hasLo = False
        self.hasHi = False

        self.net = net
        self.layer = layer
        self.row = row
        pass

    def getIndex(self):
        return (self.net, self.layer, self.row)

    @abstractmethod
    def to_smtlib(self):
        pass


    @abstractmethod
    def to_gurobi(self, model):
        pass

    def getHi(self):
        return self.hi

    def getLo(self):
        return self.lo

    def getLo_exclusive(self):
        return self.lo - fc.epsilon

    def getHi_exclusive(self):
        return self.hi + fc.epsilon

    @abstractmethod
    def tighten_interval(self):
        pass

    def update_bounds(self, l, h):
        if l > self.lo:
            self.lo = l
            self.hasLo = True
        if h < self.hi:
            self.hi = h
            self.hasHi = True


class Constant(Expression):

    def __init__(self, value, net, layer, row):
        # Any idea how to get rid of net, layer, row for constants?
        super(Constant, self).__init__(net, layer, row)
        self.value = value
        self.hi = value
        self.lo = value
        self.hasLo = True
        self.hasHi = True

    def tighten_interval(self):
        pass

    def to_smtlib(self):
        s = None
        if isinstance(self.value, numbers.Integral):
            if self.value < 0:
                s = '(- ' + str(-self.value) + ')'
            else:
                s = str(self.value)
        else:
            s = ffp(self.value)

        return s

    def to_gurobi(self, model):
        return self.value

    def __repr__(self):
        return str(self.value)


class Variable(Expression):

    def __init__(self, layer, row, netPrefix, prefix_name='x', type='Real'):
        super(Variable, self).__init__(netPrefix, layer, row)
        self.prefix_name = prefix_name

        self.name = ''
        if not netPrefix == '':
            self.name += netPrefix + '_'

        self.name += prefix_name + '_' + str(layer) + '_' + str(row)
        self.type = type

        self.hasLo = False
        self.hasHi = False
        self.lo = -fc.default_bound
        self.hi = fc.default_bound

        self.has_grb_var = False
        self.grb_var = None

    def tighten_interval(self):
        pass

    def to_smtlib(self):
        return self.name

    def get_smtlib_decl(self):
        return '(declare-const ' + self.name + ' ' + self.type + ')'

    def get_smtlib_bounds(self):
        bounds = ''
        if self.hasHi:
            bounds += makeLeq(self.name, ffp(self.hi))
        if self.hasLo:
            bounds += '\n' + makeGeq(self.name, ffp(self.lo))

        return bounds

    def register_to_gurobi(self, model):
        lower = - grb.GRB.INFINITY
        upper = grb.GRB.INFINITY
        var_type = None
        if self.hasHi:
            upper = self.hi
        if self.hasLo:
            lower = self.lo
        # only types used are Int and Real
        # for Int only 0-1 are used -> Binary for gurobi
        if self.type == 'Int':
            var_type = grb.GRB.BINARY
        else:
            var_type = grb.GRB.CONTINUOUS

        self.grb_var = model.addVar(lb=lower, ub=upper, vtype=var_type, name=self.name)
        self.has_grb_var = True

    def to_gurobi(self, model):
        return self.grb_var

    def setLo(self, val):
        self.hasLo = True
        self.lo = val

    def setHi(self, val):
        self.hasHi = True
        self.hi = val

    def __repr__(self):
        return self.name


class Sum(Expression):

    def __init__(self, terms):
        net, layer, row = terms[0].getIndex()
        super(Sum, self).__init__(net, layer, row)
        self.children = terms
        self.lo = -fc.default_bound
        self.hi = fc.default_bound

    def tighten_interval(self):
        l = 0
        h = 0
        for term in self.children:
            term.tighten_interval()
            l += term.getLo()
            h += term.getHi()

        super(Sum, self).update_bounds(l, h)

    def to_smtlib(self):
        sum = '(+'
        for term in self.children:
            sum += ' ' + term.to_smtlib()

        sum += ')'
        return sum

    def to_gurobi(self, model):
        return grb.quicksum([t.to_gurobi(model) for t in self.children])

    def __repr__(self):
        sum = '(' + str(self.children[0])
        for term in self.children[1:]:
            sum += ' + ' + str(term)
        sum += ')'

        return sum


class Neg(Expression):

    def __init__(self, input):
        net, layer, row = input.getIndex()
        super(Neg, self).__init__(net, layer, row)
        self.input = input
        self.hasHi = input.hasLo
        self.hasLo = input.hasHi
        self.lo = -input.getHi()
        self.hi = -input.getLo()

    def tighten_interval(self):
        l = -self.input.getHi()
        h = -self.input.getLo()
        super(Neg, self).update_bounds(l, h)

    def to_smtlib(self):
        return '(- ' + self.input.to_smtlib() + ')'

    def to_gurobi(self, model):
        return -self.input.to_gurobi(model)

    def __repr__(self):
        return '(- ' + str(self.input) + ')'


class Multiplication(Expression):

    def __init__(self, constant, variable):
        net, layer, row = variable.getIndex()
        super(Multiplication, self).__init__(net, layer, row)
        self.constant = constant
        self.variable = variable
        self.lo = -fc.default_bound
        self.hi = fc.default_bound

    def tighten_interval(self):
        val1 = self.constant.value * self.variable.getLo()
        val2 = self.constant.value * self.variable.getHi()
        l = min(val1, val2)
        h = max(val1, val2)

        super(Multiplication, self).update_bounds(l, h)

    def to_smtlib(self):
        return '(* ' + self.constant.to_smtlib() + ' ' + self.variable.to_smtlib() + ')'

    def to_gurobi(self, model):
        if not self.variable.has_grb_var:
            raise ValueError('Variable {v} has not been registered to gurobi model!'.format(v=self.variable.name))

        return self.constant.to_gurobi(model) * self.variable.to_gurobi(model)

    def __repr__(self):
        return '(' + str(self.constant) + ' * ' + str(self.variable) + ')'


class Linear(Expression):

    def __init__(self, input, output):
        net, layer, row = output.getIndex()
        super(Linear, self).__init__(net, layer, row)
        self.output = output
        self.input = input
        self.lo = input.getLo()
        self.hi = input.getHi()

    def tighten_interval(self):
        self.input.tighten_interval()
        l = self.input.getLo()
        h = self.input.getHi()
        super(Linear, self).update_bounds(l, h)
        self.output.update_bounds(l, h)

    def to_smtlib(self):
        return makeEq(self.output.to_smtlib(), self.input.to_smtlib())

    def to_gurobi(self, model):
        return model.addConstr(self.output.to_gurobi(model) == self.input.to_gurobi(model))

    def __repr__(self):
        return '(' + str(self.output) + ' = ' + str(self.input) + ')'


class Relu(Expression):

    def __init__(self, input, output, delta):
        net, layer, row = output.getIndex()
        super(Relu, self).__init__(net, layer, row)
        self.output = output
        self.output.setLo(0)
        self.input = input
        self.lo = 0
        self.hi = fc.default_bound
        self.delta = delta
        self.delta.setLo(0)
        self.delta.setHi(1)

    def tighten_interval(self):
        self.input.tighten_interval()
        h = self.input.getHi()
        l = self.input.getLo()
        if h <= 0:
            # ReLU inactive delta=0
            self.hi = 0
            self.lo = 0
            self.output.update_bounds(0,0)
            self.delta.update_bounds(0,0)
        elif l > 0:
            # ReLU active delta=1
            super(Relu, self).update_bounds(l, h)
            self.output.update_bounds(l, h)
            self.delta.update_bounds(1,1)
        else:
            # don't know inactive/active
            super(Relu, self).update_bounds(l, h)
            self.output.update_bounds(l, h)

    def to_smtlib(self):
        # maybe better with asymmetric bounds
        m = max(abs(self.input.getLo()), abs(self.input.getHi()))

        dm = Multiplication(Constant(m, self.net, self.layer, self.row), self.delta)
        inOneMinusDM = Sum([self.input, Constant(m, self.net, self.layer, self.row), Neg(dm)])

        enc  = makeGeq(self.output.to_smtlib(), '0')
        enc += '\n' + makeGeq(self.output.to_smtlib(), self.input.to_smtlib())
        enc += '\n' + makeLeq(Sum([self.input, Neg(dm)]).to_smtlib(), '0')
        enc += '\n' + makeGeq(inOneMinusDM.to_smtlib(), '0')
        enc += '\n' + makeLeq(self.output.to_smtlib(), inOneMinusDM.to_smtlib())
        enc += '\n' + makeLeq(self.output.to_smtlib(), dm.to_smtlib())

        return enc

    def to_gurobi(self, model):
        c_name = 'ReLU_{n}_{layer}_{row}'.format(n=self.net, layer=self.layer, row=self.row)
        ret_constr = None

        if self.input.getLo() >= 0:
            # relu must be active
            ret_constr = model.addConstr(self.output.to_gurobi(model) == self.input.to_gurobi(model), name=c_name)
        elif self.input.getHi() <= 0:
            # relu must be inactive
            ret_constr = model.addConstr(self.output.to_gurobi(model) == 0, name=c_name)
        elif fc.use_grb_native:
            ret_constr = model.addConstr(self.output.to_gurobi(model) == grb.max_(self.input.to_gurobi(model), 0), name=c_name)
        elif fc.use_asymmetric_bounds:
            model.addConstr(self.output.to_gurobi(model) >= 0, name=c_name + '_a')
            model.addConstr(self.output.to_gurobi(model) >= self.input.to_gurobi(model), name=c_name + '_b')

            M_input = self.input.getHi()
            m_input = self.input.getLo()
            model.addConstr(self.input.to_gurobi(model) - M_input * self.delta.to_gurobi(model) <= 0, name=c_name + '_c')
            model.addConstr(self.input.to_gurobi(model) + (1 - self.delta.to_gurobi(model)) * -m_input
                            >= 0, name=c_name + '_d')

            M_active = max(abs(self.input.getLo()), abs(self.input.getHi()))
            model.addConstr(self.output.to_gurobi(model)
                            <= self.input.to_gurobi(model) + (1 - self.delta.to_gurobi(model)) * M_active, name=c_name + '_e')

            M_output = self.output.getHi()
            ret_constr = model.addConstr(self.output.to_gurobi(model) <= self.delta.to_gurobi(model) * M_output, name=c_name + '_f')
        else:
            bigM = max(abs(self.input.getLo()), abs(self.input.getHi()))
            model.addConstr(self.output.to_gurobi(model) >= 0, name=c_name + '_a')
            model.addConstr(self.output.to_gurobi(model) >= self.input.to_gurobi(model), name=c_name + '_b')
            model.addConstr(self.input.to_gurobi(model) - bigM * self.delta.to_gurobi(model) <= 0,
                            name=c_name + '_c')
            model.addConstr(self.input.to_gurobi(model) + (1 - self.delta.to_gurobi(model)) * bigM
                            >= 0, name=c_name + '_d')
            model.addConstr(self.output.to_gurobi(model)
                            <= self.input.to_gurobi(model) + (1 - self.delta.to_gurobi(model)) * bigM,
                            name=c_name + '_e')
            ret_constr = model.addConstr(self.output.to_gurobi(model) <= self.delta.to_gurobi(model) * bigM,
                                         name=c_name + '_f')
        return ret_constr


    def __repr__(self):
        return str(self.output) + ' =  ReLU(' + str(self.input) + ')'


class Max(Expression):

    def __init__(self, in_a, in_b, output, delta):
        net, layer, row = output.getIndex()
        super(Max, self).__init__(net, layer, row)
        self.output = output
        self.in_a = in_a
        self.in_b = in_b
        self.lo = -fc.default_bound
        self.hi = fc.default_bound
        self.delta = delta
        self.delta.setLo(0)
        self.delta.setHi(1)

    def tighten_interval(self):
        la = self.in_a.getLo()
        ha = self.in_a.getHi()
        lb = self.in_b.getLo()
        hb = self.in_b.getHi()

        if la > hb:
            # a is maximum
            self.output.update_bounds(la, ha)
            self.delta.update_bounds(0,0)
            super(Max, self).update_bounds(la, ha)
        elif lb > ha:
            # b is maximum
            self.output.update_bounds(lb, hb)
            self.delta.update_bounds(1,1)
            super(Max, self).update_bounds(lb, hb)
        else:
            # don't know which entry is max
            l = max(la, lb)
            h = max(ha, hb)
            self.output.update_bounds(l, h)
            super(Max, self).update_bounds(l, h)

    def to_smtlib(self):
        # maybe better with asymmetric bounds
        la = self.in_a.getLo()
        ha = self.in_a.getHi()
        lb = self.in_b.getLo()
        hb = self.in_b.getHi()
        m = max(abs(la), abs(ha), abs(lb), abs(hb))

        dm = Multiplication(Constant(m, self.net, self.layer, self.row), self.delta)
        in_bOneMinusDM = Sum([self.in_b, Constant(m, self.net, self.layer, self.row), Neg(dm)])

        enc  = makeGeq(self.output.to_smtlib(), self.in_a.to_smtlib())
        enc += '\n' + makeGeq(self.output.to_smtlib(), self.in_b.to_smtlib())
        enc += '\n' + makeLeq(self.output.to_smtlib(), Sum([self.in_a, dm]).to_smtlib())
        enc += '\n' + makeLeq(self.output.to_smtlib(), in_bOneMinusDM.to_smtlib())

        return enc

    def to_gurobi(self, model):
        return model.addConstr(self.output.to_gurobi(model) == grb.max_(self.in_a.to_gurobi(model), self.in_b.to_gurobi(model)))

    def __repr__(self):
        return str(self.output) +  ' = max(' + str(self.in_a) + ', ' + str(self.in_b) + ')'


class One_hot(Expression):
    # returns 1, iff input >= 0, 0 otherwise

    def __init__(self, input, output):
        net, layer, row = output.getIndex()
        super(One_hot, self).__init__(net, layer, row)
        self.output = output
        self.input = input
        self.output.setLo(0)
        self.output.setHi(1)
        self.lo = 0
        self.hi = 1

    def tighten_interval(self):
        self.input.tighten_interval()
        l_i = self.input.getLo()
        h_i = self.input.getHi()

        if l_i >= 0:
            self.output.update_bounds(1, 1)
            super(One_hot, self).update_bounds(1, 1)
        elif h_i < 0:
            self.output.update_bounds(0, 0)
            super(One_hot, self).update_bounds(0, 0)

    def to_smtlib(self):
        l_i = self.input.getLo()
        h_i = self.input.getHi_exclusive()

        h_i_out = Multiplication(Constant(h_i, self.net, self.layer, self.row), self.output)
        l_i_const = Constant(l_i, self.net, self.layer, self.row)
        l_i_out = Multiplication(l_i_const, self.output)

        enc = makeGt(h_i_out.to_smtlib(), self.input.to_smtlib())
        enc += '\n' + makeGeq(self.input.to_smtlib(), Sum([l_i_const, Neg(l_i_out)]).to_smtlib())

        return enc

    def to_gurobi(self, model):
        l_i = self.input.getLo()
        h_i = self.input.getHi_exclusive()

        c_name = 'OneHot_{layer}_{row}'.format(layer=self.layer, row=self.row)

        # convert to greater than
        # normal (hi * output) - eps >= ... doesn't work
        c1 = model.addConstr(h_i * (self.output.to_gurobi(model) - fc.epsilon) >= self.input.to_gurobi(model), name=c_name + '_a')
        c2 = model.addConstr(self.input.to_gurobi(model) >= (1 - self.output.to_gurobi(model)) * l_i, name=c_name + '_b')

        return c1, c2

    def __repr__(self):
        return str(self.output) + ' = OneHot(' + str(self.input) + ')'


class Greater_Zero(Expression):
    # returns 1, iff lhs > 0, 0 otherwise

    def __init__(self, lhs, delta):
        net, layer, row = delta.getIndex()
        super(Greater_Zero, self).__init__(net, layer, row)
        self.lhs = lhs
        self.delta = delta
        self.delta.setLo(0)
        self.delta.setHi(1)
        self.lo = 0
        self.hi = 1

    def tighten_interval(self):
        self.lhs.tighten_interval()
        l = self.lhs.getLo()
        h = self.lhs.getHi()

        if l > 0:
            self.delta.update_bounds(1, 1)
            super(Greater_Zero, self).update_bounds(1, 1)
        elif h <= 0:
            self.delta.update_bounds(0, 0)
            super(Greater_Zero, self).update_bounds(0, 0)

    def to_smtlib(self):
        l = self.lhs.getLo_exclusive()
        h = self.lhs.getHi()

        hd = Multiplication(Constant(h, self.net, self.layer, self.row), self.delta)
        l_const = Constant(l, self.net, self.layer, self.row)
        ld = Multiplication(l_const, self.delta)

        enc = makeLeq(self.lhs.to_smtlib(), hd.to_smtlib())
        enc += '\n' + makeGt(self.lhs.to_smtlib(), Sum([l_const, Neg(ld)]).to_smtlib())

        return enc

    def to_gurobi(self, model):
        l = self.lhs.getLo_exclusive()
        h = self.lhs.getHi()

        c_name = 'Gt0_{layer}_{row}'.format(layer=self.layer, row=self.row)

        c1 = model.addConstr(self.lhs.to_gurobi(model) <= h * self.delta.to_gurobi(model), name=c_name + '_a')
        # convert to greater than
        # with epsilon otherwise, when lhs == 0, delta == 1 would also be ok, with epsilon forced to take 0
        c2 = model.addConstr(self.lhs.to_gurobi(model) - fc.epsilon >= (1 - self.delta.to_gurobi(model)) * l, name=c_name + '_b')

        return c1, c2

    def __repr__(self):
        return str(self.lhs) + ' > 0 <==> ' + str(self.delta) + ' = 1'


class Gt_Int(Expression):

    def __init__(self, lhs, rhs, delta):
        net, layer, row = lhs.getIndex()
        super(Gt_Int, self).__init__(net, layer, row)
        self.lhs = lhs
        self.rhs = rhs
        self.delta = delta
        self.delta.setLo(0)
        self.delta.setHi(1)
        self.lo = 0
        self.hi = 1

    def tighten_interval(self):
        self.lhs.tighten_interval()
        self.rhs.tighten_interval()
        ll = self.lhs.getLo()
        hl = self.lhs.getHi()
        lr = self.rhs.getLo()
        hr = self.rhs.getHi()

        # can't set bounds like this, because upper and lower bounds of rhs/lhs are possible
        # because of OR in equiv constraint
        # TODO: tighten bounds depending on bounds of delta
        #if hr > hl:
        #    self.rhs.update_bounds(hr, hl)

        #if lr > ll:
        #    self.lhs.update_bounds(lr, hl)

        if ll > hr:
            self.delta.update_bounds(1, 1)
            super(Gt_Int, self).update_bounds(1, 1)
        elif hl <= lr:
            self.delta.update_bounds(0, 0)
            super(Gt_Int, self).update_bounds(0, 0)

    def to_smtlib(self):
        one = Constant(1, self.net, self.layer, self.row)

        constr1 = Sum([Neg(self.lhs), self.rhs, one])
        bigM1 = Constant(constr1.getHi(), self.net, self.layer, self.row)

        constr2 = Sum([self.lhs, Neg(self.rhs)])
        bigM2 = Constant(constr2.getHi(), self.net, self.layer, self.row)

        # delta == 1 -->  lhs - rhs > 0 (>= 1)
        enc = makeLeq(constr1.to_smtlib(), Sum([bigM1, Neg(Multiplication(bigM1, self.delta))]).to_smtlib())
        # delta == 0 --> lhs - rhs <= 0
        enc += '\n' + makeLeq(constr2.to_smtlib(), Multiplication(bigM2, self.delta).to_smtlib())

        return enc

    def to_gurobi(self, model):
        c_name = 'Gt_Int_{layer}_{row}'.format(layer=self.layer, row=self.row)

        if fc.use_grb_native:
            c1 = model.addConstr((self.delta.to_gurobi(model) == 1)
                                 >> (self.lhs.to_gurobi(model) - self.rhs.to_gurobi(model) >= 1), name=c_name + '_a')
            c2 = model.addConstr((self.delta.to_gurobi(model) == 0)
                                 >> (self.lhs.to_gurobi(model) - self.rhs.to_gurobi(model) <= 0), name=c_name + '_b')
        else:
            one = Constant(1, self.net, self.layer, self.row)
            constr1 = Sum([Neg(self.lhs), self.rhs, one])
            constr2 = Sum([self.lhs, Neg(self.rhs)])

            bigM1 = constr1.getHi()
            bigM2 = constr2.getHi()

            c1 = model.addConstr(constr1.to_gurobi(model) <= bigM1 * (1 - self.delta.to_gurobi(model)), name=c_name + '_a')
            c2 = model.addConstr(constr2.to_gurobi(model) <= bigM2 * self.delta.to_gurobi(model), name=c_name + '_b')

        return c1, c2

    def __repr__(self):
        return str(self.lhs) + ' > ' + str(self.rhs) + ' <==> ' + str(self.delta) + ' = 1'


class Geq(Expression):
    # TODO: no return value as no real expression, just a constraint (better idea where to put it?)
    # could return 0/1 but would need more complicated delta stmt instead of just proxy for printing geq

    def __init__(self, lhs, rhs):
        net, layer, row = lhs.getIndex()
        super(Geq, self).__init__(net, layer, row)
        self.lhs = lhs
        self.rhs = rhs
        self.lo = 0
        self.hi = 1

    def tighten_interval(self):
        self.lhs.tighten_interval()
        self.rhs.tighten_interval()
        llhs = self.lhs.getLo()
        hlhs = self.lhs.getHi()
        lrhs = self.rhs.getLo()
        hrhs = self.rhs.getHi()

        if hrhs > hlhs:
            self.rhs.update_bounds(hrhs, hlhs)

        if lrhs > llhs:
            self.lhs.update_bounds(lrhs, hlhs)

        if llhs >= hrhs:
            super(Geq, self).update_bounds(1, 1)
        elif hlhs < lrhs:
            super(Geq, self).update_bounds(0, 0)

    def to_smtlib(self):
        return makeGeq(self.lhs.to_smtlib(), self.rhs.to_smtlib())

    def to_gurobi(self, model):
        return model.addConstr(self.lhs.to_gurobi(model) >= self.rhs.to_gurobi(model))

    def __repr__(self):
        return str(self.lhs) + ' >= ' + str(self.rhs)


class BinMult(Expression):
    # multiplication of a binary variable and another expression
    # can be linearized and expressed by this expression

    def __init__(self, binvar, factor, result_var):
        net, layer, row = result_var.getIndex()
        super(BinMult, self).__init__(net, layer, row)
        self.binvar = binvar
        self.binvar.setLo(0)
        self.binvar.setHi(1)
        self.factor = factor
        self.result_var = result_var
        self.lo = -fc.default_bound
        self.hi = fc.default_bound

    def tighten_interval(self):
        self.factor.tighten_interval()
        fl = self.factor.getLo()
        fh = self.factor.getHi()

        # 0 <= bl <= bh <= 1
        bl = self.binvar.getLo()
        bh = self.binvar.getHi()

        yl = min(bl * fl, bl * fh, bh * fl, bh * fh)
        yh = max(bl * fl, bl * fh, bh * fl, bh * fh)

        self.result_var.update_bounds(yl, yh)
        super(BinMult, self).update_bounds(yl, yh)

    def to_smtlib(self):
        bigM = Constant(self.factor.getHi(), self.net, self.layer, self.row)
        bigMbinvar = Multiplication(bigM, self.binvar)

        enc = makeLeq(self.result_var.to_smtlib(), bigMbinvar.to_smtlib())
        enc += '\n' + makeLeq(self.result_var.to_smtlib(), self.factor.to_smtlib())
        enc += '\n' + makeLeq(Sum([self.factor, Neg(self.result_var)]).to_smtlib(), Sum([bigM, Neg(bigMbinvar)]).to_smtlib())

        return enc

    def to_gurobi(self, model):
        if not self.binvar.has_grb_var:
            raise ValueError('Variable {v} has not been registered to gurobi model!'.format(v=self.binvar.name))

        c_name = 'BinMult_{net}_{layer}_{row}'.format(net=self.net, layer=self.layer, row=self.row)

        ret_constr = None

        if fc.use_grb_native:
            model.addConstr((self.binvar.to_gurobi(model) == 0) >> (self.result_var.to_gurobi(model) == 0), name=c_name + '_1')
            ret_constr = model.addConstr((self.binvar.to_gurobi(model) == 1)
                                         >> (self.result_var.to_gurobi(model) == self.factor.to_gurobi(model)), name=c_name + '_2')
        else:
            M_res = self.result_var.getHi()
            m_res = self.result_var.getLo()
            model.addConstr(self.result_var.to_gurobi(model) <= M_res * self.binvar.to_gurobi(model),
                            name=c_name + '_y<=0')
            model.addConstr(self.result_var.to_gurobi(model) >= m_res * self.binvar.to_gurobi(model),
                            name=c_name + '_y>=0')

            # upper and lower bounds of res_var - factor
            M = self.result_var.getHi() - self.factor.getLo()
            m = self.result_var.getLo() - self.factor.getHi()
            model.addConstr(self.result_var.to_gurobi(model) - self.factor.to_gurobi(model)
                            <= M * (1 - self.binvar.to_gurobi(model)), name=c_name + '_y<=x')
            ret_constr = model.addConstr(self.result_var.to_gurobi(model) - self.factor.to_gurobi(model)
                                         >= m * (1 - self.binvar.to_gurobi(model)), name=c_name + '_y>=x')

        # return last added constraint, don't know what to return instead and all other to_gurobis return a constraint
        return ret_constr

    def __repr__(self):
        return str(self.result_var) + ' = BinMult(' + str(self.binvar) + ', ' + str(self.factor) + ')'


class Impl(Expression):
    # Implication: delta = c --> lhs <= rhs , for binary constant c

    def __init__(self, delta, constant, lhs, rhs):
        net, layer, row = delta.getIndex()
        super(Impl, self).__init__(net, layer, row)
        self.delta = delta
        self.delta.setLo(0)
        self.delta.setHi(1)
        self.constant = constant
        self.lhs = lhs
        self.rhs = rhs
        self.lo = 0
        self.hi = 1

    def tighten_interval(self):
        self.lhs.tighten_interval()
        self.rhs.tighten_interval()

        # consequence of implication has to be true
        if (self.delta.getLo() == 1 and self.constant == 1) or (self.delta.getHi() == 0 and self.constant == 0):
            self.lhs.update_bounds(self.lhs.getLo(), self.rhs.getHi())
            self.rhs.update_bounds(self.lhs.getLo(), self.rhs.getHi())

        # antecedent has to be false
        if self.rhs.getHi() < self.lhs.getLo():
            self.delta.update_bounds(1 - self.constant, 1 - self.constant)
            self.update_bounds(1 - self.constant, 1 - self.constant)

    def to_smtlib(self):
        term = Sum([self.lhs, Neg(self.rhs)])
        bigM = Constant(term.getHi(), self.net, self.layer, self.row)

        if self.constant == 0:
            enc = makeLeq(term.to_smtlib(), Multiplication(bigM, self.delta).to_smtlib())
        else:
            enc = makeLeq(term.to_smtlib(), Sum([bigM, Neg(Multiplication(bigM, self.delta))]).to_smtlib())

        return enc

    def to_gurobi(self, model):
        if not self.delta.has_grb_var:
            raise ValueError('Variable {v} has not been registered to gurobi model!'.format(v=self.delta.name))

        c_name = 'Impl_{net}_{layer}_{row}'.format(net=self.net, layer=self.layer, row=self.row)

        ret_constr = None

        if fc.use_grb_native:
            ret_constr = model.addConstr((self.delta.to_gurobi(model) == self.constant)
                                         >> (self.lhs.to_gurobi(model) <= self.rhs.to_gurobi(model)), name=c_name)
        else:
            term = Sum([self.lhs, Neg(self.rhs)])
            term.tighten_interval()
            bigM = term.getHi()

            if self.constant == 0:
                ret_constr = model.addConstr(term.to_gurobi(model) <= bigM * self.delta.to_gurobi(model), name=c_name)
            else:
                ret_constr = model.addConstr(term.to_gurobi(model) <= bigM * (1 - self.delta.to_gurobi(model)), name=c_name)

        return ret_constr

    def __repr__(self):
        return '{d} = {c} --> {left} <= {right}'.format(d=str(self.delta),
                                                        c=str(self.constant), left=str(self.lhs), right=str(self.rhs))


class IndicatorToggle(Expression):
    # sets diff_i = term_i, if indicator = constant, otherwise diff_i <= min(x_is)

    def __init__(self, indicators, constant, terms, diffs):
        net, layer, row = indicators[0].getIndex()
        super(IndicatorToggle, self).__init__(net, layer, row)
        self.indicators = indicators
        for indicator in self.indicators:
            indicator.setLo(0)
            indicator.setHi(1)
        self.constant = constant
        self.terms = terms
        self.diffs = diffs
        self.lo = -fc.default_bound
        self.hi = fc.default_bound
        self.terms_lo = -fc.default_bound

    def tighten_interval(self):
        terms_lo_new = fc.default_bound
        max_hi = -fc.default_bound
        max_lo = fc.default_bound
        # TODO: take indicators into account
        for t, d in zip(self.terms, self.diffs):
            t.tighten_interval()
            tlo = t.getLo()
            thi = t.getHi()
            d.update_bounds(-fc.default_bound, thi)
            # was intended to calc max directly, but put got other idea
            # self.update_bounds(max_lo, max_hi)
            if tlo < terms_lo_new:
                terms_lo_new = tlo

        if terms_lo_new > self.terms_lo:
            self.terms_lo = terms_lo_new

        for d in self.diffs:
            d.update_bounds(self.terms_lo, fc.default_bound)

    def to_smtlib(self):
        enc = []
        bigL = Constant(self.terms_lo, self.net, self.layer, self.row)
        for t, ind, diff in zip(self.terms, self.indicators, self.diffs):
            enc_i = Impl(ind, self.constant, t, diff).to_smtlib()
            enc_i += '\n' + Impl(ind, self.constant, diff, t).to_smtlib()
            enc_i += '\n' + Impl(ind, 1 - self.constant, bigL, diff).to_smtlib()
            enc_i += '\n' + Impl(ind, 1 - self.constant, diff, bigL).to_smtlib()
            enc.append(enc_i)

        return '\n'.join(enc)

    def to_gurobi(self, model):
        ret_constr = None

        bigL = Constant(self.terms_lo, self.net, self.layer, self.row)
        for t, ind, diff in zip(self.terms, self.indicators, self.diffs):
            Impl(ind, self.constant, t, diff).to_gurobi(model)
            Impl(ind, self.constant, diff, t).to_gurobi(model)
            # somehow Impl(..., bigL, diff) yields invalid sense for indicator constraint
            Impl(ind, (1 - self.constant), Neg(diff), Neg(bigL)).to_gurobi(model)
            ret_constr = Impl(ind, 1 - self.constant, diff, bigL).to_gurobi(model)

        return ret_constr

    def __repr__(self):
        reps = []
        for t, ind, diff in zip(self.terms, self.indicators, self.diffs):
            reps.append('{d} = {c} --> {left} = {right}'.format(d=str(ind), c=str(self.constant), left=str(diff),
                                                                right=str(t)))
            reps.append('{d} = {c} --> {left} = {right}'.format(d=str(ind), c=str(1 - self.constant), left=str(diff),
                                                                right=str(self.terms_lo)))
        return '\n'.join(reps)


class TopKGroup(Expression):
    # performs bounds tightening, s.t. bounds for element are updated to the bounds of the top-k element

    def __init__(self, out, ins, k):
        '''
        Initializes TopK context group. After call to tighten_interval(), the output element's bounds are
        tightened to the k-greatest upper bound and the k-greatest lower bound (with k starting from 1 for the
        greatest element)
        :param out: the output element
        :param ins: list of input elements
        :param k: 1 - for the greatest element, ... k - for the k-greatest element
        '''
        net, layer, row = out.getIndex()
        super(TopKGroup, self).__init__(net, layer, row)
        self.ins = ins
        self.k = k
        self.out = out
        self.out.update_bounds(-fc.default_bound, fc.default_bound)
        self.lo = self.out.getLo()
        self.hi = self.out.getHi()

    def tighten_interval(self):
        for i in self.ins:
            i.tighten_interval()
        self.out.tighten_interval()

        in_lo = sorted(self.ins, key=lambda x: x.getLo())[-self.k].getLo()
        in_hi = sorted(self.ins, key=lambda x: x.getHi())[-self.k].getHi()
        self.out.update_bounds(in_lo, in_hi)
        self.update_bounds(in_lo, in_hi)

    def to_smtlib(self):
        return ''

    def to_gurobi(self, model):
        return None

    def __repr__(self):
        in_rep = []
        for i in self.ins:
            in_rep.append(str(i))

        return '{o} in top_{k}({irs})'.format(o=str(self.out), k=self.k, irs=', '.join(in_rep))


class ExtremeGroup(Expression):
    # performs bounds tightening, s.t. bounds for element are updated to the most extreme bounds in the input values

    def __init__(self, out, ins):
        '''
        Initializes Extreme context group. After call to tighten_interval(), the output element's bounds are
        tightened to the greatest upper bound and the smallest lower bound amongst all input elements
        :param out: the output element
        :param ins: list of input elements
        '''
        net, layer, row = out.getIndex()
        super(ExtremeGroup, self).__init__(net, layer, row)
        self.ins = ins
        self.out = out
        self.out.update_bounds(-fc.default_bound, fc.default_bound)
        self.lo = self.out.getLo()
        self.hi = self.out.getHi()

    def tighten_interval(self):
        for i in self.ins:
            i.tighten_interval()
        self.out.tighten_interval()

        in_lo = sorted(self.ins, key=lambda x: x.getLo())[0].getLo()
        in_hi = sorted(self.ins, key=lambda x: x.getHi())[-1].getHi()
        self.out.update_bounds(in_lo, in_hi)
        self.update_bounds(in_lo, in_hi)

    def to_smtlib(self):
        return ''

    def to_gurobi(self, model):
        return None

    def __repr__(self):
        in_rep = []
        for i in self.ins:
            in_rep.append(str(i))

        return '{o} in extreme({irs})'.format(o=str(self.out), irs=', '.join(in_rep))