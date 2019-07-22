from abc import ABC, abstractmethod
from numpy import format_float_positional as ffp
import numbers

default_bound = 999999
epsilon = 1e-8

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
        self.lo = -default_bound
        self.hi = default_bound
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

    def getHi(self):
        return self.hi

    def getLo(self):
        return self.lo

    def getLo_exclusive(self):
        return self.lo - epsilon

    def getHi_exclusive(self):
        return self.hi + epsilon

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
            s = str(self.value)
        else:
            s = ffp(self.value)

        return s

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
        self.lo = -default_bound
        self.hi = default_bound

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
        self.lo = -default_bound
        self.hi = default_bound

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

    def __repr__(self):
        return '(- ' + str(self.input) + ')'


class Multiplication(Expression):

    def __init__(self, constant, variable):
        net, layer, row = variable.getIndex()
        super(Multiplication, self).__init__(net, layer, row)
        self.constant = constant
        self.variable = variable
        self.lo = -default_bound
        self.hi = default_bound

    def tighten_interval(self):
        val1 = self.constant.value * self.variable.getLo()
        val2 = self.constant.value * self.variable.getHi()
        l = min(val1, val2)
        h = max(val1, val2)

        super(Multiplication, self).update_bounds(l, h)

    def to_smtlib(self):
        return '(* ' + self.constant.to_smtlib() + ' ' + self.variable.to_smtlib() + ')'

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

    def __repr__(self):
        return '(' + str(self.output) + ' = ' + str(self.input) + ')'


class Relu(Expression):

    def __init__(self, input, output, delta):
        net, layer, row = output.getIndex()
        super(Relu, self).__init__(net, layer, row)
        self.output = output
        self.input = input
        self.lo = 0
        self.hi = default_bound
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

    def __repr__(self):
        return str(self.output) + ' =  ReLU(' + str(self.input) + ')'


class Max(Expression):

    def __init__(self, in_a, in_b, output, delta):
        net, layer, row = output.getIndex()
        super(Max, self).__init__(net, layer, row)
        self.output = output
        self.in_a = in_a
        self.in_b = in_b
        self.lo = -default_bound
        self.hi = default_bound
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
        l_i_out = Multiplication(Constant(l_i, self.net, self.layer, self.row), self.output)

        enc = makeGt(h_i_out.to_smtlib(), self.input.to_smtlib())
        enc += '\n' + makeGeq(self.input.to_smtlib(), Sum([self.input, Neg(l_i_out)]).to_smtlib())

        return enc

    def __repr__(self):
        return str(self.output) + ' = OneHot(' + str(self.input) + ')'


class Greater_Zero(Expression):
    # returns 1, iff input >= 0, 0 otherwise

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

    def getLo(self):
        return self.lo

    def getHi(self):
        return self.hi

    def to_smtlib(self):
        l = self.lhs.getLo_exclusive()
        h = self.lhs.getHi()

        hd = Multiplication(Constant(h, self.net, self.layer, self.row), self.delta)
        l_const = Constant(l, self.net, self.layer, self.row)
        ld = Multiplication(l_const, self.delta)

        enc = makeLt(self.lhs.to_smtlib(), hd.to_smtlib())
        enc += '\n' + makeGt(self.lhs.to_smtlib(), Sum([l_const, Neg(ld)]).to_smtlib())

        return enc

    def __repr__(self):
        return str(self.lhs) + ' > 0'

