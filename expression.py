from abc import ABC, abstractmethod
from numpy import format_float_positional as ffp
import numbers

default_bound = 999999


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

    @abstractmethod
    def getHi(self):
        pass

    @abstractmethod
    def getLo(self):
        pass

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
        self.hasLo = True
        self.hasHi = True

    def getHi(self):
        return self.value

    def getLo(self):
        return self.value

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

    def getLo(self):
        return self.lo

    def getHi(self):
        return self.hi

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
        print('sum.tighten_interval called')
        l = 0
        h = 0
        for term in self.children:
            l += term.getLo()
            h += term.getHi()

        # TODO: find bug, that makes l, h so big
        print('[l, h] = ' + str(l) + ', ' + str(h))

        super(Sum, self).update_bounds(l, h)

    def getLo(self):
        return self.lo

    def getHi(self):
        return self.hi

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
        self.hasHi = input.hasHi
        self.hasLo = input.hasLo
        self.lo = -input.getHi()
        self.hi = -input.getLo()

    def tighten_interval(self):
        l = -self.input.getHi()
        h = -self.input.getLo()
        super(Neg, self).update_bounds(l, h)

    def getHi(self):
        return self.hi

    def getLo(self):
        return self.lo

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

    def getHi(self):
        return self.hi

    def getLo(self):
        return self.lo

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
        print('Linear.tighten_interval called!!!')
        self.input.tighten_interval()
        l = self.input.getLo()
        h = self.input.getHi()
        super(Linear, self).update_bounds(l, h)
        self.output.update_bounds(l, h)

    def getLo(self):
        return self.lo

    def getHi(self):
        return self.hi

    def to_smtlib(self):
        return '(= ' + self.output.to_smtlib() + ' ' + self.input.to_smtlib() + ')'

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
        h = self.input.getHi()
        l = self.input.getLo()
        if h <= 0:
            # ReLU inactive delta=0
            self.hi = 0
            self.lo = 0
            self.delta.update_bounds(0,0)
        elif l > 0:
            # ReLU active delta=1
            if l > self.lo: self.lo = l
            if h < self.hi: self.hi = h
            self.delta.update_bounds(1,1)
        else:
            # don't know inactive/active
            if l > self.lo: self.lo = l
            if h < self.hi: self.hi = h

    def getLo(self):
        return self.lo

    def getHi(self):
        return self.hi

    def to_smtlib(self):
        # maybe better with asymmetric bounds
        m = max(abs(self.input.getLo), abs(self.input.getHi))

        dm = Multiplication(Constant(m, self.net, self.layer, self.row), self.delta)
        inOneMinusDM = Sum([input, Constant(m, self.net, self.layer, self.row), Neg(dm)])

        enc  = makeGeq(self.output.to_smtlib, '0')
        enc += '\n' + makeGeq(self.output.to_smtlib, self.input.to_smtlib())
        enc += '\n' + makeLeq(Sum([self.input, Neg(dm)]), '0')
        enc += '\n' + makeGeq(inOneMinusDM, '0')
        enc += '\n' + makeLeq(self.output.to_smtlib(), inOneMinusDM)
        enc += '\n' + makeLeq(self.output.to_smtlib(), dm)

        return enc

    def __repr__(self):
        return str(self.output) + ' =  ReLU(' + str(self.input) + ')'


def encode_inputs(lower_bounds, upper_bounds, netPrefix=''):
    vars = []
    for i, (l, h) in enumerate(zip(lower_bounds, upper_bounds)):
        input_var = Variable(0, i, netPrefix, 'i')
        input_var.setLo(l)
        input_var.setHi(h)
        vars.append(input_var)

    return vars

def encode_linear_layer(prev_neurons, weights, numNeurons, layerIndex, netPrefix):
    vars = []
    equations = []
    prev_num = len(prev_neurons)
    for i in range(0, numNeurons):
        var = Variable(layerIndex, i, netPrefix, 'x')
        vars.append(var)
        terms = [Multiplication(Constant(weights[row][i], netPrefix, layerIndex, row), prev_neurons[row]) for row in range(0, prev_num)]
        terms.append(Constant(weights[-1][i], netPrefix, layerIndex, prev_num))
        equations.append(Linear(Sum(terms), var))

    return vars, equations

def encode_relu_layer(prev_neurons, layerIndex, netPrefix):
    deltas = []
    outs = []
    ineqs = []
    for i, neuron in enumerate(prev_neurons):
        output = Variable(layerIndex, i, netPrefix, 'o')
        delta = Variable(layerIndex, i, netPrefix, 'd', 'Int')
        outs.append(output)
        deltas.append(delta)
        ineqs.append(Relu(neuron, output, delta))

    return outs, deltas, ineqs


def encodeExample():
    invars = encode_inputs([0,1,2], [1,2,3])

    weights = [[1,5],[2,6],[3,7],[4,8]]
    linvars, eqs = encode_linear_layer(invars, weights, 2, 1, '')

    for eq in eqs:
        eq.tighten_interval()

    reluouts, reludeltas, ineqs = encode_relu_layer(linvars, 2, '')




    print('### invars ###')
    for i in invars:
        print(str(i) + ': [' + str(i.getLo()) + ', ' + str(i.getHi()) + ']')

    print('### linears ###')
    for var, eq in zip(linvars, eqs):
        print(eq)
        print(str(var) + ': [' + str(var.getLo()) + ', ' + str(var.getHi()) + ']')

    print('### relus ###')
    for ineq in ineqs:
        print(ineq)
    for out, delta in zip(reluouts, reludeltas):
        print(str(out) + ': [' + str(out.getLo()) + ', ' + str(out.getHi()) + ']')
        print(str(delta) + ': [' + str(delta.getLo()) + ', ' + str(delta.getHi()) + ']')



