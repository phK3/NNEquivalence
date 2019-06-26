
class Variable:

    #types are Real and Int just as in SMTLib
    def __init__(self,layer, row, prefix_name='x', type='Real'):
        self.layer = layer
        self.row = row
        self.prefix_name = prefix_name
        self.name = prefix_name + '_' + str(layer) + '_' + str(row)
        self.type = type
        # lower and upper bounds
        self.hasLo = False
        self.hasHi = False
        self.lo = 0
        self.hi = 0

    def setLo(self, lowerBound):
        self.lo = lowerBound
        self.hasLo = True
        return self.lo

    def setHi(self, upperBound):
        self.hi = upperBound
        self.hasHi = True
        return self.hi

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

