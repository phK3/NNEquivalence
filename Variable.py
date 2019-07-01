
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
        # set to +/- 99999 as default
        # TODO: change them later to useful values
        self.lo = -99999
        self.hi = 99999

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

