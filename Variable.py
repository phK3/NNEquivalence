
class Variable:

    def __init__(self,layer, row, prefix_name='x'):
        self.layer = layer
        self.row = row
        self.prefix_name = prefix_name
        self.name = prefix_name + '_' + str(layer) + '_' + str(row)
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

