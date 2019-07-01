
import numpy as np
from Variable import *
import NNFileAccess

# flatten list of lists
def flatten(list):
    return [x for sublist in list for x in sublist]

class NNEncoder:

    def __init__(self, file, freeVar):
        self.file = file
        #list of list of vars
        #last list always contains output of last layer
        self.vars = []


    #only use strings in args for make... methods
    def makeMult(self, constStr, varName):
        return '(* ' + constStr + ' ' + varName + ')'

    def makeSum(self, terms):
        sum = '(+'
        for term in terms:
            sum += ' ' + term

        sum += ')'
        return sum

    def makeNeg(self, term):
        return '(- ' + term + ')'

    def makeLeq(self, lhs, rhs):
        return '(assert (<= ' + lhs + ' ' + rhs + '))'

    def makeGeq(self, lhs, rhs):
        #maybe switch to other representation later
        return self.makeLeq(rhs, lhs)

    def makeEq(self, lhs, rhs):
        return '(assert (= ' + lhs + ' ' + rhs + '))'

    def makeLt(self, lhs, rhs):
        return '(assert (< ' + lhs + ' ' + rhs + '))'

    def makeGt(self, lhs, rhs):
        return self.makeLt(rhs, lhs)


    def encodeInputsReadable(self, lowerBounds, upperBounds):
        if not (len(lowerBounds) == len(upperBounds)):
            raise IOError('lowerBounds and upperBounds need to match the number of inputs')

        #no constraints printed here. Later for all vars lower and upper bound constraints are added
        inputVars = []
        i = 0
        for lo, hi in zip(lowerBounds, upperBounds):
            var = Variable(0, i, 'i')
            var.setLo(lo)
            var.setHi(hi)
            inputVars.append(var)
            i += 1

        self.vars.append(inputVars)


    # weights is matrix where one column holds the weights for a neuron.
    # Bias for that neuron is the last entry in that column
    # numNeurons is the number of neurons in that layer
    def encodeLinearLayer(self, weights, numNeurons, layerIndex):
        enc = '# --- linear constraints layer ' + str(layerIndex) + ' ---'
        prevNeurons = self.vars[-1]
        prevNum = len(prevNeurons)
        currentNeurons = []
        for i in range(0, numNeurons):
            var = Variable(layerIndex, i, 'x')
            currentNeurons.append(var)
            terms = [self.makeMult(str(weights[row][i]), prevNeurons[row].name) for row in range(0, prevNum)]
            terms.append(str(weights[-1][i]))
            enc += '\n' + self.makeEq(var.name, self.makeSum(terms))

        self.vars.append(currentNeurons)

        return enc

    # maxpool function, takes list of inputNeurons and output neuron.
    # can't be used as activationEncoder here, because list of input neurons
    # returns (enc, intermediateVars)
    def encodeMaxPoolReadable(self, inNeurons, outNeuron):
        num = len(inNeurons)
        enc = ''
        vars = []

        # TODO: think of other ways to name the vars uniquely (too long names)
        if num == 1:
            enc = self.makeEq(inNeurons[0].name, outNeuron.name)
            return (enc, vars)

        if num == 2:
            maxVarA = inNeurons[0]
            maxVarB = inNeurons[1]

        if num > 2:
            maxVarA = Variable(outNeuron.layer, outNeuron.row, outNeuron.name + 'a')
            maxVarB = Variable(outNeuron.layer, outNeuron.row, outNeuron.name + 'b')
            enc1, vars1 = self.encodeMaxPoolReadable(inNeurons[:num//2], maxVarA)
            enc2, vars2 = self.encodeMaxPoolReadable(inNeurons[num//2:], maxVarB)

            enc += enc1 + '\n' + enc2
            vars.append(vars1)
            vars.append(vars2)

        delta = Variable(outNeuron.layer, outNeuron.row, outNeuron.name + 'd', 'Int')
        delta.setLo(0)
        delta.setHi(1)
        vars.append(delta)

        m = 99999

        md = self.makeMult(str(m), delta.name)
        enc += '\n' + self.makeGeq(outNeuron.name, maxVarA.name)
        enc += '\n' + self.makeGeq(outNeuron.name, maxVarB.name)
        enc += '\n' + self.makeLeq(outNeuron.name, self.makeSum([maxVarA.name, md]))
        enc += '\n' + self.makeLeq(outNeuron.name, self.makeSum([maxVarB.name, str(m), self.makeNeg(md)]))

        return (enc, vars)


    # encoding of ReLU function,
    # takes input neuron (result of weighted summation) and output neuron
    # returns encoding for ReLU for this output neuron
    # and list of intermediate variables generated
    def encodeRelu(self, inNeuron, outNeuron):
        enc = ''
        layerIndex = inNeuron.layer
        rowIndex = inNeuron.row

        delta = Variable(layerIndex, rowIndex, 'd', 'Int')
        delta.setLo(0)
        delta.setHi(1)

        # later use bound on sn for m
        m = 99999
        dm = self.makeMult(str(m), delta.name)

        enc += self.makeGeq(outNeuron.name, '0')
        enc += '\n' + self.makeGeq(outNeuron.name, inNeuron.name)
        enc += '\n' + self.makeLeq(self.makeSum([inNeuron.name, self.makeNeg(dm)]), '0')
        enc += '\n' + self.makeGeq(self.makeSum([inNeuron.name, str(m), self.makeNeg(dm)]), '0')
        enc += '\n' + self.makeLeq(outNeuron.name, self.makeSum([inNeuron.name, str(m), self.makeNeg(dm)]))
        enc += '\n' + self.makeLeq(outNeuron.name, dm)

        return (enc, [delta])


    # encodes a layer with an activation function
    # takes an activationEncoder method as argument
    # this method has to have signature:
    # (Encoding, IntermediateVars) activationEncoder(inputNeuron, outputNeuron)
    def encodeActivationLayer(self, numNeurons, layerIndex, activatonEncoder):
        enc = '# --- activation constrainst layer ' + str(layerIndex) + ' ---'
        sumNeurons = self.vars[-1]
        deltas = []
        outNeurons = []
        for i in range(0, numNeurons):
            sn = sumNeurons[i]
            out = Variable(layerIndex, i, 'o')
            outNeurons.append(out)

            activatonEncoded, intermediateVars = activatonEncoder(sn, out)
            enc += '\n' + activatonEncoded
            deltas.append(intermediateVars)

        self.vars.append(flatten(deltas))
        self.vars.append(outNeurons)

        return enc

    def makePreambleReadable(self):
        preamble = '(set-option :produce-models true)\n(set-logic AUFLIRA)'
        decls = ''
        bounds = ''
        for list in self.vars:
            for var in list:
                decls += '\n' + '(declare-const ' + var.name + ' ' + var.type +')'
                if var.hasHi:
                    bounds += '\n' + self.makeLeq(var.name, str(var.hi))
                if var.hasLo:
                    bounds += '\n' + self.makeGeq(var.name, str(var.lo))

        return preamble + decls + '\n# ---- Bounds ----' + bounds

    def makeSuffix(self):
        return '(check-sat)\n(get-model)'

    def encodeNN(self, numLayers, numInputs, layers, input_lowerBounds, input_upperBounds, mode='strict'):
        inputEnc = self.encodeInputs(numInputs, input_lowerBounds, input_upperBounds)

        index = 0
        numNeuronsPrev = numInputs
        layersEnc = ''
        for activation, numNeurons, weights in layers:
            index += 1
            layersEnc += self.encodeLayer(index, numNeurons, numNeuronsPrev, weights, activation, mode) + '\n'
            numNeuronsPrev = numNeurons

        varDecls = ''
        varsUsed = self.getNewVar()
        for i in range(0, varsUsed):
            if i in self.intVars:
                varDecls += '(declare-const x' + str(i) + ' Int)\n'
            else:
                varDecls += '(declare-const x' + str(i) + ' Real)\n'

        return varDecls + '\n' + inputEnc + '\n' + layersEnc


    def encodeOneHotLayerReadable(self, layerIndex):
        inNeurons = self.vars[-1]
        maxNeuron = Variable(layerIndex, 0, 'max')
        maxEnc, maxVars = self.encodeMaxPoolReadable(inNeurons, maxNeuron)

        outNeurons = []
        diffNeurons = []
        diffConstraints = ''
        enc = ''
        for i in range(0, len(inNeurons)):
            out = Variable(layerIndex + 1, i, 'o', 'Int')
            out.setLo(0)
            out.setHi(1)
            outNeurons.append(out)

            inNeuron = inNeurons[i]

            diff = Variable(layerIndex + 1, i, 'x')
            diffNeurons.append(diff)
            diffConstraints += '\n' + self.makeEq(diff.name, self.makeSum([inNeuron.name, self.makeNeg(maxNeuron.name)]))

            enc += '\n' + self.makeLt(self.makeMult(str(diff.hi), out.name), diff.name)
            sum = self.makeSum([str(diff.lo), self.makeNeg(self.makeMult(str(diff.lo), out.name))])
            enc += '\n' + self.makeGeq(diff.name, sum)

        self.vars.append(maxVars)
        self.vars.append(diffNeurons)
        self.vars.append(outNeurons)

        return '# --- one hot layer constraints ---' + maxEnc + diffConstraints + enc


    def encodeNNReadableFixed(self):
        # encode simple one layer NN with relu function
        inputs = [1,2]
        weights = [[1,4],[2,5],[3,6]]

        self.encodeInputsReadable(inputs, inputs)
        linearEnc = self.encodeLinearLayer(weights, 2, 1)
        reluEnc = self.encodeActivationLayer(2, 1, self.encodeRelu)

        preamble = self.makePreambleReadable()
        suffix = self.makeSuffix()

        return preamble + '\n' + linearEnc + '\n' + reluEnc + '\n' + suffix

