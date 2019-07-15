from variable import *
import nn_loader


# flatten list of lists
def flatten(list):
    return [x for sublist in list for x in sublist]


class NNEncoder:

    def __init__(self, file):
        self.file = file
        # list of list of vars
        # last list always contains output of last layer
        self.vars = []

    # only use strings in args for make... methods
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
        # maybe switch to other representation later
        return self.makeLeq(rhs, lhs)

    def makeEq(self, lhs, rhs):
        return '(assert (= ' + lhs + ' ' + rhs + '))'

    def makeLt(self, lhs, rhs):
        return '(assert (< ' + lhs + ' ' + rhs + '))'

    def makeGt(self, lhs, rhs):
        return self.makeLt(rhs, lhs)

    def encodeInputsReadable(self, lowerBounds, upperBounds, netPrefix):
        if not (len(lowerBounds) == len(upperBounds)):
            raise IOError('lowerBounds and upperBounds need to match the number of inputs')

        # no constraints printed here. Later for all vars lower and upper bound constraints are added
        inputVars = []
        i = 0
        for lo, hi in zip(lowerBounds, upperBounds):
            var = Variable(0, i, netPrefix, 'i')
            var.setLo(lo)
            var.setHi(hi)
            inputVars.append(var)
            i += 1

        self.vars.append(inputVars)

    # weights is matrix where one column holds the weights for a neuron.
    # Bias for that neuron is the last entry in that column
    # numNeurons is the number of neurons in that layer
    def encodeLinearLayer(self, weights, numNeurons, layerIndex, netPrefix):
        enc = '; --- linear constraints layer ' + str(layerIndex) + ' ---'
        prevNeurons = self.vars[-1]
        prevNum = len(prevNeurons)
        currentNeurons = []
        for i in range(0, numNeurons):
            var = Variable(layerIndex, i, netPrefix, 'x')
            currentNeurons.append(var)
            terms = [self.makeMult(str(weights[row][i]), prevNeurons[row].name) for row in range(0, prevNum)]
            terms.append(str(weights[-1][i]))
            enc += '\n' + self.makeEq(var.name, self.makeSum(terms))

        self.vars.append(currentNeurons)

        return enc

    # maxpool function, takes list of inputNeurons and output neuron.
    # can't be used as activationEncoder here, because list of input neurons
    # returns (enc, intermediateVars)
    def encodeMaxPoolReadable(self, inNeurons, outNeuron, netPrefix):
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
            maxVarA = Variable(outNeuron.layer, outNeuron.row, netPrefix, outNeuron.name + 'a')
            maxVarB = Variable(outNeuron.layer, outNeuron.row, netPrefix, outNeuron.name + 'b')
            enc1, vars1 = self.encodeMaxPoolReadable(inNeurons[:num // 2], maxVarA)
            enc2, vars2 = self.encodeMaxPoolReadable(inNeurons[num // 2:], maxVarB)

            enc += enc1 + '\n' + enc2
            vars.append(vars1)
            vars.append(vars2)

        delta = Variable(outNeuron.layer, outNeuron.row, netPrefix, outNeuron.name + 'd', 'Int')
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
    def encodeRelu(self, inNeuron, outNeuron, netPrefix):
        enc = ''
        layerIndex = inNeuron.layer
        rowIndex = inNeuron.row

        delta = Variable(layerIndex, rowIndex, netPrefix, 'd', 'Int')
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
    def encodeActivationLayer(self, numNeurons, layerIndex, netPrefix, activatonEncoder):
        enc = '; --- activation constrainst layer ' + str(layerIndex) + ' ---'
        sumNeurons = self.vars[-1]
        deltas = []
        outNeurons = []
        for i in range(0, numNeurons):
            sn = sumNeurons[i]
            out = Variable(layerIndex, i, netPrefix, 'o')
            outNeurons.append(out)

            activatonEncoded, intermediateVars = activatonEncoder(sn, out, netPrefix)
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
                decls += '\n' + '(declare-const ' + var.name + ' ' + var.type + ')'
                if var.hasHi:
                    bounds += '\n' + self.makeLeq(var.name, str(var.hi))
                if var.hasLo:
                    bounds += '\n' + self.makeGeq(var.name, str(var.lo))

        return preamble + decls + '\n; ---- Bounds ----' + bounds

    def makeSuffix(self):
        return '(check-sat)\n(get-model)'

    # encodes all interior layers (not input layer)
    def encodeAllLayers(self, layers, netPrefix='', withOneHot=False):
        # assumes input has been encoded before (inputs are at self.vars[-1])
        index = 0
        layersEnc = '; --- Encoding of layers ---'
        for activation, numNeurons, weights in layers:
            index += 1
            linearEnc = self.encodeLinearLayer(weights, numNeurons, index, netPrefix)
            # only for ReLU and linear for now, need other function encoders for different activation functions
            layersEnc += '\n' + linearEnc
            if activation == 'relu':
                activationEnc = '\n' + self.encodeActivationLayer(numNeurons, index, netPrefix, self.encodeRelu)
            elif activation == 'linear':
                # only here for completeness, no activation function applied in linear layer
                pass

            # make sure activationEnc is with leading '\n'
            layersEnc += activationEnc

        if withOneHot:
            layersEnc += '\n' + self.encodeOneHotLayerReadable(len(layers) + 1, netPrefix)

        return layersEnc

    def encodeNN(self, layers, input_lowerBounds, input_upperBounds, withOneHot=False):
        self.encodeInputsReadable(input_lowerBounds, input_upperBounds, '')
        layersEnc = self.encodeAllLayers(layers, '', withOneHot)

        preamble = self.makePreambleReadable()
        suffix = self.makeSuffix()

        return preamble + '\n' + layersEnc + '\n' + suffix

    def encodeOneHotLayerReadable(self, layerIndex, netPrefix=''):
        inNeurons = self.vars[-1]
        maxNeuron = Variable(layerIndex, 0, netPrefix, 'max')
        maxEnc, maxVars = self.encodeMaxPoolReadable(inNeurons, maxNeuron, netPrefix)

        outNeurons = []
        diffNeurons = []
        diffConstraints = ''
        enc = ''
        for i in range(0, len(inNeurons)):
            out = Variable(layerIndex + 1, i, netPrefix, 'o', 'Int')
            out.setLo(0)
            out.setHi(1)
            outNeurons.append(out)

            inNeuron = inNeurons[i]

            diff = Variable(layerIndex + 1, i, netPrefix, 'x')
            diffNeurons.append(diff)
            diffConstraints += '\n' + self.makeEq(diff.name,
                                                  self.makeSum([inNeuron.name, self.makeNeg(maxNeuron.name)]))

            enc += '\n' + self.makeGt(self.makeMult(str(diff.hi), out.name), diff.name)
            sum = self.makeSum([str(diff.lo), self.makeNeg(self.makeMult(str(diff.lo), out.name))])
            enc += '\n' + self.makeGeq(diff.name, sum)

        self.vars.append(maxVars)
        self.vars.append([maxNeuron])
        self.vars.append(diffNeurons)
        self.vars.append(outNeurons)

        return '; --- one hot layer constraints ---' + maxEnc + diffConstraints + enc

    def encodeNNReadableFixed(self, withOneHot=False):
        # encode simple one layer NN with relu function
        inputs = [1, 2]
        weights = [[1, 4], [2, 5], [3, 6]]

        netPrefix = ''
        enc = ''
        self.encodeInputsReadable(inputs, inputs, netPrefix)
        enc += self.encodeLinearLayer(weights, 2, 1, netPrefix)
        enc += '\n' + self.encodeActivationLayer(2, 1, netPrefix, self.encodeRelu)

        if withOneHot:
            enc += '\n' + self.encodeOneHotLayerReadable(2)

        preamble = self.makePreambleReadable()
        suffix = self.makeSuffix()

        return preamble + '\n' + enc + '\n' + suffix


    def encodeEquivalence(self, layers1, layers2, input_lowerBounds, input_upperBounds, withOneHot=False):
        self.encodeInputsReadable(input_lowerBounds, input_upperBounds, 'I')
        inputVars = self.vars[-1]

        encNN1 = self.encodeAllLayers(layers1, 'A', withOneHot)
        nn1Outs = self.vars[-1]

        # only need to encode input vars once for both nets,
        # remember position in list, so we can delete duplicate later
        lengthNN1 = len(self.vars)
        self.vars.append(inputVars)

        encNN2 = self.encodeAllLayers(layers2, 'B', withOneHot)
        nn2Outs = self.vars[-1]

        # remove duplicate input vars
        del self.vars[lengthNN1]

        if not len(nn1Outs) == len(nn2Outs):
             raise IOError('only NNs with equal number of outputs can be equivalent')

        eqConstraints = '; --- Equality Constraints --- '
        u = 99999
        l = -99999
        deltas = []
        for out1, out2 in zip(nn1Outs, nn2Outs):
            # out1 - out2 should be 0, if they are equal
            diff = self.makeSum([out1.name, self.makeNeg(out2.name)])

            deltaG0 = Variable(0, out1.row, 'E', 'dG0', 'Int')
            deltaG0.setLo(0)
            deltaG0.setHi(1)
            deltas.append(deltaG0)

            eqConstraints += '\n' + self.makeLeq(diff, self.makeMult(str(u), deltaG0.name))
            eqConstraints += '\n' + self.makeGt(diff, self.makeSum([str(l), self.makeNeg(self.makeMult(str(l), deltaG0.name))]))

            deltaL0 = Variable(0, out1.row, 'E', 'dL0', 'Int')
            deltaL0.setLo(0)
            deltaL0.setHi(1)
            deltas.append(deltaL0)

            eqConstraints += '\n' + self.makeLt(diff, self.makeSum([str(u), self.makeNeg(self.makeMult(str(u), deltaL0.name))]))
            eqConstraints += '\n' + self.makeGeq(diff, self.makeMult(str(l), deltaL0.name))

        # at least one of the not-equals should be true
        eqConstraints += '\n' + self.makeGeq(self.makeSum([delta.name for delta in deltas]), str(1))

        self.vars.append(deltas)

        preamble = self.makePreambleReadable()
        suffix = self.makeSuffix()

        return preamble + '\n' + encNN1 + '\n' + encNN2 + '\n' + eqConstraints + '\n' + suffix
