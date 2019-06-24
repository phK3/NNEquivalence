
import numpy as np
import NNFileAccess

def futile(x):
    print(x)

class NNEncoder:

    def __init__(self, file, freeVar):
        self.file = file
        self.minFreeVar = freeVar
        self.intVars = []

    def getNewVar(self):
        res = self.minFreeVar
        self.minFreeVar += 1
        return res

    def getNewVars(self, num):
        res = self.minFreeVar
        self.minFreeVar += num
        return res

    def encodeLeq(self, sum, constant):
        return '(assert (<= ' + sum + ' ' + constant + '))'

    def encodeGeq(self, sum, constant):
        return '(assert (<= (- ' + sum + ') ' + '(- ' + constant + ')))'

    #don't want >, but encoding for one-hot doesn't work right now
    def encodeGt(self, sum, constant, mode='strict'):
        if mode == 'strict':
            res = '(assert (< (- ' + sum + ') ' + '(- ' + constant + ')))'
        else:
            res = '(assert (> ' + sum + ' ' + ' ' + constant + '))'
        return res

    def encodeLt(self, sum, constant):
        return '(assert (< ' + sum + ' ' + ' ' + constant + '))'

    def encodeEq(self, sum, constant, mode='strict'):
        if mode == 'strict':
            res = self.encodeLeq(sum, constant) + '\n' + self.encodeGeq(sum, constant)
        elif mode == 'readable':
            res = '(assert (= ' + sum + ' ' + constant + '))'

        return res

    #def encodeNN(self, min):
    #   fa = NNFileAccess(self.file)

    def encodeInputs(self, numInputs, lowerBounds, upperBounds):
        min = self.getNewVars(numInputs)
        vars = range(min, min + numInputs)
        constraints = ''
        for i, lo, hi in zip(vars, lowerBounds, upperBounds):
            constraints += (self.encodeLeq('x' + str(i), str(hi))) + '\n'
            constraints += (self.encodeGeq('x' + str(i), str(lo))) + '\n'

        return constraints

    def getOutPrevLayer(self, min, numNeuronsPrev):
        #returns an array of the var-numbers of the outputs of the previous layer
        return np.arange(min - numNeuronsPrev, min)

    def encodeMultiplication(self, constant, factor):
        return '(* ' + str(constant) + ' x' + str(factor) + ')'

    #not pretty, but i need c^T*x - a == 0 later, so included - a
    #also adds in bias, which should be in neuronsVec.size location of weightsVec
    def encodeDotProductAndVar(self, weightsVec, neuronsVec, var):
        sumTerm = '(+ '
        for i in range(0, neuronsVec.size):
            sumTerm += self.encodeMultiplication(weightsVec[i], neuronsVec[i]) + ' '
        sumTerm += str(weightsVec[neuronsVec.size]) + ' '
        sumTerm += '(- x' + str(var) + '))'
        return sumTerm

    def getOutNeuron(self, neuron, numNeurons):
        return neuron + 2*numNeurons

    def getDelta(self, neuron, numNeurons):
        return neuron + numNeurons

    #later return better M
    def getM(self, layer, row):
        return 99999

    # numNeuronsPrev with bias, as is weights matrix (last row are biases)
    def encodeLayer(self, index, numNeurons, numNeuronsPrev, weights, activation='relu', mode='strict'):
        min = self.getNewVars(3*numNeurons)

        constraints = ''

        prevNeuronsVec = self.getOutPrevLayer(min, numNeuronsPrev)

        #why does the linebreak character not work? -> only works with print not with direct output
        #constraints for intermediate vars
        #neuron is var-id of neuron, matrix needs relative neuron number
        for neuron in range(min, min + numNeurons):
            sumTerm = self.encodeDotProductAndVar(weights[:,neuron - min], prevNeuronsVec, neuron)
            constraints += self.encodeEq(sumTerm, str(0), mode) + "\n"

            # TODO: factor out activation function encoding
            if activation == 'relu':
                interNeuronStr = 'x' + str(neuron)
                outNeuronStr = 'x' + str(self.getOutNeuron(neuron, numNeurons))
                mStr = str(self.getM(index, neuron - min))
                delta = self.getDelta(neuron, numNeurons)
                deltaM = self.encodeMultiplication(self.getM(index, neuron - min), self.getDelta(neuron, numNeurons))

                #deltas are integer vars
                self.intVars.append(delta)

                #0 <= delta <= 1
                constraints += (self.encodeLeq('x' + str(delta), str(1))) + '\n'
                constraints += (self.encodeGeq('x' + str(delta), str(0))) + '\n'

                maxConstraint = self.encodeGeq(outNeuronStr, str(0)) + '\n'
                maxConstraint += self.encodeGeq(outNeuronStr, interNeuronStr)

                activeConstraint = self.encodeLeq('(+ ' + interNeuronStr + ' (- ' + deltaM + '))', str(0)) + '\n'
                activeConstraint += self.encodeGeq('(+ ' + interNeuronStr + ' ' + mStr + ' ' + '(- ' + deltaM + '))', str(0))

                outputConstraint = self.encodeGeq('(+ ' + interNeuronStr + ' ' + mStr + ' ' + '(- ' + deltaM + ') ' + '( - ' + outNeuronStr + '))', str(0)) + '\n'
                outputConstraint += self.encodeGeq('(+ ' + deltaM + ' (- ' + outNeuronStr + '))', str(0))

                constraints += maxConstraint + '\n' + activeConstraint + '\n' + outputConstraint + '\n'

        return constraints

    def makePreamble(self):
        return '(set-option :produce-models true)\n(set-logic AUFLIRA)'

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

    def encodeNNPreamble(self, numLayers, numInputs, layers, input_lowerBounds, input_upperBounds, mode='strict'):
        encNN = self.encodeNN(numLayers, numInputs, layers, input_lowerBounds, input_upperBounds, mode)
        return self.makePreamble() + '\n' + encNN + '\n' + self.makeSuffix()

    def encodeMaxPool(self, inputVars, outputVar, upperBounds, mode):
        enc = ''
        inSize = len(inputVars)
        if inSize > 2:
            interOutVar = self.getNewVar()
            enc = self.encodeMaxPool(inputVars[:inSize//2], interOutVar, upperBounds[:inSize//2], mode) \
                  + self.encodeMaxPool(inputVars[inSize//2:], interOutVar, upperBounds[inSize//2:], mode)
        elif inSize == 2:
            delta = self.getNewVar()
            deltaStr = 'x' + str(delta)
            inStr1 = 'x' + str(inputVars[0])
            inStr2 = 'x' + str(inputVars[0])
            outStr = 'x' + str(outputVar)
            upperBound = max(upperBounds[0], upperBounds[1])
            enc += self.encodeLeq(inStr1, outStr) + '\n'
            enc += self.encodeLeq(inStr2, outStr) + '\n'
            enc += self.encodeLeq(outStr, '(+ ' + inStr1 + ' ' + self.encodeMultiplication(upperBound, delta) + ' )') + '\n'
            enc += self.encodeLeq(outStr, '(+ ' + inStr2 + ' ' + str(upperBound) +
                                  ' (- ' + self.encodeMultiplication(upperBound, delta) + '))') + '\n'
            enc += self.encodeLeq(deltaStr, str(1)) + '\n'
            enc += self.encodeGeq(deltaStr, str(0))
            self.intVars.append(delta)
        elif inSize == 1:
            enc += self.encodeEq('x' + str(inputVars[0]), 'x' + str(outputVar), mode)

        return enc

    def encodeOneHotLayer(self, outputVars, upperBounds, mode):
        maxVar = self.getNewVar()

        maxConstraints = self.encodeMaxPool(outputVars, maxVar, upperBounds, mode)

        minFree = self.getNewVars(len(outputVars))
        minDelta = self.getNewVars(len(outputVars))

        stepConstraints = ''
        m = max(upperBounds)
        interVar = minFree
        delta = minDelta
        for x in outputVars:
            interVarStr = 'x' + str(interVar)
            deltaStr = 'x' + str(delta)
            #x_i = x_o - x_max
            stepConstraints += self.encodeEq(interVarStr, '(+ x' + str(x) + ' (- x' + str(maxVar) + '))') + '\n'
            stepConstraints += self.encodeLeq(interVarStr, '0') + '\n'

            #x_i + m - delta*m
            stepExpression = '(+ ' + interVarStr + ' ' + str(m) + ' (- ' + self.encodeMultiplication(m, delta) + '))'
            stepConstraints += self.encodeGeq(stepExpression, '0') + '\n'
            stepConstraints += self.encodeLt(stepExpression, str(m)) + '\n'

            stepConstraints += self.encodeLeq(deltaStr, str(1)) + '\n'
            stepConstraints += self.encodeGeq(deltaStr, str(0)) + '\n'
            self.intVars.append(delta)

            interVar += 1
            delta += 1

        return maxConstraints + '\n' + stepConstraints

    def encodeToOneHot(self, net, mode):
        nnEnc = self.encodeNN(net.numLayers, net.numInputs, net.layers, net.input_lowerBounds,
                      net.input_upperBounds, mode)
        _, numOutputs, _ = net.layers[-1]
        outputVars = range(self.minFreeVar - numOutputs, self.minFreeVar)
        upperBounds = [self.getM(net.numLayers - 1, i) for i in range(0, numOutputs)]

        oneHotEnc = self.encodeOneHot(outputVars, upperBounds, mode)
        firstOneHotOut = self.minFreeVar - numOutputs

        return (firstOneHotOut, numOutputs, oneHotEnc)

    def encodeEquivalence(self, net1, net2, mode):
        # remove preamble, etc from encodeNN, factor out
        nn1FirstOut, nn1NumOut, nn1Enc = self.encodeToOneHot(net1.numLayers, net1.numInputs,
                                                             net1.layers, net1.input_lowerBounds, net1.input_upperBounds, mode)

        nn2FirstOut, nn2NumOut, nn2Enc = self.encodeToOneHot(net2.numLayers, net2.numInputs, net2.layers, net2.input_lowerBounds,
                               net2.input_upperBounds, mode)

        eqConstraint = ''
        for i, j in zip(range(nn1FirstOut, nn1FirstOut + nn1NumOut), range(nn2FirstOut, nn2FirstOut + nn2NumOut)):
            eqConstraint += self.encodeEq('x' + str(i), 'x' + str(j), mode) + '\n'

        return nn1Enc + '\n' + nn2Enc + '\n' + eqConstraint

    def encodeEquivPreamble(self, net1, net2, mode='strict'):
        enc = self.encodeEquivalence(net1, net2, mode)
        return self.makePreamble() + '\n' + enc + '\n' + self.makeSuffix()


    def encodeNNFixed(self, numLayers, numInputs, layers):
        inputEnc = self.encodeInputs(2, [0,2], [0,2])

        weights1 = np.array([[1,2], [3,4], [5,6]])

        layersEnc1 = self.encodeLayer(1, 2,2,weights1, activation='relu', mode='readable')

        weights2 = np.array([[5,6], [7,8], [9,10]])

        layersEnc2 = self.encodeLayer(2, 2, 2, weights2, activation='none', mode='readable')

        varDecls = ''
        varsUsed = self.getNewVar()
        for i in range(0, varsUsed):
            if i in self.intVars:
                varDecls += '(declare-const x' + str(i) + ' Int)\n'
            else:
                varDecls += '(declare-const x' + str(i) + ' Real)\n'

        preamble = '(set-option :produce-models true)\n(set-logic AUFLIRA)'
        suffix = '(check-sat)\n(get-model)'

        return preamble + '\n' + varDecls + '\n' + inputEnc + '\n' + layersEnc1 + '\n' + layersEnc2 + '\n' + suffix


