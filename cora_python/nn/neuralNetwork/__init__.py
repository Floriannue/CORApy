"""
@neuralNetwork - Method attachments for NeuralNetwork class

This module attaches all the individual method files to the NeuralNetwork class.
"""

from .evaluate_ import evaluate_
from .calcSensitivity import calcSensitivity
from .refine import refine
from .verify import verify
from .explain import explain
from .getRefinableLayers import getRefinableLayers
from .getInputNeuronOrder import getInputNeuronOrder
from .readONNXNetwork import readONNXNetwork
from .convertDLToolboxNetwork import convertDLToolboxNetwork
from .castWeights import castWeights
from .prepareForZonoBatchEval import prepareForZonoBatchEval
from .evaluate import evaluate
from .evaluateZonotopeBatch import evaluateZonotopeBatch
from .evaluateZonotopeBatch_ import evaluateZonotopeBatch_
from .backprop import backprop
from .train import train
from .initWeights import initWeights
from .getNumNeurons import getNumNeurons
from .setInputSize import setInputSize
from .display import display
from .reset import reset
from .resetApproxOrder import resetApproxOrder
from .resetBounds import resetBounds
from .resetGNN import resetGNN
from .readNetwork import readNetwork
from .generateRandom import generateRandom
from .exportAsJSON import exportAsJSON
from .exportAsStruct import exportAsStruct
from .visLossLandscape import visLossLandscape
from .copyNeuralNetwork import copyNeuralNetwork
from .importFromJSON import importFromJSON
from .importFromStruct import importFromStruct
from .getNormalForm import getNormalForm
from .reduceGNNForNode import reduceGNNForNode
from .getNumMessagePassingSteps import getNumMessagePassingSteps
from .getOrderPattern import getOrderPattern
from .readSherlockNetwork import readSherlockNetwork
from .readYMLNetwork import readYMLNetwork
from .getFromCellArray import getFromCellArray
from .propagateBounds import propagateBounds


# Import the main class to attach methods to
from .neuralNetwork import NeuralNetwork

# Attach all methods to the class
NeuralNetwork.evaluate_ = evaluate_
NeuralNetwork.calcSensitivity = calcSensitivity
NeuralNetwork.refine = refine
NeuralNetwork.verify = verify
NeuralNetwork.explain = explain
NeuralNetwork.getRefinableLayers = getRefinableLayers
NeuralNetwork.getInputNeuronOrder = getInputNeuronOrder
NeuralNetwork.readONNXNetwork = staticmethod(readONNXNetwork)
NeuralNetwork.convertDLToolboxNetwork = staticmethod(convertDLToolboxNetwork)

# Attach all the new methods
NeuralNetwork.castWeights = castWeights
NeuralNetwork.prepareForZonoBatchEval = prepareForZonoBatchEval
NeuralNetwork.evaluate = evaluate
NeuralNetwork.evaluateZonotopeBatch = evaluateZonotopeBatch
NeuralNetwork.evaluateZonotopeBatch_ = evaluateZonotopeBatch_
NeuralNetwork.backprop = backprop
NeuralNetwork.train = train
NeuralNetwork.initWeights = initWeights
NeuralNetwork.getNumNeurons = getNumNeurons
NeuralNetwork.setInputSize = setInputSize
NeuralNetwork.display = display
NeuralNetwork.reset = reset
NeuralNetwork.resetApproxOrder = resetApproxOrder
NeuralNetwork.resetBounds = resetBounds
NeuralNetwork.resetGNN = resetGNN
NeuralNetwork.readNetwork = staticmethod(readNetwork)
NeuralNetwork.generateRandom = staticmethod(generateRandom)
NeuralNetwork.exportAsJSON = exportAsJSON
NeuralNetwork.exportAsStruct = exportAsStruct
NeuralNetwork.visLossLandscape = visLossLandscape
NeuralNetwork.copyNeuralNetwork = copyNeuralNetwork
NeuralNetwork.importFromJSON = staticmethod(importFromJSON)
NeuralNetwork.importFromStruct = staticmethod(importFromStruct)
NeuralNetwork.getNormalForm = staticmethod(getNormalForm)
NeuralNetwork.reduceGNNForNode = reduceGNNForNode
NeuralNetwork.getNumMessagePassingSteps = getNumMessagePassingSteps
NeuralNetwork.getOrderPattern = getOrderPattern
NeuralNetwork.readSherlockNetwork = staticmethod(readSherlockNetwork)
NeuralNetwork.readYMLNetwork = staticmethod(readYMLNetwork)
NeuralNetwork.getFromCellArray = staticmethod(getFromCellArray)
NeuralNetwork.propagateBounds = propagateBounds


# Export the class with attached methods
__all__ = ['NeuralNetwork']
