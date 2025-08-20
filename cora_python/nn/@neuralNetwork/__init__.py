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

# Import the main class to attach methods to
from ..neuralNetwork import NeuralNetwork

# Attach all methods to the class
NeuralNetwork.evaluate_ = evaluate_
NeuralNetwork.calcSensitivity = calcSensitivity
NeuralNetwork.refine = refine
NeuralNetwork.verify = verify
NeuralNetwork.explain = explain
NeuralNetwork.getRefinableLayers = getRefinableLayers
NeuralNetwork.getInputNeuronOrder = getInputNeuronOrder

# Export the class with attached methods
__all__ = ['NeuralNetwork']
