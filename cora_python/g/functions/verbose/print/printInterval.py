from cora_python.g.functions.matlab.validate.postprocessing.CORAwarning import CORAwarning
from cora_python.contSet.contSet.printSet import printSet

def printInterval(I):
    """
    prints an interval such that if one executes this command
    in the workspace, this interval would be created
    """

    CORAwarning("CORA:deprecated", "function", "printInterval(I)", "CORA v2025.0.2", "Please use printSet(I) instead.", "")
    printSet(I) 