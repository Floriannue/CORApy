from cora_python.g.functions.matlab.validate.postprocessing.CORAerror import CORAerror

def getPrintSetInfo(self):
    """
    returns all information to properly print a set 
    to the command window 

    Syntax:
        [abbrev,propertyOrder] = getPrintSetInfo(S)

    Inputs:
        S - contSet

    Outputs:
        abbrev - set abbreviation
        propertyOrder - order of the properties
    """

    raise CORAerror('CORA:noops', self) 